#!/usr/bin/env python3
"""
Binance USD-M Futures (Testnet/Mainnet) bot implementing the user's strategy EXACTLY:

- Trade futures only.
- After each daily candle closes:
    - Place TWO conditional entry orders:
        1) BUY STOP_MARKET at prior day's High (breakout)
        2) SELL STOP_MARKET at prior day's Low (breakdown)
    - If an order is not hit for that day, it must be deleted at the next daily close and replaced.
    - If one order is executed, the other stays waiting; if not activated by next day, it is deleted and replaced with new day orders.

- TP = 1200 points from entry price.
- SL = 130 points from entry price.
- SL trailing:
    - When profit reaches +26 points, move SL to +4 points into profit (break-even-ish).
    - Then SL trails by 30 points behind best price.

CRITICAL Binance API note (Dec 9, 2025+):
Conditional order types (STOP_MARKET / TAKE_PROFIT_MARKET / STOP / TAKE_PROFIT / TRAILING_STOP_MARKET)
must be sent to POST /fapi/v1/algoOrder with algoType=CONDITIONAL and triggerPrice (NOT /fapi/v1/order).

Docs:
- New Algo Order: POST /fapi/v1/algoOrder uses triggerPrice.
- Cancel All Algo Open Orders: DELETE /fapi/v1/algoOpenOrders
- Open Orders: GET /fapi/v1/openOrders
- Kline stream includes k.x (is closed) for daily candle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import time
import hmac
import hashlib
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode

import requests
import websockets
from dotenv import load_dotenv


# -------------------- CONFIG --------------------

@dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    quantity: float

    tp_points: float = 1200.0
    sl_points: float = 130.0

    trail_start_points: float = 26.0
    breakeven_points: float = 4.0
    trail_distance_points: float = 30.0

    working_type: str = "MARK_PRICE"  # trigger based on MARK_PRICE or CONTRACT_PRICE
    recv_window: int = 5000


@dataclass
class SymbolFilters:
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: Optional[float]


# -------------------- REST CLIENT --------------------

class BinanceUMFuturesREST:
    def __init__(self, api_key: str, api_secret: str, base_url: str, recv_window: int = 5000):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def signed(self, method: str, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        """
        Build and sign the querystring ourselves, then send as a fully-built URL.
        This avoids signature mismatches from parameter re-encoding/re-ordering.
        """
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params.setdefault("recvWindow", self.recv_window)

        qs = urlencode(params, doseq=True)
        sig = self._sign(qs)
        url = f"{self.base_url}{path}?{qs}&signature={sig}"

        resp = self.session.request(method, url)

        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise RuntimeError(f"HTTP {resp.status_code} {method} {path} error body: {body}")

        return resp.json()

    # ---- public endpoints (no signature) ----
    def exchange_info(self, symbol: str) -> Any:
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        resp = self.session.get(url, params={"symbol": symbol})
        resp.raise_for_status()
        return resp.json()

    def klines(self, symbol: str, interval: str, limit: int = 2) -> Any:
        url = f"{self.base_url}/fapi/v1/klines"
        resp = self.session.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
        resp.raise_for_status()
        return resp.json()

    # ---- signed endpoints ----
    def cancel_all_open_orders(self, symbol: str) -> Any:
        return self.signed("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})

    def cancel_all_algo_open_orders(self, symbol: str) -> Any:
        return self.signed("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol})

    def new_algo_order(self, params: dict[str, Any]) -> Any:
        return self.signed("POST", "/fapi/v1/algoOrder", params)

    def query_algo_order(self, algo_id: int) -> Any:
        return self.signed("GET", "/fapi/v1/algoOrder", {"algoId": algo_id})

    def cancel_algo_order(self, algo_id: int) -> Any:
        return self.signed("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id})

    def position_risk(self, symbol: str) -> Any:
        return self.signed("GET", "/fapi/v3/positionRisk", {"symbol": symbol})


# -------------------- UTIL: ROUNDING --------------------

def floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step

def round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    # choose decimals based on tick
    decimals = max(0, int(round(-math.log10(tick), 0)) + 2) if tick < 1 else 2
    return round(round(x / tick) * tick, ndigits=decimals)

def fmt_num(x: float) -> str:
    return f"{x:.12f}".rstrip("0").rstrip(".")


# -------------------- BOT --------------------

class BreakoutBot:
    def __init__(self, cfg: StrategyConfig, rest: BinanceUMFuturesREST, ws_base: str):
        self.cfg = cfg
        self.rest = rest
        self.ws_base = ws_base.rstrip("/")

        self.filters: Optional[SymbolFilters] = None

        self._stop = asyncio.Event()
        self._last_daily_close_ms: Optional[int] = None

        # last mark price
        self._mark_price: Optional[float] = None

        # trailing state
        self._long_peak: Optional[float] = None
        self._short_trough: Optional[float] = None
        self._long_be_armed: bool = False
        self._short_be_armed: bool = False

        # track SL algo IDs for replace-on-trail
        self._long_sl_algo_id: Optional[int] = None
        self._short_sl_algo_id: Optional[int] = None

        self.log = logging.getLogger("bot")

    async def run(self) -> None:
        await self._load_symbol_filters()
        await self._startup_rotate_from_last_closed_daily()

        symbol_lower = self.cfg.symbol.lower()
        streams = f"{symbol_lower}@kline_1d/{symbol_lower}@markPrice@1s"
        url = f"{self.ws_base}/stream?streams={streams}"

        self.log.info("WebSocket: %s", url)

        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        await self._on_ws_message(raw)
            except Exception as e:
                self.log.exception("WS error (reconnecting): %s", e)
                await asyncio.sleep(2)

    async def stop(self) -> None:
        self._stop.set()

    # ---------- startup ----------

    async def _load_symbol_filters(self) -> None:
        info = await asyncio.to_thread(self.rest.exchange_info, self.cfg.symbol)
        sym = info["symbols"][0]

        tick_size = step_size = min_qty = None
        min_notional = None

        for f in sym.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
            elif f.get("filterType") == "LOT_SIZE":
                step_size = float(f["stepSize"])
                min_qty = float(f["minQty"])
            elif f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                try:
                    min_notional = float(f.get("notional", f.get("minNotional")))
                except Exception:
                    min_notional = None

        if tick_size is None or step_size is None or min_qty is None:
            raise RuntimeError(f"Failed to parse exchangeInfo filters: {sym.get('filters')}")

        self.filters = SymbolFilters(
            tick_size=tick_size,
            step_size=step_size,
            min_qty=min_qty,
            min_notional=min_notional,
        )

        self.log.info("Filters: tick=%s step=%s minQty=%s", tick_size, step_size, min_qty)

    async def _startup_rotate_from_last_closed_daily(self) -> None:
        kl = await asyncio.to_thread(self.rest.klines, self.cfg.symbol, "1d", 2)
        # Use prior closed day:
        # if we request 2 candles, kl[-1] is current (possibly open), kl[-2] is last fully closed.
        prior = kl[-2] if len(kl) >= 2 else kl[-1]
        prev_high = float(prior[2])
        prev_low = float(prior[3])
        close_time_ms = int(prior[6])

        self._last_daily_close_ms = close_time_ms
        await self.rotate_daily_orders(prev_high, prev_low, close_time_ms, reason="startup")

    # ---------- websocket handling ----------

    async def _on_ws_message(self, raw: str) -> None:
        msg = json.loads(raw)
        payload = msg.get("data", msg)

        et = payload.get("e")
        if et == "kline":
            k = payload.get("k", {})
            if k.get("i") == "1d" and bool(k.get("x")) is True:
                close_time_ms = int(k["T"])
                if self._last_daily_close_ms == close_time_ms:
                    return
                self._last_daily_close_ms = close_time_ms

                prev_high = float(k["h"])
                prev_low = float(k["l"])
                await self.rotate_daily_orders(prev_high, prev_low, close_time_ms, reason="daily_close")

        elif et == "markPriceUpdate":
            try:
                self._mark_price = float(payload["p"])
            except Exception:
                return
            await self._manage_positions_and_trailing()

    # ---------- daily rotation ----------

    async def rotate_daily_orders(self, prev_high: float, prev_low: float, close_time_ms: int, reason: str) -> None:
        self.log.info("Rotate (%s): prev_high=%s prev_low=%s close=%s", reason, prev_high, prev_low, close_time_ms)

        # Delete all old unfilled orders from prior day:
        # - regular open orders
        # - algo open orders (where conditional orders live now)
        try:
            await asyncio.to_thread(self.rest.cancel_all_open_orders, self.cfg.symbol)
        except Exception as e:
            self.log.warning("Cancel allOpenOrders failed: %s", e)

        try:
            await asyncio.to_thread(self.rest.cancel_all_algo_open_orders, self.cfg.symbol)
        except Exception as e:
            self.log.warning("Cancel algoOpenOrders failed: %s", e)

        # Place new daily entry orders immediately after daily candle close
        qty = self._rounded_qty(self.cfg.quantity)
        buy_trig = self._rounded_price(prev_high)
        sell_trig = self._rounded_price(prev_low)

        buy_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "BUY",
            "positionSide": "LONG",
            "type": "STOP_MARKET",
            "triggerPrice": fmt_num(buy_trig),
            "quantity": fmt_num(qty),
            "timeInForce": "GTC",
            "workingType": self.cfg.working_type,
            "newOrderRespType": "ACK",
        }
        sell_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "SELL",
            "positionSide": "SHORT",
            "type": "STOP_MARKET",
            "triggerPrice": fmt_num(sell_trig),
            "quantity": fmt_num(qty),
            "timeInForce": "GTC",
            "workingType": self.cfg.working_type,
            "newOrderRespType": "ACK",
        }

        buy_resp = await asyncio.to_thread(self.rest.new_algo_order, buy_params)
        sell_resp = await asyncio.to_thread(self.rest.new_algo_order, sell_params)

        self.log.info("Entry orders placed: buy_algoId=%s sell_algoId=%s",
                      buy_resp.get("algoId"), sell_resp.get("algoId"))

        # reset trailing trackers daily (positions are handled separately)
        self._long_peak = None
        self._short_trough = None
        self._long_be_armed = False
        self._short_be_armed = False
        self._long_sl_algo_id = None
        self._short_sl_algo_id = None

    # ---------- position management ----------

    async def _manage_positions_and_trailing(self) -> None:
        if self._mark_price is None:
            return

        try:
            positions = await asyncio.to_thread(self.rest.position_risk, self.cfg.symbol)
        except Exception as e:
            self.log.warning("positionRisk failed: %s", e)
            return

        for p in positions:
            if p.get("symbol") != self.cfg.symbol:
                continue

            side = p.get("positionSide")
            if side not in ("LONG", "SHORT"):
                continue

            amt = float(p.get("positionAmt", "0"))
            entry = float(p.get("entryPrice", "0"))
            if entry <= 0:
                continue

            if side == "LONG":
                if amt <= 0:
                    continue
                await self._long_logic(entry)
            else:
                if amt >= 0:
                    continue
                await self._short_logic(entry)

    # ---------- LONG ----------

    async def _long_logic(self, entry: float) -> None:
        mp = float(self._mark_price)

        if self._long_peak is None:
            self._long_peak = mp
        self._long_peak = max(self._long_peak, mp)

        # Ensure initial TP/SL exist once we detect a position
        if self._long_sl_algo_id is None:
            await self._place_long_tp_sl(entry)

        profit = mp - entry
        be_trigger = entry + self.cfg.breakeven_points

        if (not self._long_be_armed) and profit >= self.cfg.trail_start_points:
            self._long_be_armed = True
            await self._replace_long_sl(be_trigger)

        if self._long_be_armed:
            desired = max(be_trigger, self._long_peak - self.cfg.trail_distance_points)
            await self._replace_long_sl(desired)

    async def _place_long_tp_sl(self, entry: float) -> None:
        tp = self._rounded_price(entry + self.cfg.tp_points)
        sl = self._rounded_price(entry - self.cfg.sl_points)

        # TP (close position): SELL TAKE_PROFIT_MARKET triggers when price >= triggerPrice
        tp_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "SELL",
            "positionSide": "LONG",
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": fmt_num(tp),
            "closePosition": "true",
            "workingType": self.cfg.working_type,
        }
        # SL (close position): SELL STOP_MARKET triggers when price <= triggerPrice
        sl_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "SELL",
            "positionSide": "LONG",
            "type": "STOP_MARKET",
            "triggerPrice": fmt_num(sl),
            "closePosition": "true",
            "workingType": self.cfg.working_type,
        }

        await asyncio.to_thread(self.rest.new_algo_order, tp_params)
        sl_resp = await asyncio.to_thread(self.rest.new_algo_order, sl_params)
        self._long_sl_algo_id = sl_resp.get("algoId")

        self.log.info("LONG TP/SL placed: TP@%s SL@%s SL_algoId=%s", tp, sl, self._long_sl_algo_id)

    async def _replace_long_sl(self, new_trigger: float) -> None:
        new_trigger = self._rounded_price(new_trigger)

        if self._long_sl_algo_id is not None:
            try:
                cur = await asyncio.to_thread(self.rest.query_algo_order, int(self._long_sl_algo_id))
                cur_trig = float(cur.get("triggerPrice", "0") or "0")
                # only improve upwards
                if cur_trig >= new_trigger - self.filters.tick_size:
                    return
                await asyncio.to_thread(self.rest.cancel_algo_order, int(self._long_sl_algo_id))
            except Exception as e:
                self.log.debug("LONG SL query/cancel failed (continuing): %s", e)

        sl_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "SELL",
            "positionSide": "LONG",
            "type": "STOP_MARKET",
            "triggerPrice": fmt_num(new_trigger),
            "closePosition": "true",
            "workingType": self.cfg.working_type,
        }
        sl_resp = await asyncio.to_thread(self.rest.new_algo_order, sl_params)
        self._long_sl_algo_id = sl_resp.get("algoId")
        self.log.info("LONG SL updated -> %s (algoId=%s)", new_trigger, self._long_sl_algo_id)

    # ---------- SHORT ----------

    async def _short_logic(self, entry: float) -> None:
        mp = float(self._mark_price)

        if self._short_trough is None:
            self._short_trough = mp
        self._short_trough = min(self._short_trough, mp)

        if self._short_sl_algo_id is None:
            await self._place_short_tp_sl(entry)

        profit = entry - mp
        be_trigger = entry - self.cfg.breakeven_points

        if (not self._short_be_armed) and profit >= self.cfg.trail_start_points:
            self._short_be_armed = True
            await self._replace_short_sl(be_trigger)

        if self._short_be_armed:
            desired = min(be_trigger, self._short_trough + self.cfg.trail_distance_points)
            await self._replace_short_sl(desired)

    async def _place_short_tp_sl(self, entry: float) -> None:
        tp = self._rounded_price(entry - self.cfg.tp_points)
        sl = self._rounded_price(entry + self.cfg.sl_points)

        # TP (close position): BUY TAKE_PROFIT_MARKET triggers when price <= triggerPrice
        tp_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "BUY",
            "positionSide": "SHORT",
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": fmt_num(tp),
            "closePosition": "true",
            "workingType": self.cfg.working_type,
        }
        # SL (close position): BUY STOP_MARKET triggers when price >= triggerPrice
        sl_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "BUY",
            "positionSide": "SHORT",
            "type": "STOP_MARKET",
            "triggerPrice": fmt_num(sl),
            "closePosition": "true",
            "workingType": self.cfg.working_type,
        }

        await asyncio.to_thread(self.rest.new_algo_order, tp_params)
        sl_resp = await asyncio.to_thread(self.rest.new_algo_order, sl_params)
        self._short_sl_algo_id = sl_resp.get("algoId")

        self.log.info("SHORT TP/SL placed: TP@%s SL@%s SL_algoId=%s", tp, sl, self._short_sl_algo_id)

    async def _replace_short_sl(self, new_trigger: float) -> None:
        new_trigger = self._rounded_price(new_trigger)

        if self._short_sl_algo_id is not None:
            try:
                cur = await asyncio.to_thread(self.rest.query_algo_order, int(self._short_sl_algo_id))
                cur_trig = float(cur.get("triggerPrice", "0") or "0")
                # only improve downwards
                if cur_trig <= new_trigger + self.filters.tick_size:
                    return
                await asyncio.to_thread(self.rest.cancel_algo_order, int(self._short_sl_algo_id))
            except Exception as e:
                self.log.debug("SHORT SL query/cancel failed (continuing): %s", e)

        sl_params = {
            "algoType": "CONDITIONAL",
            "symbol": self.cfg.symbol,
            "side": "BUY",
            "positionSide": "SHORT",
            "type": "STOP_MARKET",
            "triggerPrice": fmt_num(new_trigger),
            "closePosition": "true",
            "workingType": self.cfg.working_type,
        }
        sl_resp = await asyncio.to_thread(self.rest.new_algo_order, sl_params)
        self._short_sl_algo_id = sl_resp.get("algoId")
        self.log.info("SHORT SL updated -> %s (algoId=%s)", new_trigger, self._short_sl_algo_id)

    # ---------- rounding ----------

    def _rounded_price(self, price: float) -> float:
        assert self.filters is not None
        return round_to_tick(price, self.filters.tick_size)

    def _rounded_qty(self, qty: float) -> float:
        assert self.filters is not None
        q = floor_to_step(qty, self.filters.step_size)
        if q < self.filters.min_qty:
            raise RuntimeError(f"Quantity {q} < minQty {self.filters.min_qty}")
        return q


# -------------------- MAIN --------------------

def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

def load_config() -> tuple[StrategyConfig, dict[str, str]]:
    load_dotenv()

    api_key = os.getenv("API_KEY", "").strip()
    api_secret = os.getenv("API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("Missing API_KEY/API_SECRET in environment (.env).")

    symbol = os.getenv("SYMBOL", "ETHUSDT").strip().upper()
    quantity = float(os.getenv("QUANTITY", "0.01"))

    rest_base = os.getenv("REST_BASE_URL", "https://demo-fapi.binance.com").strip()
    ws_base = os.getenv("WS_BASE_URL", "wss://fstream.binance.com").strip()

    cfg = StrategyConfig(
        symbol=symbol,
        quantity=quantity,
        tp_points=float(os.getenv("TP_POINTS", "1200")),
        sl_points=float(os.getenv("SL_POINTS", "130")),
        trail_start_points=float(os.getenv("TRAIL_START_POINTS", "26")),
        breakeven_points=float(os.getenv("BREAKEVEN_POINTS", "4")),
        trail_distance_points=float(os.getenv("TRAIL_DISTANCE_POINTS", "30")),
        working_type=os.getenv("WORKING_TYPE", "MARK_PRICE").strip(),
        recv_window=int(os.getenv("RECV_WINDOW", "5000")),
    )

    return cfg, {"API_KEY": api_key, "API_SECRET": api_secret, "REST_BASE_URL": rest_base, "WS_BASE_URL": ws_base}

async def main() -> None:
    setup_logging()
    cfg, env = load_config()

    rest = BinanceUMFuturesREST(env["API_KEY"], env["API_SECRET"], env["REST_BASE_URL"], recv_window=cfg.recv_window)
    bot = BreakoutBot(cfg, rest, env["WS_BASE_URL"])

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))
        except NotImplementedError:
            pass

    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
