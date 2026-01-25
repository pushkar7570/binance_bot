import os
import time
import hmac
import hashlib
import requests

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode


# ---------------------------
# Exceptions
# ---------------------------
class BinanceAPIError(Exception):
    def __init__(self, http_status: int, code: Any, msg: str, payload: Any):
        super().__init__(f"HTTP {http_status} error: {payload}")
        self.http_status = http_status
        self.code = code
        self.msg = msg
        self.payload = payload


# ---------------------------
# Binance REST client (USD-M Futures)
# ---------------------------
class BinanceFuturesREST:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        self.time_offset_ms = 0
        self.recv_window = 5000

    def sync_time(self) -> None:
        data = self._request("GET", "/fapi/v1/time", signed=False)
        server_time = int(data["serverTime"])
        local_time = int(time.time() * 1000)
        self.time_offset_ms = server_time - local_time

    def _timestamp(self) -> int:
        return int(time.time() * 1000 + self.time_offset_ms)

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p = {k: v for k, v in params.items() if v is not None}
        p.setdefault("recvWindow", self.recv_window)
        p["timestamp"] = self._timestamp()

        query = urlencode(p, doseq=True)
        sig = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        p["signature"] = sig
        return p

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        if params is None:
            params = {}

        if signed:
            params = self._sign_params(params)

        url = self.base_url + path
        resp = self.session.request(method, url, params=params, timeout=20)

        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            raise

        if resp.status_code != 200:
            raise BinanceAPIError(
                http_status=resp.status_code,
                code=data.get("code"),
                msg=str(data.get("msg", "")),
                payload=data,
            )
        return data

    # ---- Public endpoints ----
    def exchange_info(self) -> Any:
        return self._request("GET", "/fapi/v1/exchangeInfo", signed=False)

    def klines(self, symbol: str, interval: str, limit: int = 3) -> Any:
        return self._request(
            "GET",
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
            signed=False,
        )

    def ticker_price(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol}, signed=False)
        return float(data["price"])

    # ---- Account/trade endpoints ----
    def change_position_mode(self, hedge_mode: bool) -> Any:
        return self._request(
            "POST",
            "/fapi/v1/positionSide/dual",
            {"dualSidePosition": "true" if hedge_mode else "false"},
            signed=True,
        )

    def change_leverage(self, symbol: str, leverage: int) -> Any:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
            signed=True,
        )

    def position_risk(self, symbol: str) -> Any:
        return self._request("GET", "/fapi/v3/positionRisk", {"symbol": symbol}, signed=True)

    def new_market_order(self, symbol: str, side: str, position_side: str, quantity: str) -> Any:
        params = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
        }
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    # ---- Conditional (Algo) orders ----
    def open_algo_orders(self, symbol: str) -> Any:
        return self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol}, signed=True)

    def new_algo_order(self, params: Dict[str, Any]) -> Any:
        return self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

    def cancel_algo_order(self, algo_id: Optional[int] = None, client_algo_id: Optional[str] = None) -> Any:
        params = {"algoId": algo_id, "clientAlgoId": client_algo_id}
        return self._request("DELETE", "/fapi/v1/algoOrder", params, signed=True)


# ---------------------------
# Helpers (formatting to tick/step)
# ---------------------------
def format_to_step(value: float, step: Decimal, rounding) -> str:
    d = Decimal(str(value))
    q = (d / step).to_integral_value(rounding=rounding) * step
    q = q.quantize(step, rounding=rounding)
    return format(q, "f")


@dataclass
class SideState:
    pos_side: str          # "LONG" or "SHORT"
    qty: float

    entry_price: float = 0.0
    position_amt: float = 0.0

    tp_client_id: Optional[str] = None
    sl_client_id: Optional[str] = None
    sl_trigger: float = 0.0

    trailing_active: bool = False
    trail_best_price: float = 0.0
    trail_last_update_best: float = 0.0
    last_sl_replace_ms: int = 0


class PriorDayBreakoutBot:
    def __init__(self, client: BinanceFuturesREST, symbol: str):
        self.client = client
        self.symbol = symbol

        # === Strategy constants (exactly your numbers) ===
        self.qty = 10.0
        self.leverage = 20

        self.tp_offset = 48.0
        self.sl_offset = 5.2

        self.trail_profit_trigger = 0.52
        self.be_offset = 0.16
        self.trail_distance = 1.2

        # Symbol filters (loaded at startup)
        self.tick_size = Decimal("0.01")
        self.step_size = Decimal("0.001")

        self.last_processed_prev_close_time = 0

        self.long = SideState("LONG", qty=self.qty)
        self.short = SideState("SHORT", qty=self.qty)

    # ---------------------------
    # Startup / setup
    # ---------------------------
    def load_symbol_filters(self) -> None:
        info = self.client.exchange_info()
        sym = next(s for s in info["symbols"] if s["symbol"] == self.symbol)

        price_filter = next(f for f in sym["filters"] if f["filterType"] == "PRICE_FILTER")
        lot_filter = next(f for f in sym["filters"] if f["filterType"] == "LOT_SIZE")

        self.tick_size = Decimal(price_filter["tickSize"])
        self.step_size = Decimal(lot_filter["stepSize"])

        print(f"[filters] tickSize={self.tick_size}, stepSize={self.step_size}")

    def setup_account(self) -> None:
        # Hedge mode is required (your strategy keeps both long/short entries active)
        try:
            self.client.change_position_mode(hedge_mode=True)
            print("[setup] Hedge mode enabled")
        except BinanceAPIError as e:
            print(f"[setup] change_position_mode: {e.payload}")

        # Leverage - demo sometimes returns -1000; retry a few times and continue if it fails
        for i in range(3):
            try:
                self.client.change_leverage(self.symbol, self.leverage)
                print(f"[setup] Leverage set to {self.leverage}x")
                break
            except BinanceAPIError as e:
                print(f"[setup] change_leverage attempt {i+1}/3: {e.payload}")
                time.sleep(0.6)

    # ---------------------------
    # Daily levels
    # ---------------------------
    def get_prev_day_levels(self) -> Tuple[float, float, int]:
        kl = self.client.klines(self.symbol, "1d", limit=3)
        prev = kl[-2]  # previous closed daily candle
        prev_high = float(prev[2])
        prev_low = float(prev[3])
        prev_close_time = int(prev[6])
        return prev_high, prev_low, prev_close_time

    # ---------------------------
    # Entry order lifecycle (daily)
    # ---------------------------
    def cancel_all_entry_algo_orders(self) -> None:
        try:
            open_algos = self.client.open_algo_orders(self.symbol)
        except BinanceAPIError as e:
            print(f"[cancel] open_algo_orders failed: {e.payload}")
            return

        for o in open_algos:
            cid = str(o.get("clientAlgoId", ""))
            if cid.startswith("PDHL_E_"):
                try:
                    self.client.cancel_algo_order(client_algo_id=cid)
                    print(f"[cancel] canceled entry algo {cid}")
                except BinanceAPIError as e:
                    # -2011 Unknown order sent -> already canceled/triggered; ignore
                    if e.code == -2011:
                        continue
                    print(f"[cancel] failed to cancel {cid}: {e.payload}")

    def place_daily_entry_orders(self, prev_high: float, prev_low: float) -> None:
        self.cancel_all_entry_algo_orders()

        try:
            current_price = self.client.ticker_price(self.symbol)
        except BinanceAPIError as e:
            print(f"[entry] ticker_price failed: {e.payload}")
            return

        qty_str = format_to_step(self.qty, self.step_size, ROUND_DOWN)

        # ---- LONG entry ----
        if current_price >= prev_high:
            print(f"[entry] price {current_price} >= prev_high {prev_high} -> MARKET BUY LONG")
            try:
                self.client.new_market_order(self.symbol, "BUY", "LONG", qty_str)
            except BinanceAPIError as e:
                print(f"[entry] MARKET BUY failed: {e.payload}")
        else:
            trigger = format_to_step(prev_high, self.tick_size, ROUND_UP)
            cid = f"PDHL_E_L_{int(time.time() * 1000)}"
            params = {
                "algoType": "CONDITIONAL",
                "symbol": self.symbol,
                "side": "BUY",
                "positionSide": "LONG",
                "type": "STOP_MARKET",
                "quantity": qty_str,
                "triggerPrice": trigger,
                "workingType": "CONTRACT_PRICE",
                "clientAlgoId": cid,
                "newOrderRespType": "ACK",
            }
            try:
                self.client.new_algo_order(params)
                print(f"[entry] placed BUY STOP_MARKET LONG trigger={trigger} clientAlgoId={cid}")
            except BinanceAPIError as e:
                print(f"[entry] place BUY stop failed: {e.payload}")

        # ---- SHORT entry ----
        if current_price <= prev_low:
            print(f"[entry] price {current_price} <= prev_low {prev_low} -> MARKET SELL SHORT")
            try:
                self.client.new_market_order(self.symbol, "SELL", "SHORT", qty_str)
            except BinanceAPIError as e:
                print(f"[entry] MARKET SELL failed: {e.payload}")
        else:
            trigger = format_to_step(prev_low, self.tick_size, ROUND_DOWN)
            cid = f"PDHL_E_S_{int(time.time() * 1000)}"
            params = {
                "algoType": "CONDITIONAL",
                "symbol": self.symbol,
                "side": "SELL",
                "positionSide": "SHORT",
                "type": "STOP_MARKET",
                "quantity": qty_str,
                "triggerPrice": trigger,
                "workingType": "CONTRACT_PRICE",
                "clientAlgoId": cid,
                "newOrderRespType": "ACK",
            }
            try:
                self.client.new_algo_order(params)
                print(f"[entry] placed SELL STOP_MARKET SHORT trigger={trigger} clientAlgoId={cid}")
            except BinanceAPIError as e:
                print(f"[entry] place SELL stop failed: {e.payload}")

    # ---------------------------
    # Exit orders + trailing management
    # ---------------------------
    def _find_open_algo_by_prefix(self, open_algos: List[Dict[str, Any]], prefix: str) -> Optional[Dict[str, Any]]:
        matches = [o for o in open_algos if str(o.get("clientAlgoId", "")).startswith(prefix)]
        if not matches:
            return None
        matches.sort(key=lambda x: int(x.get("updateTime", x.get("createTime", 0))), reverse=True)
        return matches[0]

    def _position_open_now(self, pos_side: str) -> Tuple[bool, float]:
        """
        Returns (is_open, entry_price) for the given side using a fresh positionRisk call.
        """
        try:
            positions = self.client.position_risk(self.symbol)
        except BinanceAPIError:
            return False, 0.0

        p = next((x for x in positions if x.get("positionSide") == pos_side), None)
        if not p:
            return False, 0.0
        amt = float(p.get("positionAmt", "0") or "0")
        entry = float(p.get("entryPrice", "0") or "0")
        return abs(amt) > 0, entry

    def ensure_exits_for_side(self, side_state: SideState, entry_price: float, open_algos: List[Dict[str, Any]]) -> None:
        side_letter = "L" if side_state.pos_side == "LONG" else "S"
        tp_prefix = f"PDHL_TP_{side_letter}_"
        sl_prefix = f"PDHL_SL_{side_letter}_"

        tp_order = self._find_open_algo_by_prefix(open_algos, tp_prefix)
        sl_order = self._find_open_algo_by_prefix(open_algos, sl_prefix)

        # ---- Take Profit (48.0) ----
        if tp_order is None:
            if side_state.pos_side == "LONG":
                tp_price = entry_price + self.tp_offset
                trigger = format_to_step(tp_price, self.tick_size, ROUND_UP)
                side = "SELL"
                position_side = "LONG"
            else:
                tp_price = entry_price - self.tp_offset
                trigger = format_to_step(tp_price, self.tick_size, ROUND_DOWN)
                side = "BUY"
                position_side = "SHORT"

            cid = f"{tp_prefix}{int(time.time() * 1000)}"
            params = {
                "algoType": "CONDITIONAL",
                "symbol": self.symbol,
                "side": side,
                "positionSide": position_side,
                "type": "TAKE_PROFIT_MARKET",
                "closePosition": "true",
                "triggerPrice": trigger,
                "workingType": "CONTRACT_PRICE",
                "clientAlgoId": cid,
                "newOrderRespType": "ACK",
            }
            try:
                self.client.new_algo_order(params)
                side_state.tp_client_id = cid
                print(f"[exit] placed TP {side_state.pos_side} trigger={trigger} clientAlgoId={cid}")
            except BinanceAPIError as e:
                print(f"[exit] TP placement failed: {e.payload}")

        # ---- Stop Loss (5.2) ----
        if sl_order is None:
            if side_state.pos_side == "LONG":
                sl_price = entry_price - self.sl_offset
                trigger = format_to_step(sl_price, self.tick_size, ROUND_DOWN)
                side = "SELL"
                position_side = "LONG"
            else:
                sl_price = entry_price + self.sl_offset
                trigger = format_to_step(sl_price, self.tick_size, ROUND_UP)
                side = "BUY"
                position_side = "SHORT"

            cid = f"{sl_prefix}{int(time.time() * 1000)}"
            params = {
                "algoType": "CONDITIONAL",
                "symbol": self.symbol,
                "side": side,
                "positionSide": position_side,
                "type": "STOP_MARKET",
                "closePosition": "true",
                "triggerPrice": trigger,
                "workingType": "CONTRACT_PRICE",
                "clientAlgoId": cid,
                "newOrderRespType": "ACK",
            }
            try:
                self.client.new_algo_order(params)
                side_state.sl_client_id = cid
                side_state.sl_trigger = float(trigger)

                # Trailing only begins AFTER BE+0.16 move happens later
                side_state.trailing_active = False
                side_state.trail_best_price = entry_price
                side_state.trail_last_update_best = entry_price
                side_state.last_sl_replace_ms = 0

                print(f"[exit] placed SL {side_state.pos_side} trigger={trigger} clientAlgoId={cid}")
            except BinanceAPIError as e:
                print(f"[exit] SL placement failed: {e.payload}")

        # Refresh current SL trigger if it exists
        if sl_order is not None:
            try:
                trig = float(sl_order.get("triggerPrice", "0"))
                side_state.sl_trigger = trig
                side_state.sl_client_id = str(sl_order.get("clientAlgoId", ""))
            except Exception:
                pass

    def cancel_exit_orders_for_side(self, side_state: SideState) -> None:
        side_letter = "L" if side_state.pos_side == "LONG" else "S"
        prefixes = [f"PDHL_TP_{side_letter}_", f"PDHL_SL_{side_letter}_"]

        try:
            open_algos = self.client.open_algo_orders(self.symbol)
        except BinanceAPIError as e:
            print(f"[exit] open_algo_orders failed while canceling exits: {e.payload}")
            return

        for o in open_algos:
            cid = str(o.get("clientAlgoId", ""))
            if any(cid.startswith(p) for p in prefixes):
                try:
                    self.client.cancel_algo_order(client_algo_id=cid)
                    print(f"[exit] canceled leftover {cid}")
                except BinanceAPIError as e:
                    if e.code == -2011:
                        continue
                    print(f"[exit] failed to cancel {cid}: {e.payload}")

        side_state.tp_client_id = None
        side_state.sl_client_id = None
        side_state.sl_trigger = 0.0
        side_state.trailing_active = False
        side_state.trail_best_price = 0.0
        side_state.trail_last_update_best = 0.0
        side_state.last_sl_replace_ms = 0

    def _replace_stop(self, side_state: SideState, new_trigger_price: float) -> None:
        """
        Replace the STOP_MARKET closePosition SL with a new trigger price.
        Fixes your crash by:
          - ignoring -2011 on cancel (order already gone)
          - re-checking position before placing new closePosition order
          - ignoring -4509 (position already closed)
        """
        now_ms = int(time.time() * 1000)

        # throttle to avoid cancel/replace race spam
        if now_ms - side_state.last_sl_replace_ms < 800:
            return

        # Ensure the position still exists right now
        is_open, _ = self._position_open_now(side_state.pos_side)
        if not is_open:
            return

        # Find and cancel current open SL by prefix (don’t trust cached id)
        side_letter = "L" if side_state.pos_side == "LONG" else "S"
        sl_prefix = f"PDHL_SL_{side_letter}_"

        try:
            open_algos = self.client.open_algo_orders(self.symbol)
        except BinanceAPIError:
            open_algos = []

        current_sl = self._find_open_algo_by_prefix(open_algos, sl_prefix)
        if current_sl is not None:
            current_cid = str(current_sl.get("clientAlgoId", ""))
            try:
                self.client.cancel_algo_order(client_algo_id=current_cid)
            except BinanceAPIError as e:
                if e.code != -2011:
                    print(f"[trail] cancel SL failed ({current_cid}): {e.payload}")

        # Re-check position before placing new closePosition SL
        is_open2, _ = self._position_open_now(side_state.pos_side)
        if not is_open2:
            return

        if side_state.pos_side == "LONG":
            side = "SELL"
            position_side = "LONG"
            trigger_str = format_to_step(new_trigger_price, self.tick_size, ROUND_DOWN)
        else:
            side = "BUY"
            position_side = "SHORT"
            trigger_str = format_to_step(new_trigger_price, self.tick_size, ROUND_UP)

        cid = f"{sl_prefix}{int(time.time() * 1000)}"
        params = {
            "algoType": "CONDITIONAL",
            "symbol": self.symbol,
            "side": side,
            "positionSide": position_side,
            "type": "STOP_MARKET",
            "closePosition": "true",
            "triggerPrice": trigger_str,
            "workingType": "CONTRACT_PRICE",
            "clientAlgoId": cid,
            "newOrderRespType": "ACK",
        }

        try:
            self.client.new_algo_order(params)
            side_state.sl_client_id = cid
            side_state.sl_trigger = float(trigger_str)
            side_state.last_sl_replace_ms = now_ms
            print(f"[trail] moved SL {side_state.pos_side} -> trigger={trigger_str} clientAlgoId={cid}")
        except BinanceAPIError as e:
            if e.code == -4509:
                # position disappeared during the replace; do not crash
                return
            print(f"[trail] SL replace failed: {e.payload}")

    def update_trailing_stop(self, side_state: SideState, current_price: float, entry_price: float) -> None:
        """
        Exact rules:
        - No trailing at all until profit >= 0.52
        - then move SL to BE+0.16 (long) or BE-0.16 (short)
        - then step-trail: every +1.2 favorable move, SL stays 1.2 behind best price
        """
        if not side_state.sl_client_id:
            return

        if side_state.pos_side == "LONG":
            profit = current_price - entry_price

            # Activate BE move
            if not side_state.trailing_active:
                if profit >= self.trail_profit_trigger:
                    be_stop = entry_price + self.be_offset
                    if be_stop > side_state.sl_trigger:
                        self._replace_stop(side_state, be_stop)
                        # only mark active after we actually moved SL to (about) BE+0.16
                        if side_state.sl_trigger >= (be_stop - float(self.tick_size)):
                            side_state.trailing_active = True
                            side_state.trail_best_price = current_price
                            side_state.trail_last_update_best = current_price
                return

            # Trailing active
            side_state.trail_best_price = max(side_state.trail_best_price, current_price)

            if side_state.trail_best_price - side_state.trail_last_update_best >= self.trail_distance:
                desired_stop = side_state.trail_best_price - self.trail_distance
                if desired_stop > side_state.sl_trigger:
                    self._replace_stop(side_state, desired_stop)
                    side_state.trail_last_update_best = side_state.trail_best_price

        else:
            profit = entry_price - current_price

            if not side_state.trailing_active:
                if profit >= self.trail_profit_trigger:
                    be_stop = entry_price - self.be_offset
                    if be_stop < side_state.sl_trigger or side_state.sl_trigger == 0.0:
                        self._replace_stop(side_state, be_stop)
                        if side_state.sl_trigger <= (be_stop + float(self.tick_size)):
                            side_state.trailing_active = True
                            side_state.trail_best_price = current_price
                            side_state.trail_last_update_best = current_price
                return

            side_state.trail_best_price = min(side_state.trail_best_price, current_price)

            if side_state.trail_last_update_best - side_state.trail_best_price >= self.trail_distance:
                desired_stop = side_state.trail_best_price + self.trail_distance
                if desired_stop < side_state.sl_trigger:
                    self._replace_stop(side_state, desired_stop)
                    side_state.trail_last_update_best = side_state.trail_best_price

    # ---------------------------
    # Main poll loop
    # ---------------------------
    def poll_and_manage(self) -> None:
        try:
            current_price = self.client.ticker_price(self.symbol)
            positions = self.client.position_risk(self.symbol)
            open_algos = self.client.open_algo_orders(self.symbol)
        except BinanceAPIError as e:
            print(f"[loop] API error: {e.payload}")
            return

        long_pos = next((p for p in positions if p.get("positionSide") == "LONG"), None)
        short_pos = next((p for p in positions if p.get("positionSide") == "SHORT"), None)

        # LONG
        amt_l = float(long_pos.get("positionAmt", "0") or "0") if long_pos else 0.0
        entry_l = float(long_pos.get("entryPrice", "0") or "0") if long_pos else 0.0

        was_open_l = abs(self.long.position_amt) > 0
        is_open_l = abs(amt_l) > 0

        if is_open_l:
            if not was_open_l:
                self.long.trailing_active = False
                self.long.trail_best_price = entry_l
                self.long.trail_last_update_best = entry_l
                self.long.last_sl_replace_ms = 0

            self.long.position_amt = amt_l
            self.long.entry_price = entry_l

            self.ensure_exits_for_side(self.long, entry_l, open_algos)
            self.update_trailing_stop(self.long, current_price, entry_l)
        else:
            if was_open_l:
                self.cancel_exit_orders_for_side(self.long)
            self.long.position_amt = 0.0
            self.long.entry_price = 0.0

        # SHORT
        amt_s = float(short_pos.get("positionAmt", "0") or "0") if short_pos else 0.0
        entry_s = float(short_pos.get("entryPrice", "0") or "0") if short_pos else 0.0

        was_open_s = abs(self.short.position_amt) > 0
        is_open_s = abs(amt_s) > 0

        if is_open_s:
            if not was_open_s:
                self.short.trailing_active = False
                self.short.trail_best_price = entry_s
                self.short.trail_last_update_best = entry_s
                self.short.last_sl_replace_ms = 0

            self.short.position_amt = amt_s
            self.short.entry_price = entry_s

            self.ensure_exits_for_side(self.short, entry_s, open_algos)
            self.update_trailing_stop(self.short, current_price, entry_s)
        else:
            if was_open_s:
                self.cancel_exit_orders_for_side(self.short)
            self.short.position_amt = 0.0
            self.short.entry_price = 0.0

    def run(self) -> None:
        self.client.sync_time()
        self.load_symbol_filters()
        self.setup_account()

        prev_high, prev_low, prev_close_time = self.get_prev_day_levels()
        self.last_processed_prev_close_time = prev_close_time

        print(f"[daily] startup -> prev_high={prev_high} prev_low={prev_low} prev_close_time={prev_close_time}")
        self.place_daily_entry_orders(prev_high, prev_low)

        last_daily_check = 0.0
        while True:
            now = time.time()

            if now - last_daily_check >= 5.0:
                last_daily_check = now
                try:
                    prev_high, prev_low, prev_close_time = self.get_prev_day_levels()
                    if prev_close_time > self.last_processed_prev_close_time:
                        self.last_processed_prev_close_time = prev_close_time
                        print(f"[daily] new day detected -> prev_high={prev_high}, prev_low={prev_low}")
                        self.place_daily_entry_orders(prev_high, prev_low)
                except BinanceAPIError as e:
                    print(f"[daily] check failed: {e.payload}")

            self.poll_and_manage()
            time.sleep(1.0)


def main() -> None:
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    base_url = os.environ.get("BINANCE_BASE_URL", "https://demo-fapi.binance.com").strip()

    if not api_key or not api_secret:
        raise SystemExit("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

    symbol = "ETHUSDT"
    print(f"[config] base_url={base_url} symbol={symbol}")

    client = BinanceFuturesREST(api_key=api_key, api_secret=api_secret, base_url=base_url)
    bot = PriorDayBreakoutBot(client, symbol)
    bot.run()


if __name__ == "__main__":
    main()
