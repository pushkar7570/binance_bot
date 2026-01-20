from __future__ import annotations

import os
import json
import time
import hmac
import hashlib
import threading
from dataclasses import dataclass, asdict
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN
from urllib.parse import urlencode

import requests
import websocket
from dotenv import load_dotenv

# High precision for crypto decimals
getcontext().prec = 28

# --- Binance endpoints (USDⓈ-M Futures) ---
REST_MAINNET = "https://fapi.binance.com"
WS_MAINNET = "wss://fstream.binance.com"

# Per Binance USDⓈ-M Futures general info: testnet REST + WS bases
REST_TESTNET = "https://demo-fapi.binance.com"
WS_TESTNET_PRIMARY = "wss://fstream.binancefuture.com"  # official docs
WS_TESTNET_FALLBACK = "wss://stream.binancefuture.com"  # commonly used fallback


def now_ms() -> int:
    return int(time.time() * 1000)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


def fmt_dec(d: Decimal) -> str:
    # Binance accepts decimals as strings; strip trailing zeros safely
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


@dataclass
class StrategyParams:
    qty: Decimal
    entry_offset_ticks: int
    profit_target_ticks: int
    stop_loss_ticks: int
    be_trigger_ticks: int
    be_plus_ticks: int
    trail_dist_ticks: int
    trail_act_ticks_in: int
    use_market_on_gap: bool

    @property
    def trail_act_ticks(self) -> int:
        # Pine: trailActTicksIn == 0 ? (beTriggerTicks + trailDistTicks) : trailActTicksIn
        return self.trail_act_ticks_in or (self.be_trigger_ticks + self.trail_dist_ticks)


@dataclass
class BotState:
    # Daily
    day_open_time_ms: int | None = None
    day_open_price: str | None = None
    buy_stop: str | None = None
    sell_stop: str | None = None
    long_trigger_active: bool = False
    short_trigger_active: bool = False

    # Position management (mirrors Pine vars)
    position_amt: str = "0"
    entry_price: str | None = None
    high_water: str | None = None
    low_water: str | None = None
    be_done: bool = False
    stop_price: str | None = None

    # Exchange order IDs we manage
    sl_client_algo_id: str | None = None  # STOP_MARKET via /fapi/v1/algoOrder
    tp_client_order_id: str | None = None  # LIMIT reduceOnly via /fapi/v1/order

    # Price tracking
    last_price: str | None = None
    prev_price: str | None = None


class BinanceFuturesClient:
    """
    Minimal REST client for Binance USDⓈ-M Futures (testnet/mainnet).
    Uses signed requests for user endpoints.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        self.time_offset_ms = 0
        self.recv_window = 5000

        self.sync_time()

    def sync_time(self) -> None:
        # Server time endpoint is /fapi/v1/time (mentioned in exchangeInfo docs)
        r = self.session.get(f"{self.base_url}/fapi/v1/time", timeout=10)
        r.raise_for_status()
        server_time = int(r.json()["serverTime"])
        self.time_offset_ms = server_time - now_ms()
        log(f"Time sync offset={self.time_offset_ms}ms")

    def _signed_request(self, method: str, path: str, params: dict) -> dict:
        p = dict(params)
        p["timestamp"] = now_ms() + self.time_offset_ms
        p["recvWindow"] = self.recv_window

        qs = urlencode(p, doseq=True)
        sig = hmac.new(self.api_secret, qs.encode("utf-8"), hashlib.sha256).hexdigest()
        url = f"{self.base_url}{path}?{qs}&signature={sig}"

        for attempt in range(1, 4):
            try:
                if method == "GET":
                    r = self.session.get(url, timeout=10)
                elif method == "POST":
                    r = self.session.post(url, timeout=10)
                elif method == "DELETE":
                    r = self.session.delete(url, timeout=10)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                data = r.json()
                # Binance returns {"code":..., "msg":...} for many errors
                if isinstance(data, dict) and "code" in data and data.get("code") not in (0, 200):
                    raise RuntimeError(f"Binance error {data.get('code')}: {data.get('msg')}")
                return data
            except Exception as e:
                if attempt == 3:
                    raise
                log(f"REST retry {attempt}/3 after error: {e}")
                time.sleep(1.0)

        raise RuntimeError("Unreachable")

    def _public_get(self, path: str, params: dict | None = None) -> dict:
        params = params or {}
        url = f"{self.base_url}{path}"
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    # --- Market data ---
    def exchange_info(self) -> dict:
        return self._public_get("/fapi/v1/exchangeInfo")

    def klines(self, symbol: str, interval: str, limit: int = 2) -> list:
        return self._public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    # --- Account / trading ---
    def position_risk(self, symbol: str) -> dict:
        # Position Information V3: GET /fapi/v3/positionRisk
        data = self._signed_request("GET", "/fapi/v3/positionRisk", {"symbol": symbol})
        # Response can be list or dict depending on params; normalize:
        if isinstance(data, list):
            return data[0] if data else {}
        return data

    def place_order(self, params: dict) -> dict:
        # New Order endpoint: POST /fapi/v1/order
        return self._signed_request("POST", "/fapi/v1/order", params)

    def cancel_order(self, symbol: str, orig_client_order_id: str) -> dict:
        # Cancel Order endpoint: DELETE /fapi/v1/order (origClientOrderId supported)
        return self._signed_request("DELETE", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": orig_client_order_id})

    def open_orders(self, symbol: str) -> list:
        # GET /fapi/v1/openOrders
        data = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": symbol})
        return data if isinstance(data, list) else []

    # Conditional orders must use Algo endpoints (post-migration)
    def place_algo_order(self, params: dict) -> dict:
        # POST /fapi/v1/algoOrder
        return self._signed_request("POST", "/fapi/v1/algoOrder", params)

    def cancel_algo_order(self, client_algo_id: str) -> dict:
        # DELETE /fapi/v1/algoOrder (clientAlgoId supported)
        return self._signed_request("DELETE", "/fapi/v1/algoOrder", {"clientAlgoId": client_algo_id})

    def open_algo_orders(self, symbol: str) -> list:
        # GET /fapi/v1/openAlgoOrders
        data = self._signed_request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol})
        return data if isinstance(data, list) else []

    # Optional convenience settings
    def set_margin_type(self, symbol: str, margin_type: str) -> None:
        try:
            self._signed_request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})
            log(f"Margin type set to {margin_type}")
        except Exception as e:
            # Binance returns error if already set; safe to ignore
            log(f"Margin type not changed (ok): {e}")

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self._signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})
            log(f"Leverage set to {leverage}x")
        except Exception as e:
            log(f"Leverage not changed (ok): {e}")

    def set_one_way_mode(self) -> None:
        # POST /fapi/v1/positionSide/dual dualSidePosition="false" => one-way mode
        try:
            self._signed_request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"})
            log("Position mode set to One-way")
        except Exception as e:
            log(f"Position mode not changed (ok): {e}")


class PriorDayDualBreakoutBot:
    """
    Implements your PineScript logic exactly:
    - Prior day high/low from daily candles
    - Daily reset: 2 triggers (or only opposite if already in a position)
    - Entry: stop-style trigger (virtual) OR market-on-gap
    - Exits: TP limit + SL STOP_MARKET (Algo order)
    - BE + trail stop logic in ticks, using symbol tickSize as "tick"
    """

    STATE_FILE = "state.json"
    ID_PREFIX = "PDDB"  # keep short; Binance client IDs max length 36

    def __init__(self, client: BinanceFuturesClient, symbol: str, params: StrategyParams):
        self.client = client
        self.symbol = symbol.upper()
        self.params = params

        self.lock = threading.Lock()
        self.state = self._load_state()

        self.tick = Decimal("0")
        self.qty_step = Decimal("0")

        self.new_day_event = threading.Event()
        self.stop_event = threading.Event()

        self.ws_url = None

        # Track current day open from WS (for gap logic)
        self._ws_day_open_time_ms = None
        self._ws_day_open_price = None

    def _load_state(self) -> BotState:
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return BotState(**data)
            except Exception:
                pass
        return BotState()

    def _save_state(self) -> None:
        with open(self.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2)

    # --- Rounding (mirrors Pine roundToTick + tick-based distances) ---
    def round_to_tick(self, x: Decimal) -> Decimal:
        # Pine: math.round(x / tick) * tick
        if self.tick <= 0:
            return x
        n = (x / self.tick).to_integral_value(rounding=ROUND_HALF_UP)
        return (n * self.tick).quantize(self.tick)

    def floor_qty(self, q: Decimal) -> Decimal:
        # Use floor to step to avoid invalid lot sizes
        if self.qty_step <= 0:
            return q
        n = (q / self.qty_step).to_integral_value(rounding=ROUND_DOWN)
        return (n * self.qty_step).quantize(self.qty_step)

    def _load_symbol_filters(self) -> None:
        info = self.client.exchange_info()
        sym = None
        for s in info.get("symbols", []):
            if s.get("symbol") == self.symbol:
                sym = s
                break
        if not sym:
            raise RuntimeError(f"Symbol not found in exchangeInfo: {self.symbol}")

        tick = None
        step = None
        for f in sym.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick = Decimal(f["tickSize"])
            if f.get("filterType") == "LOT_SIZE":
                step = Decimal(f["stepSize"])

        if tick is None or step is None:
            raise RuntimeError("Could not parse tickSize/stepSize from exchangeInfo.filters")

        self.tick = tick
        self.qty_step = step
        log(f"{self.symbol} tickSize={self.tick} stepSize={self.qty_step}")

    # --- Prior day levels from daily candles ---
    def _get_prior_day_levels(self) -> tuple[Decimal, Decimal, Decimal]:
        k = self.client.klines(self.symbol, "1d", limit=2)
        if len(k) < 2:
            raise RuntimeError("Not enough daily candles returned")

        prior = k[-2]
        curr = k[-1]

        prior_high = Decimal(prior[2])
        prior_low = Decimal(prior[3])
        day_open = Decimal(curr[1])
        return prior_high, prior_low, day_open

    def _compute_stops(self, prior_high: Decimal, prior_low: Decimal) -> tuple[Decimal, Decimal]:
        buy_stop = self.round_to_tick(prior_high + Decimal(self.params.entry_offset_ticks) * self.tick)
        sell_stop = self.round_to_tick(prior_low - Decimal(self.params.entry_offset_ticks) * self.tick)
        return buy_stop, sell_stop

    # --- Position polling (pos + entry) ---
    def _poll_position(self) -> tuple[Decimal, Decimal]:
        pr = self.client.position_risk(self.symbol)
        pos_amt = Decimal(pr.get("positionAmt", "0"))
        entry_price = Decimal(pr.get("entryPrice", "0"))
        return pos_amt, entry_price

    # --- Trading actions ---
    def _place_market_to_target(self, target_pos: Decimal) -> None:
        """
        Target position semantics (exact TradingView behavior):
        - If currently -1 and target +1 => buy 2
        - If currently +1 and target -1 => sell 2
        """
        pos_amt, _ = self._poll_position()

        delta = target_pos - pos_amt
        if delta == 0:
            return

        if delta > 0:
            side = "BUY"
            qty = delta
        else:
            side = "SELL"
            qty = -delta

        qty = self.floor_qty(qty)
        if qty <= 0:
            log(f"Computed qty <= 0, skipping. pos={pos_amt} target={target_pos}")
            return

        log(f"MARKET {side} qty={qty} to target_pos={target_pos} (from pos={pos_amt})")
        self.client.place_order({
            "symbol": self.symbol,
            "side": side,
            "type": "MARKET",
            "quantity": fmt_dec(qty),
            "newOrderRespType": "RESULT",
        })

    def _cancel_exit_orders(self) -> None:
        # Cancel SL algo order
        if self.state.sl_client_algo_id:
            try:
                self.client.cancel_algo_order(self.state.sl_client_algo_id)
                log(f"Canceled SL algo {self.state.sl_client_algo_id}")
            except Exception as e:
                log(f"SL cancel failed (ok if already filled/closed): {e}")
            self.state.sl_client_algo_id = None

        # Cancel TP limit order
        if self.state.tp_client_order_id:
            try:
                self.client.cancel_order(self.symbol, self.state.tp_client_order_id)
                log(f"Canceled TP order {self.state.tp_client_order_id}")
            except Exception as e:
                log(f"TP cancel failed (ok if already filled/closed): {e}")
            self.state.tp_client_order_id = None

    def _place_brackets_for_position(self, pos_amt: Decimal, entry_price: Decimal) -> None:
        """
        Places:
        - SL: STOP_MARKET via /fapi/v1/algoOrder (reduceOnly=true)
        - TP: LIMIT reduceOnly via /fapi/v1/order
        """
        if pos_amt == 0:
            return

        qty = self.floor_qty(abs(pos_amt))
        if qty <= 0:
            raise RuntimeError("Position qty invalid after rounding")

        # Initial stop exactly like Pine reset:
        if pos_amt > 0:
            stop_price = self.round_to_tick(entry_price - Decimal(self.params.stop_loss_ticks) * self.tick)
            tp_price = self.round_to_tick(entry_price + Decimal(self.params.profit_target_ticks) * self.tick)
            sl_side = "SELL"
            tp_side = "SELL"
        else:
            stop_price = self.round_to_tick(entry_price + Decimal(self.params.stop_loss_ticks) * self.tick)
            tp_price = self.round_to_tick(entry_price - Decimal(self.params.profit_target_ticks) * self.tick)
            sl_side = "BUY"
            tp_side = "BUY"

        # Save management state (Pine vars)
        self.state.entry_price = fmt_dec(entry_price)
        self.state.be_done = False
        self.state.stop_price = fmt_dec(stop_price)

        last_price = Decimal(self.state.last_price) if self.state.last_price else entry_price
        if pos_amt > 0:
            self.state.high_water = fmt_dec(max(last_price, entry_price))
            self.state.low_water = None
        else:
            self.state.low_water = fmt_dec(min(last_price, entry_price))
            self.state.high_water = None

        # Place SL as Algo conditional order (STOP_MARKET)
        sl_id = f"{self.ID_PREFIX}_SL_{int(time.time())}"
        self.client.place_algo_order({
            "algoType": "CONDITIONAL",
            "symbol": self.symbol,
            "side": sl_side,
            "type": "STOP_MARKET",
            "quantity": fmt_dec(qty),
            "triggerPrice": fmt_dec(stop_price),
            "workingType": "CONTRACT_PRICE",
            "reduceOnly": "true",
            "clientAlgoId": sl_id,
            "newOrderRespType": "ACK",
        })
        self.state.sl_client_algo_id = sl_id
        log(f"Placed SL STOP_MARKET {sl_side} qty={qty} trigger={stop_price} id={sl_id}")

        # Place TP as reduceOnly LIMIT order
        tp_id = f"{self.ID_PREFIX}_TP_{int(time.time())}"
        self.client.place_order({
            "symbol": self.symbol,
            "side": tp_side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": fmt_dec(qty),
            "price": fmt_dec(tp_price),
            "reduceOnly": "true",
            "newClientOrderId": tp_id,
            "newOrderRespType": "ACK",
        })
        self.state.tp_client_order_id = tp_id
        log(f"Placed TP LIMIT {tp_side} qty={qty} price={tp_price} id={tp_id}")

    # --- Core logic handlers ---
    def _on_new_day(self, day_open_time_ms: int, day_open_price: Decimal | None) -> None:
        # Refresh prior day levels from REST
        prior_high, prior_low, rest_day_open = self._get_prior_day_levels()
        buy_stop, sell_stop = self._compute_stops(prior_high, prior_low)

        # Use WS open price if available; else REST open
        day_open = day_open_price if day_open_price is not None else rest_day_open

        pos_amt, _ = self._poll_position()

        self.state.day_open_time_ms = day_open_time_ms
        self.state.day_open_price = fmt_dec(day_open)
        self.state.buy_stop = fmt_dec(buy_stop)
        self.state.sell_stop = fmt_dec(sell_stop)

        # Activate triggers exactly like Pine's "canPlaceLong/canPlaceShort" logic
        self.state.long_trigger_active = (pos_amt <= 0)
        self.state.short_trigger_active = (pos_amt >= 0)

        log(f"NEW DAY -> priorHigh={prior_high} priorLow={prior_low} buyStop={buy_stop} sellStop={sell_stop} open={day_open} pos={pos_amt}")

        # Market-on-gap rule (exact)
        if self.params.use_market_on_gap:
            if self.state.long_trigger_active and day_open >= buy_stop:
                log("Gap above buyStop -> MARKET entry long")
                self._place_market_to_target(self.params.qty)
                self.state.long_trigger_active = False

            if self.state.short_trigger_active and day_open <= sell_stop:
                log("Gap below sellStop -> MARKET entry short")
                self._place_market_to_target(-self.params.qty)
                self.state.short_trigger_active = False

        # Reset prev_price so crossing logic behaves correctly on new day
        self.state.prev_price = None

        self._save_state()

    def _update_entry_triggers_on_price(self, price: Decimal) -> None:
        if not self.state.buy_stop or not self.state.sell_stop:
            return

        buy_stop = Decimal(self.state.buy_stop)
        sell_stop = Decimal(self.state.sell_stop)

        prev = Decimal(self.state.prev_price) if self.state.prev_price else None

        # LONG trigger: crossing up through buy_stop
        if self.state.long_trigger_active:
            crossed = (prev is None and price >= buy_stop) or (prev is not None and prev < buy_stop and price >= buy_stop)
            if crossed:
                log(f"Triggered LONG breakout at price={price} >= buyStop={buy_stop}")
                self._place_market_to_target(self.params.qty)
                self.state.long_trigger_active = False

        # SHORT trigger: crossing down through sell_stop
        if self.state.short_trigger_active:
            crossed = (prev is None and price <= sell_stop) or (prev is not None and prev > sell_stop and price <= sell_stop)
            if crossed:
                log(f"Triggered SHORT breakdown at price={price} <= sellStop={sell_stop}")
                self._place_market_to_target(-self.params.qty)
                self.state.short_trigger_active = False

    def _update_be_and_trail(self, price: Decimal, pos_amt: Decimal) -> None:
        if pos_amt == 0 or not self.state.entry_price or not self.state.stop_price:
            return

        entry_price = Decimal(self.state.entry_price)
        stop_price = Decimal(self.state.stop_price)

        be_done = self.state.be_done
        tick = self.tick

        # LONG
        if pos_amt > 0:
            high_water = Decimal(self.state.high_water) if self.state.high_water else entry_price
            high_water = max(high_water, price)
            self.state.high_water = fmt_dec(high_water)

            mfe_ticks = (high_water - entry_price) / tick

            # Stage 1: break-even(+)
            if (not be_done) and mfe_ticks >= Decimal(self.params.be_trigger_ticks):
                be_stop = self.round_to_tick(entry_price + Decimal(self.params.be_plus_ticks) * tick)
                if be_stop > stop_price:
                    stop_price = be_stop
                be_done = True

            # Stage 2: trail after activation threshold
            if be_done and mfe_ticks >= Decimal(self.params.trail_act_ticks):
                trail_stop = self.round_to_tick(high_water - Decimal(self.params.trail_dist_ticks) * tick)
                if trail_stop > stop_price:
                    stop_price = trail_stop

        # SHORT
        else:
            low_water = Decimal(self.state.low_water) if self.state.low_water else entry_price
            low_water = min(low_water, price)
            self.state.low_water = fmt_dec(low_water)

            mfe_ticks = (entry_price - low_water) / tick

            # Stage 1: break-even(+)
            if (not be_done) and mfe_ticks >= Decimal(self.params.be_trigger_ticks):
                be_stop = self.round_to_tick(entry_price - Decimal(self.params.be_plus_ticks) * tick)
                if be_stop < stop_price:
                    stop_price = be_stop
                be_done = True

            # Stage 2: trail after activation threshold
            if be_done and mfe_ticks >= Decimal(self.params.trail_act_ticks):
                trail_stop = self.round_to_tick(low_water + Decimal(self.params.trail_dist_ticks) * tick)
                if trail_stop < stop_price:
                    stop_price = trail_stop

        # If stop updated, replace SL algo order
        new_stop = self.round_to_tick(stop_price)
        if fmt_dec(new_stop) != self.state.stop_price:
            log(f"Stop update: {self.state.stop_price} -> {fmt_dec(new_stop)} (beDone={be_done})")
            self.state.stop_price = fmt_dec(new_stop)
            self.state.be_done = be_done

            # Replace SL order
            if self.state.sl_client_algo_id:
                try:
                    self.client.cancel_algo_order(self.state.sl_client_algo_id)
                except Exception as e:
                    log(f"Cancel SL before replace failed (ok): {e}")

            qty = self.floor_qty(abs(pos_amt))
            if qty <= 0:
                return

            sl_side = "SELL" if pos_amt > 0 else "BUY"
            sl_id = f"{self.ID_PREFIX}_SL_{int(time.time())}"
            self.client.place_algo_order({
                "algoType": "CONDITIONAL",
                "symbol": self.symbol,
                "side": sl_side,
                "type": "STOP_MARKET",
                "quantity": fmt_dec(qty),
                "triggerPrice": fmt_dec(new_stop),
                "workingType": "CONTRACT_PRICE",
                "reduceOnly": "true",
                "clientAlgoId": sl_id,
                "newOrderRespType": "ACK",
            })
            self.state.sl_client_algo_id = sl_id
            log(f"Replaced SL id={sl_id} trigger={new_stop}")

    # --- WebSocket handling ---
    def _ws_on_message(self, ws, message: str) -> None:
        try:
            msg = json.loads(message)
            data = msg.get("data", msg)

            # Trade stream
            if data.get("e") == "trade":
                price = Decimal(data["p"])
                with self.lock:
                    self.state.last_price = fmt_dec(price)

            # Kline stream (1d)
            if data.get("e") == "kline" and "k" in data:
                k = data["k"]
                if k.get("i") == "1d":
                    open_time = int(k["t"])
                    open_price = Decimal(k["o"])

                    # Detect new day by open_time change
                    if self._ws_day_open_time_ms != open_time:
                        self._ws_day_open_time_ms = open_time
                        self._ws_day_open_price = open_price
                        self.new_day_event.set()

        except Exception:
            # Ignore parse errors
            return

    def _ws_loop(self) -> None:
        streams = f"{self.symbol.lower()}@trade/{self.symbol.lower()}@kline_1d"

        # Try official testnet WS base first, then fallback
        for base in (WS_TESTNET_PRIMARY, WS_TESTNET_FALLBACK) if "demo-fapi" in self.client.base_url else (WS_MAINNET,):
            url = f"{base}/stream?streams={streams}"
            log(f"WS trying: {url}")
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=self._ws_on_message,
                    on_error=lambda _ws, err: log(f"WS error: {err}"),
                    on_close=lambda _ws, code, msg: log(f"WS closed: code={code} msg={msg}"),
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                log(f"WS connect failed: {e}")
                time.sleep(2.0)

            if self.stop_event.is_set():
                return

            # reconnect
            time.sleep(2.0)

    def run(self) -> None:
        self._load_symbol_filters()

        # Optional account setup
        leverage = int(os.getenv("LEVERAGE", "0") or "0")
        margin_type = os.getenv("MARGIN_TYPE", "").strip().upper()
        one_way = os.getenv("ONE_WAY_MODE", "true").lower() == "true"

        if one_way:
            self.client.set_one_way_mode()
        if margin_type in ("ISOLATED", "CROSSED"):
            self.client.set_margin_type(self.symbol, margin_type)
        if leverage > 0:
            self.client.set_leverage(self.symbol, leverage)

        # Start WS thread
        t = threading.Thread(target=self._ws_loop, daemon=True)
        t.start()

        # Prime with current position
        pos_amt, entry_price = self._poll_position()
        with self.lock:
            self.state.position_amt = fmt_dec(pos_amt)

        log(f"Bot started. Initial pos={pos_amt} entry={entry_price}")

        # Main loop
        last_pos_poll = 0.0

        while True:
            if self.stop_event.is_set():
                return

            # Handle new day event
            if self.new_day_event.is_set():
                self.new_day_event.clear()
                with self.lock:
                    open_time = self._ws_day_open_time_ms
                    open_price = self._ws_day_open_price
                if open_time is not None:
                    with self.lock:
                        self._on_new_day(open_time, open_price)

            # Pull last price
            with self.lock:
                lp = Decimal(self.state.last_price) if self.state.last_price else None

            # Update entry triggers on price
            if lp is not None:
                with self.lock:
                    self._update_entry_triggers_on_price(lp)
                    self.state.prev_price = self.state.last_price

            # Poll position at 1s cadence
            if time.time() - last_pos_poll >= 1.0:
                last_pos_poll = time.time()
                new_pos, new_entry = self._poll_position()

                with self.lock:
                    old_pos = Decimal(self.state.position_amt)
                    if new_pos != old_pos:
                        log(f"POS CHANGED: {old_pos} -> {new_pos} entry={new_entry}")

                        # Pine posChanged resets everything
                        self._cancel_exit_orders()

                        # Reset/initialize management variables
                        self.state.position_amt = fmt_dec(new_pos)
                        if new_pos == 0:
                            self.state.entry_price = None
                            self.state.high_water = None
                            self.state.low_water = None
                            self.state.be_done = False
                            self.state.stop_price = None
                        else:
                            # Place fresh SL+TP for the new position (exact)
                            self._place_brackets_for_position(new_pos, new_entry)

                        self._save_state()

            # Update BE + trailing if in position
            with self.lock:
                pos_amt = Decimal(self.state.position_amt)
                if lp is not None and pos_amt != 0:
                    self._update_be_and_trail(lp, pos_amt)
                    self._save_state()

            time.sleep(0.2)


def read_bool(env_val: str, default: bool) -> bool:
    if env_val is None:
        return default
    v = env_val.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def main() -> None:
    load_dotenv()

    env = os.getenv("BINANCE_ENV", "TESTNET").strip().upper()
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    symbol = os.getenv("SYMBOL", "ETHUSDT").strip().upper()

    if not api_key or not api_secret:
        raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")

    base_url = REST_TESTNET if env == "TESTNET" else REST_MAINNET

    # Strategy inputs (ticks) — exact defaults from your Pine
    params = StrategyParams(
        qty=Decimal(os.getenv("QTY", "1")),
        entry_offset_ticks=int(os.getenv("ENTRY_OFFSET_TICKS", "0")),
        profit_target_ticks=int(os.getenv("PROFIT_TARGET_TICKS", "4800")),
        stop_loss_ticks=int(os.getenv("STOP_LOSS_TICKS", "520")),
        be_trigger_ticks=int(os.getenv("BE_TRIGGER_TICKS", "52")),
        be_plus_ticks=int(os.getenv("BE_PLUS_TICKS", "16")),
        trail_dist_ticks=int(os.getenv("TRAIL_DIST_TICKS", "120")),
        trail_act_ticks_in=int(os.getenv("TRAIL_ACT_TICKS_IN", "0")),
        use_market_on_gap=read_bool(os.getenv("USE_MARKET_ON_GAP", "true"), True),
    )

    client = BinanceFuturesClient(api_key, api_secret, base_url)
    bot = PriorDayDualBreakoutBot(client, symbol, params)
    bot.run()


if __name__ == "__main__":
    main()
