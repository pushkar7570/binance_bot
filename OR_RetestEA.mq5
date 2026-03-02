//+------------------------------------------------------------------+
//|                                                   OR_RetestEA.mq5 |
//|   Opening Range (M1) breakout -> retest -> confirm (Confirm_TF)   |
//|   Hybrid SL: Structure swing vs ATR (more conservative chosen)    |
//|   Fixed $ risk sizing + optional trailing (toggle button)         |
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "1.00"
#property description "OR breakout with retest+confirm, hybrid SL (structure vs ATR), fixed risk sizing, trailing toggle."
#property strict

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "Sessions"
input bool Trade_Midnight = true; // Trade Midnight session (UTC 00:00)
input bool Trade_London   = true; // Trade London session   (UTC 07:00)
input bool Trade_NY       = true; // Trade NY session       (UTC 12:00)

input group "Opening Range / Signals"
input int            OR_Minutes             = 5;          // OR minutes (5, 15, 30) computed from M1 candles
input ENUM_TIMEFRAMES Confirm_TF            = PERIOD_M5;  // Confirm timeframe (M1, M3, M5)
input int            MaxMinutesAfterORClose = 105;        // Deadline minutes after OR close (no trade after)

input group "Risk & Reward"
input double RiskUSD = 100.0; // Risk per trade in account currency (USD assumed)
input double RR      = 2.0;   // Risk:Reward multiplier

input group "Stop Loss - Hybrid (Structure vs ATR)"
input double         SL_Buffer_Pips = 1.0;        // Buffer in pips added beyond structure swing
input int            ATR_Period     = 14;         // ATR period
input ENUM_TIMEFRAMES ATR_TF        = PERIOD_M5;  // ATR timeframe
input double         ATR_Mult       = 1.5;        // ATR multiplier for ATR-based SL

input group "Trailing Stop (runtime toggle via chart button)"
input double Trail_TriggerProfitPips = 10.0; // Start trailing after profit reaches this many pips
input double Trail_LockInPips        = 4.0;  // Lock-in profit in pips (from entry)
input double Trail_DistancePips      = 10.0; // Trail distance from Bid/Ask in pips

input group "Trade Execution"
input int MagicNumber     = 20260217; // Magic number for EA positions
input int SlippagePoints  = 20;       // Max slippage in points

//+------------------------------------------------------------------+
//| Enums / Structs                                                  |
//+------------------------------------------------------------------+
enum OR_State
{
   BUILD_OR = 0,
   WAIT_BREAKOUT,
   WAIT_RETEST,
   WAIT_CONFIRM,
   TRADED,
   RESET
};

enum BreakoutDir
{
   DIR_NONE = 0,
   DIR_LONG,
   DIR_SHORT
};

struct SessionState
{
   string     name;
   int        start_hour_utc;
   int        start_min_utc;
   bool       enabled;

   OR_State   state;

   datetime   session_start_server;
   double     ORH;
   double     ORL;
   datetime   OR_close_time;
   datetime   window_end_time;

   BreakoutDir breakout_dir;
   datetime   breakout_bar_time;

   bool       retest_seen;
   datetime   retest_start_time;

   bool       trade_taken;
};

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade      g_trade;

int         g_atr_handle = INVALID_HANDLE;

datetime    g_last_confirm_closed_bar_time = 0;
int         g_last_utc_day_key = 0;

bool        g_trailing_enabled = true;

SessionState g_sessions[3];

const string EA_PREFIX   = "ORRetestEA-";
const string BTN_TRAIL   = EA_PREFIX + "TrailingBtn";

//+------------------------------------------------------------------+
//| Utility                                                          |
//+------------------------------------------------------------------+
int UtcDayKey(const datetime t_utc)
{
   MqlDateTime dt;
   TimeToStruct(t_utc, dt);
   return (dt.year * 10000 + dt.mon * 100 + dt.day);
}

long ServerUtcOffsetSeconds()
{
   datetime server = TimeTradeServer();
   datetime gmt    = TimeGMT();

   if(server == 0 || gmt == 0)
   {
      server = TimeCurrent();
      gmt    = TimeGMT();
   }
   return (long)(server - gmt);
}

double PipSize(const string symbol)
{
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double pt  = SymbolInfoDouble(symbol, SYMBOL_POINT);

   // Common FX convention: 5-digit/3-digit => 1 pip = 10 points
   if(digits == 3 || digits == 5)
      return (10.0 * pt);

   return pt;
}

double TickSize(const string symbol)
{
   double ts = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   if(ts <= 0.0)
      ts = SymbolInfoDouble(symbol, SYMBOL_POINT);
   return ts;
}

double RoundToTick(const string symbol, const double price, const bool round_up)
{
   if(price <= 0.0)
      return 0.0;

   const double ts = TickSize(symbol);
   if(ts <= 0.0)
      return NormalizeDouble(price, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));

   const double ticks = price / ts;
   double rounded = 0.0;

   if(round_up)
      rounded = MathCeil(ticks) * ts;
   else
      rounded = MathFloor(ticks) * ts;

   return NormalizeDouble(rounded, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));
}

double NormalizeSLForOrder(const ENUM_ORDER_TYPE order_type, const double sl)
{
   if(sl <= 0.0)
      return 0.0;

   // BUY SL is below price => round down. SELL SL is above price => round up.
   return (order_type == ORDER_TYPE_BUY) ? RoundToTick(_Symbol, sl, false)
                                         : RoundToTick(_Symbol, sl, true);
}

double NormalizeTPForOrder(const ENUM_ORDER_TYPE order_type, const double tp)
{
   if(tp <= 0.0)
      return 0.0;

   // BUY TP is above => round up. SELL TP is below => round down.
   return (order_type == ORDER_TYPE_BUY) ? RoundToTick(_Symbol, tp, true)
                                         : RoundToTick(_Symbol, tp, false);
}

int VolumeDigitsFromStep(const double step)
{
   if(step <= 0.0)
      return 2;

   for(int d = 0; d <= 8; d++)
   {
      if(MathAbs(NormalizeDouble(step, d) - step) < 1e-12)
         return d;
   }
   return 2;
}

double NormalizeLotsDown(const double lots_raw)
{
   const double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   const double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   const double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(lots_raw <= 0.0 || vmin <= 0.0 || vmax <= 0.0 || vstep <= 0.0)
      return 0.0;

   double lots = MathFloor(lots_raw / vstep) * vstep;

   if(lots > vmax)
      lots = vmax;

   if(lots < vmin)
      return 0.0;

   return NormalizeDouble(lots, VolumeDigitsFromStep(vstep));
}

bool GetAtrValue(double &atr_value)
{
   atr_value = 0.0;

   if(g_atr_handle == INVALID_HANDLE)
      return false;

   double buf[];
   ArraySetAsSeries(buf, true);

   // Use last closed ATR bar (shift 1)
   const int copied = CopyBuffer(g_atr_handle, 0, 1, 1, buf);
   if(copied != 1)
      return false;

   atr_value = buf[0];
   return (atr_value > 0.0);
}

bool ComputeOpeningRange(const datetime start_server, const int minutes, double &orh, double &orl)
{
   orh = 0.0;
   orl = 0.0;

   if(minutes <= 0)
      return false;

   const datetime end_server = start_server + (datetime)(minutes * 60);

   MqlRates rates[];
   const int copied = CopyRates(_Symbol, PERIOD_M1, start_server, end_server, rates);
   if(copied <= 0)
      return false;

   double hi = -DBL_MAX;
   double lo =  DBL_MAX;
   int used = 0;

   for(int i = 0; i < copied; i++)
   {
      const datetime bt = rates[i].time;
      if(bt < start_server || bt >= end_server)
         continue;

      if(rates[i].high > hi) hi = rates[i].high;
      if(rates[i].low  < lo) lo = rates[i].low;
      used++;
   }

   if(used <= 0 || hi <= -DBL_MAX/2.0 || lo >= DBL_MAX/2.0)
      return false;

   orh = hi;
   orl = lo;
   return true;
}

bool GetSwingExtreme(const datetime start_bar_time, const datetime end_bar_time,
                     const BreakoutDir dir, double &extreme)
{
   extreme = 0.0;

   if(start_bar_time <= 0 || end_bar_time <= 0 || end_bar_time < start_bar_time)
      return false;

   const int tf_sec = PeriodSeconds(Confirm_TF);
   if(tf_sec <= 0)
      return false;

   const datetime stop_time = end_bar_time + (datetime)tf_sec;

   MqlRates rates[];
   const int copied = CopyRates(_Symbol, Confirm_TF, start_bar_time, stop_time, rates);
   if(copied <= 0)
      return false;

   int used = 0;

   if(dir == DIR_LONG)
   {
      double minLow = DBL_MAX;

      for(int i = 0; i < copied; i++)
      {
         if(rates[i].time < start_bar_time || rates[i].time > end_bar_time)
            continue;

         if(rates[i].low < minLow)
            minLow = rates[i].low;

         used++;
      }

      if(used <= 0 || minLow >= DBL_MAX/2.0)
         return false;

      extreme = minLow;
      return true;
   }
   else if(dir == DIR_SHORT)
   {
      double maxHigh = -DBL_MAX;

      for(int i = 0; i < copied; i++)
      {
         if(rates[i].time < start_bar_time || rates[i].time > end_bar_time)
            continue;

         if(rates[i].high > maxHigh)
            maxHigh = rates[i].high;

         used++;
      }

      if(used <= 0 || maxHigh <= -DBL_MAX/2.0)
         return false;

      extreme = maxHigh;
      return true;
   }

   return false;
}

bool ValidateOrderStops(const ENUM_ORDER_TYPE order_type, const double entry, const double sl, const double tp)
{
   const double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   const double freeze_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * _Point;
   const double min_dist = MathMax(stops_level, freeze_level);

   if(entry <= 0.0 || sl <= 0.0 || tp <= 0.0)
      return false;

   if(order_type == ORDER_TYPE_BUY)
   {
      if(sl >= entry) return false;
      if(tp <= entry) return false;

      if(min_dist > 0.0)
      {
         if((entry - sl) < min_dist) return false;
         if((tp - entry) < min_dist) return false;
      }
      return true;
   }

   if(order_type == ORDER_TYPE_SELL)
   {
      if(sl <= entry) return false;
      if(tp >= entry) return false;

      if(min_dist > 0.0)
      {
         if((sl - entry) < min_dist) return false;
         if((entry - tp) < min_dist) return false;
      }
      return true;
   }

   return false;
}

double LotsForRisk(const double stop_distance_price)
{
   if(stop_distance_price <= 0.0)
      return 0.0;

   const double tick_size = TickSize(_Symbol);
   if(tick_size <= 0.0)
      return 0.0;

   double tick_value_loss = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_LOSS);
   if(tick_value_loss <= 0.0)
      tick_value_loss = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);

   if(tick_value_loss <= 0.0)
      return 0.0;

   const double risk_per_1_lot = (stop_distance_price / tick_size) * tick_value_loss;
   if(risk_per_1_lot <= 0.0)
      return 0.0;

   const double raw_lots = RiskUSD / risk_per_1_lot;
   return NormalizeLotsDown(raw_lots);
}

void ResetSession(SessionState &s, const datetime session_start_server)
{
   s.session_start_server = session_start_server;

   s.state            = BUILD_OR;
   s.ORH              = 0.0;
   s.ORL              = 0.0;
   s.OR_close_time    = 0;
   s.window_end_time  = 0;

   s.breakout_dir     = DIR_NONE;
   s.breakout_bar_time= 0;

   s.retest_seen      = false;
   s.retest_start_time= 0;

   s.trade_taken      = false;
}

void SetupNewUtcDay()
{
   const datetime now_utc = TimeGMT();
   MqlDateTime dt;
   TimeToStruct(now_utc, dt);
   dt.hour = 0;
   dt.min  = 0;
   dt.sec  = 0;

   const datetime utc_day_start = StructToTime(dt);
   const long offset = ServerUtcOffsetSeconds();

   // Update enabled flags from inputs
   g_sessions[0].enabled = Trade_Midnight;
   g_sessions[1].enabled = Trade_London;
   g_sessions[2].enabled = Trade_NY;

   for(int i = 0; i < 3; i++)
   {
      const datetime start_utc    = utc_day_start + (datetime)(g_sessions[i].start_hour_utc * 3600 + g_sessions[i].start_min_utc * 60);
      const datetime start_server = start_utc + (datetime)offset;

      if(g_sessions[i].enabled)
      {
         ResetSession(g_sessions[i], start_server);
      }
      else
      {
         g_sessions[i].session_start_server = start_server;
         g_sessions[i].state = RESET;
         g_sessions[i].trade_taken = false;
      }
   }
}

void UpdateTrailingButton()
{
   if(ObjectFind(0, BTN_TRAIL) < 0)
      return;

   ObjectSetString(0, BTN_TRAIL, OBJPROP_TEXT, g_trailing_enabled ? "Trailing: ON" : "Trailing: OFF");
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_STATE, g_trailing_enabled);
   ChartRedraw();
}

void CreateTrailingButton()
{
   if(ObjectFind(0, BTN_TRAIL) >= 0)
      ObjectDelete(0, BTN_TRAIL);

   ObjectCreate(0, BTN_TRAIL, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_YDISTANCE, 20);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_XSIZE, 140);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_YSIZE, 20);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, BTN_TRAIL, OBJPROP_HIDDEN, true);

   UpdateTrailingButton();
}

void SetBreakout(SessionState &s, const BreakoutDir dir, const datetime bar_time)
{
   s.breakout_dir      = dir;
   s.breakout_bar_time = bar_time;
   s.retest_seen       = false;
   s.retest_start_time = 0;
}

bool ModifyPositionSLTP(const ulong ticket, const double new_sl, const double new_tp)
{
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);

   req.action   = TRADE_ACTION_SLTP;
   req.symbol   = _Symbol;
   req.position = ticket;
   req.sl       = new_sl;
   req.tp       = new_tp;
   req.magic    = MagicNumber;
   req.comment  = EA_PREFIX + "trail";

   ResetLastError();
   const bool ok = OrderSend(req, res);
   if(!ok)
   {
      PrintFormat("Modify SLTP OrderSend failed. err=%d", _LastError);
      return false;
   }

   if(res.retcode != TRADE_RETCODE_DONE && res.retcode != TRADE_RETCODE_DONE_PARTIAL)
   {
      PrintFormat("Modify SLTP rejected. retcode=%d (%s)", res.retcode, res.comment);
      return false;
   }

   return true;
}

bool TrailingAllowsSL(const ENUM_POSITION_TYPE pos_type, const double new_sl, const double bid, const double ask)
{
   const double stops_level  = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   const double freeze_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * _Point;
   const double min_dist = MathMax(stops_level, freeze_level);

   if(new_sl <= 0.0)
      return false;

   if(pos_type == POSITION_TYPE_BUY)
   {
      if(new_sl >= bid) return false;
      if(min_dist > 0.0 && (bid - new_sl) < min_dist) return false;
      return true;
   }

   if(pos_type == POSITION_TYPE_SELL)
   {
      if(new_sl <= ask) return false;
      if(min_dist > 0.0 && (new_sl - ask) < min_dist) return false;
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Core: TRY_ENTER                                                  |
//+------------------------------------------------------------------+
void TryEnter(SessionState &s, const BreakoutDir dir, const datetime confirm_bar_time)
{
   if(s.trade_taken)
      return;

   const int tf_sec = PeriodSeconds(Confirm_TF);
   const datetime confirm_close_time = confirm_bar_time + (datetime)tf_sec;

   if(s.window_end_time > 0 && confirm_close_time > s.window_end_time)
   {
      s.state = RESET;
      return;
   }

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
   {
      s.state = RESET;
      return;
   }

   const double entry = (dir == DIR_LONG) ? tick.ask : tick.bid;
   if(entry <= 0.0)
   {
      s.state = RESET;
      return;
   }

   // Structure swing (Confirm_TF candles from retest_start_time to confirm_bar_time inclusive)
   const double pip = PipSize(_Symbol);
   const double buffer = SL_Buffer_Pips * pip;

   double swing_extreme = 0.0;
   if(!GetSwingExtreme(s.retest_start_time, confirm_bar_time, dir, swing_extreme))
   {
      s.state = RESET;
      return;
   }

   double structureSL = 0.0;
   if(dir == DIR_LONG)
      structureSL = swing_extreme - buffer;
   else if(dir == DIR_SHORT)
      structureSL = swing_extreme + buffer;
   else
   {
      s.state = RESET;
      return;
   }

   // ATR SL
   double atr = 0.0;
   if(!GetAtrValue(atr))
   {
      s.state = RESET;
      return;
   }

   double atrSL = 0.0;
   if(dir == DIR_LONG)
      atrSL = entry - (ATR_Mult * atr);
   else
      atrSL = entry + (ATR_Mult * atr);

   // Pick more conservative (farther) SL
   double sl = 0.0;
   if(dir == DIR_LONG)
      sl = MathMin(structureSL, atrSL); // lower = farther
   else
      sl = MathMax(structureSL, atrSL); // higher = farther

   const ENUM_ORDER_TYPE order_type = (dir == DIR_LONG) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   // Normalize SL/TP to tick size/digits
   const double stop_dist = MathAbs(entry - sl);
   if(stop_dist <= 0.0)
   {
      s.state = RESET;
      return;
   }

   const double tp_raw = (dir == DIR_LONG) ? (entry + RR * stop_dist) : (entry - RR * stop_dist);

   sl = NormalizeSLForOrder(order_type, sl);
   double tp = NormalizeTPForOrder(order_type, tp_raw);

   // Validate broker constraints
   if(!ValidateOrderStops(order_type, entry, sl, tp))
   {
      s.state = RESET;
      return;
   }

   // Position sizing
   const double lots = LotsForRisk(MathAbs(entry - sl));
   if(lots <= 0.0)
   {
      s.state = RESET;
      return;
   }

   // Send market order
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(SlippagePoints);

   const string comment = EA_PREFIX + s.name;

   bool sent = false;
   if(dir == DIR_LONG)
      sent = g_trade.Buy(lots, _Symbol, 0.0, sl, tp, comment);
   else
      sent = g_trade.Sell(lots, _Symbol, 0.0, sl, tp, comment);

   if(sent)
   {
      s.trade_taken = true;
      s.state = TRADED;
   }
   else
   {
      PrintFormat("Order send failed (%s). retcode=%d (%s)",
                  s.name, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
      s.state = RESET;
   }
}

//+------------------------------------------------------------------+
//| Per-session workflow on each new closed Confirm_TF bar            |
//+------------------------------------------------------------------+
void ProcessSessionOnClosedBar(SessionState &s,
                              const datetime closed_bar_time,
                              const datetime closed_bar_close_time,
                              const double close_price,
                              const double high_price,
                              const double low_price)
{
   if(!s.enabled)
      return;

   if(s.state == RESET || s.state == TRADED)
      return;

   // Deadline rule
   if(!s.trade_taken && s.window_end_time > 0 && closed_bar_close_time > s.window_end_time)
   {
      s.state = RESET;
      return;
   }

   switch(s.state)
   {
      case BUILD_OR:
      {
         const datetime or_end = s.session_start_server + (datetime)(OR_Minutes * 60);

         if(closed_bar_close_time >= or_end)
         {
            double orh = 0.0, orl = 0.0;
            if(ComputeOpeningRange(s.session_start_server, OR_Minutes, orh, orl))
            {
               s.ORH = orh;
               s.ORL = orl;

               s.OR_close_time   = or_end;
               s.window_end_time = s.OR_close_time + (datetime)(MaxMinutesAfterORClose * 60);

               s.breakout_dir      = DIR_NONE;
               s.breakout_bar_time = 0;
               s.retest_seen       = false;
               s.retest_start_time = 0;

               s.state = WAIT_BREAKOUT;
            }
            // If OR can't be computed yet, keep trying on next bars.
         }
         break;
      }

      case WAIT_BREAKOUT:
      {
         if(close_price > s.ORH)
         {
            SetBreakout(s, DIR_LONG, closed_bar_time);
            s.state = WAIT_RETEST;
         }
         else if(close_price < s.ORL)
         {
            SetBreakout(s, DIR_SHORT, closed_bar_time);
            s.state = WAIT_RETEST;
         }
         break;
      }

      case WAIT_RETEST:
      case WAIT_CONFIRM:
      {
         // Opposite direction allowed before trade
         if(!s.trade_taken)
         {
            if(close_price > s.ORH && s.breakout_dir != DIR_LONG)
            {
               SetBreakout(s, DIR_LONG, closed_bar_time);
               s.state = WAIT_RETEST;
               break;
            }
            if(close_price < s.ORL && s.breakout_dir != DIR_SHORT)
            {
               SetBreakout(s, DIR_SHORT, closed_bar_time);
               s.state = WAIT_RETEST;
               break;
            }
         }

         // Retest must be on a later candle than breakout candle
         if(closed_bar_time <= s.breakout_bar_time)
            break;

         if(!s.retest_seen)
         {
            if(s.breakout_dir == DIR_LONG)
            {
               if(low_price <= s.ORH)
               {
                  s.retest_seen       = true;
                  s.retest_start_time = closed_bar_time;
                  s.state             = WAIT_CONFIRM;

                  // Same candle confirm
                  if(close_price > s.ORH)
                     TryEnter(s, DIR_LONG, closed_bar_time);
               }
            }
            else if(s.breakout_dir == DIR_SHORT)
            {
               if(high_price >= s.ORL)
               {
                  s.retest_seen       = true;
                  s.retest_start_time = closed_bar_time;
                  s.state             = WAIT_CONFIRM;

                  // Same candle confirm
                  if(close_price < s.ORL)
                     TryEnter(s, DIR_SHORT, closed_bar_time);
               }
            }
         }
         else
         {
            // Retest already seen -> wait for confirm
            if(s.breakout_dir == DIR_LONG)
            {
               if(close_price > s.ORH)
                  TryEnter(s, DIR_LONG, closed_bar_time);
            }
            else if(s.breakout_dir == DIR_SHORT)
            {
               if(close_price < s.ORL)
                  TryEnter(s, DIR_SHORT, closed_bar_time);
            }
         }
         break;
      }

      default:
         break;
   }
}

//+------------------------------------------------------------------+
//| Trailing management (OnTick/OnTimer)                              |
//+------------------------------------------------------------------+
void ManageTrailing()
{
   if(!g_trailing_enabled)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
      return;

   const double pip = PipSize(_Symbol);
   if(pip <= 0.0)
      return;

   const int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      const string sym = PositionGetString(POSITION_SYMBOL);
      if(sym != _Symbol)
         continue;

      const long magic = (long)PositionGetInteger(POSITION_MAGIC);
      if(magic != MagicNumber)
         continue;

      const ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      const double curSL = PositionGetDouble(POSITION_SL);
      const double curTP = PositionGetDouble(POSITION_TP);

      if(entry <= 0.0)
         continue;

      double profit_pips = 0.0;
      if(ptype == POSITION_TYPE_BUY)
         profit_pips = (tick.bid - entry) / pip;
      else if(ptype == POSITION_TYPE_SELL)
         profit_pips = (entry - tick.ask) / pip;
      else
         continue;

      if(profit_pips < Trail_TriggerProfitPips)
         continue;

      double lockSL = 0.0, trailSL = 0.0, newSL = 0.0;

      if(ptype == POSITION_TYPE_BUY)
      {
         lockSL  = entry + Trail_LockInPips * pip;
         trailSL = tick.bid - Trail_DistancePips * pip;

         const double base = (curSL > 0.0) ? curSL : -DBL_MAX;
         newSL = MathMax(base, lockSL, trailSL);

         // Normalize for BUY SL (round down)
         newSL = RoundToTick(_Symbol, newSL, false);

         // Must improve
         if(curSL > 0.0 && newSL <= curSL + (TickSize(_Symbol) * 0.1))
            continue;

         if(!TrailingAllowsSL(ptype, newSL, tick.bid, tick.ask))
            continue;

         ModifyPositionSLTP(ticket, newSL, curTP);
      }
      else if(ptype == POSITION_TYPE_SELL)
      {
         lockSL  = entry - Trail_LockInPips * pip;
         trailSL = tick.ask + Trail_DistancePips * pip;

         const double base = (curSL > 0.0) ? curSL : DBL_MAX;
         newSL = MathMin(base, lockSL, trailSL);

         // Normalize for SELL SL (round up)
         newSL = RoundToTick(_Symbol, newSL, true);

         // Must improve (lower SL)
         if(curSL > 0.0 && newSL >= curSL - (TickSize(_Symbol) * 0.1))
            continue;

         if(!TrailingAllowsSL(ptype, newSL, tick.bid, tick.ask))
            continue;

         ModifyPositionSLTP(ticket, newSL, curTP);
      }
   }
}

//+------------------------------------------------------------------+
//| New closed bar detection                                          |
//+------------------------------------------------------------------+
void ProcessNewClosedBarIfAny()
{
   MqlRates r[];
   if(CopyRates(_Symbol, Confirm_TF, 1, 1, r) != 1)
      return;

   const datetime bar_time = r[0].time;
   if(bar_time <= 0)
      return;

   if(bar_time == g_last_confirm_closed_bar_time)
      return;

   g_last_confirm_closed_bar_time = bar_time;

   const int tf_sec = PeriodSeconds(Confirm_TF);
   const datetime bar_close_time = bar_time + (datetime)tf_sec;

   const double close_price = r[0].close;
   const double high_price  = r[0].high;
   const double low_price   = r[0].low;

   for(int i = 0; i < 3; i++)
      ProcessSessionOnClosedBar(g_sessions[i], bar_time, bar_close_time, close_price, high_price, low_price);
}

void Pulse()
{
   // UTC day change detection
   const int day_key = UtcDayKey(TimeGMT());
   if(day_key != g_last_utc_day_key)
   {
      g_last_utc_day_key = day_key;
      SetupNewUtcDay();
   }

   ProcessNewClosedBarIfAny();
   ManageTrailing();
}

//+------------------------------------------------------------------+
//| Event Handlers                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Validate key inputs
   if(OR_Minutes != 5 && OR_Minutes != 15 && OR_Minutes != 30)
   {
      Print("Invalid OR_Minutes. Allowed: 5, 15, 30.");
      return INIT_PARAMETERS_INCORRECT;
   }

   if(Confirm_TF != PERIOD_M1 && Confirm_TF != PERIOD_M3 && Confirm_TF != PERIOD_M5)
   {
      Print("Invalid Confirm_TF. Allowed: M1, M3, M5.");
      return INIT_PARAMETERS_INCORRECT;
   }

   if(RiskUSD <= 0.0 || RR <= 0.0)
   {
      Print("Invalid RiskUSD or RR. Must be > 0.");
      return INIT_PARAMETERS_INCORRECT;
   }

   if(ATR_Period <= 0 || ATR_Mult <= 0.0)
   {
      Print("Invalid ATR settings. ATR_Period and ATR_Mult must be > 0.");
      return INIT_PARAMETERS_INCORRECT;
   }

   // Session static config
   g_sessions[0].name = "Midnight";
   g_sessions[0].start_hour_utc = 0;
   g_sessions[0].start_min_utc  = 0;

   g_sessions[1].name = "London";
   g_sessions[1].start_hour_utc = 7;
   g_sessions[1].start_min_utc  = 0;

   g_sessions[2].name = "NY";
   g_sessions[2].start_hour_utc = 12;
   g_sessions[2].start_min_utc  = 0;

   // ATR handle
   g_atr_handle = iATR(_Symbol, ATR_TF, ATR_Period);
   if(g_atr_handle == INVALID_HANDLE)
   {
      PrintFormat("Failed to create ATR handle. err=%d", _LastError);
      return INIT_FAILED;
   }

   // Initialize day/session states
   g_last_utc_day_key = UtcDayKey(TimeGMT());
   SetupNewUtcDay();

   // Create trailing toggle button
   CreateTrailingButton();

   // Use a timer so the EA logic still runs even with low tick frequency
   EventSetTimer(1);

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();

   if(ObjectFind(0, BTN_TRAIL) >= 0)
      ObjectDelete(0, BTN_TRAIL);

   if(g_atr_handle != INVALID_HANDLE)
   {
      IndicatorRelease(g_atr_handle);
      g_atr_handle = INVALID_HANDLE;
   }

   Comment("");
}

void OnTick()
{
   Pulse();
}

void OnTimer()
{
   Pulse();
}

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CLICK && sparam == BTN_TRAIL)
   {
      // OBJ_BUTTON is a two-position switch by default; we use its state as the toggle.
      g_trailing_enabled = (bool)ObjectGetInteger(0, BTN_TRAIL, OBJPROP_STATE);
      UpdateTrailingButton();
   }
}
//+------------------------------------------------------------------+