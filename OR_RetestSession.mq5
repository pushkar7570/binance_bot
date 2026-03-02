//+------------------------------------------------------------------+
//| OR Retest Session EA (MT5 / MQL5)                                 |
//| Implements: OR build (M1), breakout, retest, confirm, risk sizing |
//| Trailing SL with runtime button toggle                            |
//+------------------------------------------------------------------+
#property version   "1.01"
#property strict

//--- inputs
input ulong  InpMagicNumber              = 26022026;
input int    InpDeviationPoints          = 20;

//--- Session toggles
input bool   Trade_Midnight              = true;   // 00:00 UTC
input bool   Trade_London                = true;   // 07:00 UTC
input bool   Trade_NY                    = true;   // 12:00 UTC

//--- OR minutes (allowed: 5, 15, 30)
enum ENUM_OR_MINUTES
{
   OR_5  = 5,
   OR_15 = 15,
   OR_30 = 30
};
input ENUM_OR_MINUTES InpOR_Minutes      = OR_5;

//--- Confirm timeframe (allowed: M1, M3, M5)
enum ENUM_CONFIRM_TF
{
   CONFIRM_M1 = PERIOD_M1,
   CONFIRM_M3 = PERIOD_M3,
   CONFIRM_M5 = PERIOD_M5
};
input ENUM_CONFIRM_TF InpConfirmTF       = CONFIRM_M5;

//--- Window and risk
input int    MaxMinutesAfterORClose      = 105;
input double RiskUSD                     = 100.0;
input double RR                          = 2.0;
input double SL_Buffer_Pips              = 1.0;

//--- Trailing settings (runtime toggle via button)
input bool   TrailingEnabledAtStart      = false;
input double Trail_TriggerProfitPips     = 10.0;
input double Trail_LockInPips            = 4.0;
input double Trail_DistancePips          = 10.0;

//+------------------------------------------------------------------+
//| Enums / Structs                                                   |
//+------------------------------------------------------------------+
enum SessionState
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

struct SessionData
{
   string       name;
   bool         enabled;

   datetime     start_utc;              // session start in UTC for current UTC day

   bool         start_server_locked;
   datetime     start_server;

   SessionState state;

   double       or_high;
   double       or_low;
   datetime     or_close_time;
   datetime     window_end_time;

   BreakoutDir  dir;
   datetime     breakout_bar_time;

   bool         retest_seen;
   datetime     retest_start_time;

   bool         trade_taken;
};

//+------------------------------------------------------------------+
//| Globals                                                           |
//+------------------------------------------------------------------+
SessionData      g_sessions[3];
datetime         g_last_utc_day_start     = 0;
datetime         g_last_confirm_bar_time  = 0;

bool             g_trailing_enabled       = false;
string           g_btn_name               = "OR_EA_TrailToggleBtn";

ENUM_TIMEFRAMES  g_confirm_tf;

//+------------------------------------------------------------------+
//| Utility: UTC day start                                            |
//+------------------------------------------------------------------+
datetime UtcDayStart(const datetime utc_time)
{
   MqlDateTime dt;
   TimeToStruct(utc_time, dt);
   dt.hour = 0;
   dt.min  = 0;
   dt.sec  = 0;
   return StructToTime(dt);
}

//+------------------------------------------------------------------+
//| Utility: current server UTC offset (seconds)                      |
//+------------------------------------------------------------------+
int ServerUtcOffsetSeconds()
{
   return (int)(TimeCurrent() - TimeGMT());
}

//+------------------------------------------------------------------+
//| Utility: pip size (common CFD/FX convention)                      |
//+------------------------------------------------------------------+
double PipSize()
{
   const double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   const int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(digits == 3 || digits == 5)
      return point * 10.0;

   return point;
}

//+------------------------------------------------------------------+
//| Utility: normalize price                                          |
//+------------------------------------------------------------------+
double NormPrice(const double price)
{
   const int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
}

//+------------------------------------------------------------------+
//| Utility: normalize volume DOWN (no SYMBOL_VOLUME_DIGITS needed)   |
//+------------------------------------------------------------------+
double NormalizeVolumeDown(const double volume)
{
   const double vmin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   const double vmax  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   const double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(vstep <= 0.0)
      return 0.0;

   if(volume < vmin)
      return 0.0;

   double v = volume;
   if(v > vmax)
      v = vmax;

   // Round down to step, anchored at vmin (robust even if vmin not multiple of step)
   double steps = MathFloor((v - vmin) / vstep);
   double out   = vmin + steps * vstep;

   if(out < vmin) out = vmin;
   if(out > vmax) out = vmax;

   // Two decimals is safe; step rounding enforces real precision anyway
   out = NormalizeDouble(out, 2);

   if(out < vmin)
      return 0.0;

   return out;
}

//+------------------------------------------------------------------+
//| Compute OR High/Low from M1 candles in [start, start+minutes)     |
//+------------------------------------------------------------------+
bool ComputeOpeningRange(const datetime start_server, const int or_minutes, double &orh, double &orl)
{
   datetime end_server = start_server + (datetime)(or_minutes * 60);

   MqlRates rates[];
   ArraySetAsSeries(rates, false);

   int copied = CopyRates(_Symbol, PERIOD_M1, start_server, end_server - 1, rates);

   if(copied < or_minutes)
      return false;

   orh = rates[0].high;
   orl = rates[0].low;

   for(int i = 1; i < copied; i++)
   {
      if(rates[i].high > orh) orh = rates[i].high;
      if(rates[i].low  < orl) orl = rates[i].low;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Trading permission checks                                         |
//+------------------------------------------------------------------+
bool CanTradeNow()
{
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
      return false;
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      return false;
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
      return false;

   long mode = 0;
   if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE, mode))
      return false;

   if(mode == SYMBOL_TRADE_MODE_DISABLED)
      return false;

   return true;
}

//+------------------------------------------------------------------+
//| Send market order (TRADE_ACTION_DEAL)                             |
//+------------------------------------------------------------------+
bool SendMarketOrder(const ENUM_ORDER_TYPE type,
                     const double volume,
                     const double sl,
                     const double tp,
                     const string comment)
{
   if(!CanTradeNow())
   {
      Print("Trading not allowed/connected.");
      return false;
   }

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
   {
      Print("SymbolInfoTick failed.");
      return false;
   }

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);

   req.action       = TRADE_ACTION_DEAL;
   req.symbol       = _Symbol;
   req.magic        = InpMagicNumber;
   req.deviation    = InpDeviationPoints;
   req.type         = type;
   req.volume       = volume;
   req.price        = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;
   req.sl           = sl;
   req.tp           = tp;
   req.type_time    = ORDER_TIME_GTC;

   // Filling mode (cast safe)
   req.type_filling = (ENUM_ORDER_TYPE_FILLING)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);

   req.comment      = comment;

   ResetLastError();
   if(!OrderSend(req, res))
   {
      Print(StringFormat("OrderSend failed. err=%d", GetLastError()));
      return false;
   }

   if(res.retcode == TRADE_RETCODE_DONE || res.retcode == TRADE_RETCODE_DONE_PARTIAL)
   {
      Print(StringFormat("Trade DONE. ret=%u order=%I64u deal=%I64u pos=%I64u",
                         res.retcode, res.order, res.deal, res.position));
      return true;
   }

   Print(StringFormat("Trade rejected. ret=%u comment=%s", res.retcode, res.comment));
   return false;
}

//+------------------------------------------------------------------+
//| Modify position SL/TP (TRADE_ACTION_SLTP)                         |
//+------------------------------------------------------------------+
bool ModifyPositionSLTP(const ulong position_ticket, const double new_sl, const double new_tp)
{
   if(!CanTradeNow())
      return false;

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);

   req.action   = TRADE_ACTION_SLTP;
   req.symbol   = _Symbol;
   req.position = position_ticket;
   req.magic    = InpMagicNumber;
   req.sl       = new_sl;
   req.tp       = new_tp;

   ResetLastError();
   if(!OrderSend(req, res))
   {
      Print(StringFormat("SLTP modify OrderSend failed. err=%d", GetLastError()));
      return false;
   }

   if(res.retcode == TRADE_RETCODE_DONE)
      return true;

   Print(StringFormat("SLTP modify rejected. ret=%u comment=%s", res.retcode, res.comment));
   return false;
}

//+------------------------------------------------------------------+
//| Risk-based lot size using OrderCalcProfit                         |
//+------------------------------------------------------------------+
double CalcVolumeForRisk(const BreakoutDir dir, const double entry, const double sl)
{
   if(RiskUSD <= 0.0)
      return 0.0;

   ENUM_ORDER_TYPE order_type = (dir == DIR_LONG) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   double loss_per_1_lot = 0.0;
   ResetLastError();
   if(!OrderCalcProfit(order_type, _Symbol, 1.0, entry, sl, loss_per_1_lot))
   {
      Print(StringFormat("OrderCalcProfit failed. err=%d", GetLastError()));
      return 0.0;
   }

   loss_per_1_lot = MathAbs(loss_per_1_lot);
   if(loss_per_1_lot <= 0.0)
      return 0.0;

   double raw_vol = RiskUSD / loss_per_1_lot;
   return NormalizeVolumeDown(raw_vol);
}

//+------------------------------------------------------------------+
//| TryEnter(direction)                                               |
//+------------------------------------------------------------------+
void TryEnter(SessionData &s, const BreakoutDir direction)
{
   if(s.trade_taken)
      return;

   datetime now_server = TimeCurrent();
   if(s.window_end_time > 0 && now_server > s.window_end_time)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
      return;

   const double pip    = PipSize();
   const double buffer = SL_Buffer_Pips * pip;

   double entry = (direction == DIR_LONG) ? tick.ask : tick.bid;

   double sl = 0.0;
   if(direction == DIR_LONG)
      sl = s.or_low - buffer;
   else
      sl = s.or_high + buffer;

   if(direction == DIR_LONG && sl >= entry) return;
   if(direction == DIR_SHORT && sl <= entry) return;

   double Rdist = MathAbs(entry - sl);
   if(Rdist <= 0.0)
      return;

   double tp = 0.0;
   if(direction == DIR_LONG)
      tp = entry + RR * Rdist;
   else
      tp = entry - RR * Rdist;

   sl = NormPrice(sl);
   tp = NormPrice(tp);

   const double point       = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   const double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   const double freeze_level= (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * point;
   const double min_dist    = MathMax(stops_level, freeze_level);

   double ref_price = (direction == DIR_LONG) ? tick.bid : tick.ask;

   if(direction == DIR_LONG)
   {
      if((ref_price - sl) < min_dist) return;
      if((tp - ref_price) < min_dist) return;
   }
   else
   {
      if((sl - ref_price) < min_dist) return;
      if((ref_price - tp) < min_dist) return;
   }

   double volume = CalcVolumeForRisk(direction, entry, sl);
   if(volume <= 0.0)
      return;

   ENUM_ORDER_TYPE order_type = (direction == DIR_LONG) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   string cmt = StringFormat("OR%d %s", (int)InpOR_Minutes, s.name);

   bool ok = SendMarketOrder(order_type, volume, sl, tp, cmt);

   if(ok)
   {
      s.trade_taken = true;
      s.state       = TRADED;
   }
   else
   {
      s.state       = RESET;
   }
}

//+------------------------------------------------------------------+
//| Ensure session start server locked                                |
//+------------------------------------------------------------------+
bool EnsureSessionStartServerLocked(SessionData &s)
{
   if(!s.enabled)
      return false;

   if(s.start_server_locked)
      return true;

   int offset = ServerUtcOffsetSeconds();
   datetime candidate = s.start_utc + offset;

   if(TimeCurrent() >= candidate)
   {
      s.start_server = candidate;
      s.start_server_locked = true;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Per-session processing on each new closed Confirm_TF bar           |
//+------------------------------------------------------------------+
void ProcessSessionOnClosedBar(SessionData &s,
                               const datetime bar_time,
                               const double bar_close,
                               const double bar_high,
                               const double bar_low,
                               const datetime now_server)
{
   if(!s.enabled)
      return;

   if(!EnsureSessionStartServerLocked(s))
      return;

   if(!s.trade_taken && s.window_end_time > 0 && now_server > s.window_end_time)
   {
      s.state = RESET;
      return;
   }

   switch(s.state)
   {
      case BUILD_OR:
      {
         datetime or_close = s.start_server + (datetime)((int)InpOR_Minutes * 60);

         if(now_server >= or_close)
         {
            double orh=0.0, orl=0.0;
            if(ComputeOpeningRange(s.start_server, (int)InpOR_Minutes, orh, orl))
            {
               s.or_high = orh;
               s.or_low  = orl;

               s.or_close_time    = or_close;
               s.window_end_time  = s.or_close_time + (datetime)(MaxMinutesAfterORClose * 60);

               s.dir              = DIR_NONE;
               s.breakout_bar_time= 0;
               s.retest_seen      = false;
               s.retest_start_time= 0;
               s.trade_taken      = false;

               if(now_server > s.window_end_time)
                  s.state = RESET;
               else
                  s.state = WAIT_BREAKOUT;
            }
         }
         break;
      }

      case WAIT_BREAKOUT:
      {
         if(bar_time < s.or_close_time)
            break;

         if(!s.trade_taken && now_server > s.window_end_time)
         {
            s.state = RESET;
            break;
         }

         if(bar_close > s.or_high)
         {
            s.dir = DIR_LONG;
            s.breakout_bar_time = bar_time;
            s.retest_seen = false;
            s.retest_start_time = 0;
            s.state = WAIT_RETEST;
         }
         else if(bar_close < s.or_low)
         {
            s.dir = DIR_SHORT;
            s.breakout_bar_time = bar_time;
            s.retest_seen = false;
            s.retest_start_time = 0;
            s.state = WAIT_RETEST;
         }
         break;
      }

      case WAIT_RETEST:
      case WAIT_CONFIRM:
      {
         if(bar_time < s.or_close_time)
            break;

         if(!s.trade_taken && now_server > s.window_end_time)
         {
            s.state = RESET;
            break;
         }

         if(s.trade_taken)
         {
            s.state = TRADED;
            break;
         }

         // Opposite direction allowed (before trade):
         if(bar_close > s.or_high && s.dir != DIR_LONG)
         {
            s.dir = DIR_LONG;
            s.breakout_bar_time = bar_time;
            s.retest_seen = false;
            s.retest_start_time = 0;
            s.state = WAIT_RETEST;
            break;
         }
         if(bar_close < s.or_low && s.dir != DIR_SHORT)
         {
            s.dir = DIR_SHORT;
            s.breakout_bar_time = bar_time;
            s.retest_seen = false;
            s.retest_start_time = 0;
            s.state = WAIT_RETEST;
            break;
         }

         if(bar_time <= s.breakout_bar_time)
            break;

         if(!s.retest_seen)
         {
            if(s.dir == DIR_LONG)
            {
               if(bar_low <= s.or_high)
               {
                  s.retest_seen = true;
                  s.retest_start_time = bar_time;
                  s.state = WAIT_CONFIRM;

                  if(bar_close > s.or_high)
                     TryEnter(s, DIR_LONG);
               }
            }
            else if(s.dir == DIR_SHORT)
            {
               if(bar_high >= s.or_low)
               {
                  s.retest_seen = true;
                  s.retest_start_time = bar_time;
                  s.state = WAIT_CONFIRM;

                  if(bar_close < s.or_low)
                     TryEnter(s, DIR_SHORT);
               }
            }
         }
         else
         {
            if(s.dir == DIR_LONG)
            {
               if(bar_close > s.or_high)
                  TryEnter(s, DIR_LONG);
            }
            else if(s.dir == DIR_SHORT)
            {
               if(bar_close < s.or_low)
                  TryEnter(s, DIR_SHORT);
            }
         }
         break;
      }

      case TRADED:
      case RESET:
      default:
         break;
   }
}

//+------------------------------------------------------------------+
//| Trailing management (OnTimer)                                     |
//+------------------------------------------------------------------+
void ManageTrailing()
{
   if(!g_trailing_enabled)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
      return;

   const double pip   = PipSize();
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   const double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   const double freeze_level= (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * point;
   const double min_dist    = MathMax(stops_level, freeze_level);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket_pos = PositionGetTicket(i);
      if(ticket_pos == 0)
         continue;

      if(!PositionSelectByTicket(ticket_pos))
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      if(sym != _Symbol)
         continue;

      long magic = PositionGetInteger(POSITION_MAGIC);
      if((ulong)magic != InpMagicNumber)
         continue;

      long type  = PositionGetInteger(POSITION_TYPE);
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double tp    = PositionGetDouble(POSITION_TP);

      double profit_pips = 0.0;
      if(type == POSITION_TYPE_BUY)
         profit_pips = (tick.bid - entry) / pip;
      else if(type == POSITION_TYPE_SELL)
         profit_pips = (entry - tick.ask) / pip;
      else
         continue;

      if(profit_pips < Trail_TriggerProfitPips)
         continue;

      if(type == POSITION_TYPE_BUY)
      {
         double lockSL  = entry + Trail_LockInPips * pip;
         double trailSL = tick.bid - Trail_DistancePips * pip;

         double candidate = trailSL;
         if(candidate < lockSL) candidate = lockSL;
         if(sl != 0.0 && candidate < sl) candidate = sl;

         candidate = NormPrice(candidate);

         if(sl == 0.0 || candidate > sl + point*0.5)
         {
            if((tick.bid - candidate) >= min_dist)
               ModifyPositionSLTP(ticket_pos, candidate, tp);
         }
      }
      else if(type == POSITION_TYPE_SELL)
      {
         double lockSL  = entry - Trail_LockInPips * pip;
         double trailSL = tick.ask + Trail_DistancePips * pip;

         double candidate = trailSL;
         if(candidate > lockSL) candidate = lockSL;
         if(sl != 0.0 && candidate > sl) candidate = sl;

         candidate = NormPrice(candidate);

         if(sl == 0.0 || candidate < sl - point*0.5)
         {
            if((candidate - tick.ask) >= min_dist)
               ModifyPositionSLTP(ticket_pos, candidate, tp);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Button UI                                                         |
//+------------------------------------------------------------------+
void UpdateTrailingButton()
{
   if(ObjectFind(0, g_btn_name) < 0)
      return;

   string txt = g_trailing_enabled ? "Trailing: ON" : "Trailing: OFF";
   ObjectSetString(0, g_btn_name, OBJPROP_TEXT, txt);

   ObjectSetInteger(0, g_btn_name, OBJPROP_BGCOLOR, g_trailing_enabled ? clrLime : clrTomato);
   ObjectSetInteger(0, g_btn_name, OBJPROP_COLOR, clrWhite);
}

void CreateTrailingButton()
{
   if(ObjectFind(0, g_btn_name) >= 0)
      ObjectDelete(0, g_btn_name);

   ObjectCreate(0, g_btn_name, OBJ_BUTTON, 0, 0, 0);

   ObjectSetInteger(0, g_btn_name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, g_btn_name, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, g_btn_name, OBJPROP_YDISTANCE, 20);
   ObjectSetInteger(0, g_btn_name, OBJPROP_XSIZE, 130);
   ObjectSetInteger(0, g_btn_name, OBJPROP_YSIZE, 28);

   ObjectSetInteger(0, g_btn_name, OBJPROP_HIDDEN, false);
   ObjectSetInteger(0, g_btn_name, OBJPROP_SELECTABLE, true);
   ObjectSetInteger(0, g_btn_name, OBJPROP_SELECTED, false);
   ObjectSetInteger(0, g_btn_name, OBJPROP_FONTSIZE, 10);

   UpdateTrailingButton();
}

//+------------------------------------------------------------------+
//| Reset session for new UTC day                                     |
//+------------------------------------------------------------------+
void ResetSessionForNewDay(SessionData &s)
{
   s.start_server_locked = false;
   s.start_server        = 0;

   s.state               = s.enabled ? BUILD_OR : RESET;

   s.or_high             = 0.0;
   s.or_low              = 0.0;
   s.or_close_time       = 0;
   s.window_end_time     = 0;

   s.dir                 = DIR_NONE;
   s.breakout_bar_time   = 0;

   s.retest_seen         = false;
   s.retest_start_time   = 0;

   s.trade_taken         = false;
}

//+------------------------------------------------------------------+
//| Set session UTC start times for the current UTC day               |
//+------------------------------------------------------------------+
void SetSessionUtcStartsForDay(const datetime utc_day_start)
{
   g_sessions[0].start_utc = utc_day_start + (datetime)(0  * 3600);
   g_sessions[1].start_utc = utc_day_start + (datetime)(7  * 3600);
   g_sessions[2].start_utc = utc_day_start + (datetime)(12 * 3600);
}

//+------------------------------------------------------------------+
//| Handle new UTC day                                                |
//+------------------------------------------------------------------+
void HandleNewUtcDay()
{
   datetime utc_now = TimeGMT();
   datetime utc_day = UtcDayStart(utc_now);

   if(utc_day == g_last_utc_day_start)
      return;

   g_last_utc_day_start = utc_day;

   SetSessionUtcStartsForDay(utc_day);

   for(int i=0; i<3; i++)
      ResetSessionForNewDay(g_sessions[i]);

   Print("New UTC day detected. Sessions reset.");
}

//+------------------------------------------------------------------+
//| Handle new closed bar on confirm TF                               |
//+------------------------------------------------------------------+
void HandleNewClosedConfirmBar()
{
   datetime bar_time = iTime(_Symbol, g_confirm_tf, 1);
   if(bar_time <= 0)
      return;

   if(bar_time == g_last_confirm_bar_time)
      return;

   g_last_confirm_bar_time = bar_time;

   double close = iClose(_Symbol, g_confirm_tf, 1);
   double high  = iHigh (_Symbol, g_confirm_tf, 1);
   double low   = iLow  (_Symbol, g_confirm_tf, 1);

   datetime now_server = TimeCurrent();

   for(int i=0; i<3; i++)
      ProcessSessionOnClosedBar(g_sessions[i], bar_time, close, high, low, now_server);
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   // Explicit cast fixes "cannot convert to enum" on stricter compilers
   g_confirm_tf = (ENUM_TIMEFRAMES)(int)InpConfirmTF;

   g_sessions[0].name    = "Midnight";
   g_sessions[0].enabled = Trade_Midnight;

   g_sessions[1].name    = "London";
   g_sessions[1].enabled = Trade_London;

   g_sessions[2].name    = "NewYork";
   g_sessions[2].enabled = Trade_NY;

   g_trailing_enabled = TrailingEnabledAtStart;

   g_last_utc_day_start = UtcDayStart(TimeGMT());
   SetSessionUtcStartsForDay(g_last_utc_day_start);

   for(int i=0; i<3; i++)
      ResetSessionForNewDay(g_sessions[i]);

   CreateTrailingButton();

   EventSetTimer(1);

   Print("EA initialized.");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();

   if(ObjectFind(0, g_btn_name) >= 0)
      ObjectDelete(0, g_btn_name);

   Print("EA deinitialized.");
}

//+------------------------------------------------------------------+
//| Timer event                                                       |
//+------------------------------------------------------------------+
void OnTimer()
{
   HandleNewUtcDay();
   HandleNewClosedConfirmBar();
   ManageTrailing();
}

//+------------------------------------------------------------------+
//| Chart event (button toggle)                                       |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CLICK && sparam == g_btn_name)
   {
      g_trailing_enabled = !g_trailing_enabled;
      UpdateTrailingButton();
      Print(StringFormat("Trailing toggled: %s", g_trailing_enabled ? "ON" : "OFF"));
   }
}
//+------------------------------------------------------------------+