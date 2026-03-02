//+------------------------------------------------------------------+
//|                                                OR_Retest_ATR.mq5  |
//|                        Opening Range Breakout (Retest + Confirm)  |
//|                                       ATR Volatility Stop + RR TP |
//|                                       Trailing SL (toggle button) |
//+------------------------------------------------------------------+
#property copyright ""
#property link      ""
#property version   "1.00"
#property description "Opening Range Breakout (M1 OR) with breakout->retest->confirm entry."
#property description "Sessions are defined in UTC and converted to broker/server time."
#property description "StopLoss uses ATR volatility stop (adaptive). Position sizing risks fixed money."
#property description "Optional trailing stop managed by a chart button."

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+

//--- Session toggles
input bool   Trade_Midnight              = true;   // Trade Midnight session (UTC)
input bool   Trade_London                = true;   // Trade London session (UTC)
input bool   Trade_NY                    = true;   // Trade New York session (UTC)

//--- Session start times (UTC)
input int    Midnight_UTC_Hour           = 0;      // 0..23
input int    Midnight_UTC_Minute         = 0;      // 0..59
input int    London_UTC_Hour             = 7;      // 0..23
input int    London_UTC_Minute           = 0;      // 0..59
input int    NY_UTC_Hour                 = 12;     // 0..23
input int    NY_UTC_Minute               = 0;      // 0..59

//--- OR & signal settings
input int              OR_Minutes        = 15;     // {5,15,30}  (OR computed from M1 candles)
input ENUM_TIMEFRAMES  Confirm_TF        = PERIOD_M5; // {PERIOD_M1, PERIOD_M3, PERIOD_M5}
input int              MaxMinutesAfterORClose = 105;   // Max minutes after OR close to allow entries

//--- Risk & RR
input double RiskUSD                      = 100.0; // Risk per trade in account currency (assumed USD)
input double RR                           = 2.0;    // Risk:Reward multiplier
input double SL_Buffer_Pips               = 0.0;    // Optional deterministic SL buffer (pips)

//--- Trailing (runtime toggle via button)
input bool   TrailingEnabledInitial       = true;  // Initial state (button can toggle at runtime)
input double Trail_TriggerProfitPips      = 10.0;  // Start trailing after this profit (pips)
input double Trail_LockInPips             = 4.0;   // Lock-in profit (pips) once trailing triggers
input double Trail_DistancePips           = 10.0;  // Trail distance from price (pips)

//--- ATR Volatility Stop (Adaptive)
input int              ATR_Period         = 14;
input ENUM_TIMEFRAMES  ATR_TF             = PERIOD_M15; // Can be same as Confirm_TF or fixed TF
input double           ATR_Mult           = 2.0;        // e.g., 1.5–3.0

//--- Trade settings
input ulong  MagicNumber                  = 17022026;
input int    DeviationPoints              = 20;     // Max deviation in points for market orders
input bool   EnableLogs                   = true;  // Print state transitions & trade diagnostics

//+------------------------------------------------------------------+
//| Enums & Structs                                                  |
//+------------------------------------------------------------------+
enum SESSION_STATE
{
   BUILD_OR = 0,
   WAIT_BREAKOUT,
   WAIT_RETEST,
   WAIT_CONFIRM,
   TRADED,
   RESET
};

enum BREAKOUT_DIR
{
   DIR_NONE = 0,
   DIR_LONG,
   DIR_SHORT
};

struct SessionContext
{
   string        name;
   bool          enabled;

   int           utc_hour;
   int           utc_minute;

   datetime      start_utc;
   datetime      start_server;

   SESSION_STATE state;

   double        ORH;
   double        ORL;
   datetime      OR_close_time;
   datetime      window_end_time;

   BREAKOUT_DIR  breakout_dir;
   datetime      breakout_bar_time;

   bool          retest_seen;
   datetime      retest_start_time;

   bool          trade_taken;
};

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CTrade          g_trade;

int             g_atr_handle                 = INVALID_HANDLE;

bool            g_trailing_enabled           = false;

datetime        g_last_confirm_closed_time   = 0;
datetime        g_last_utc_day_start         = 0;

int             g_server_utc_offset_sec      = 0;

int             g_volume_digits              = 2;

SessionContext  g_sessions[3];

//--- UI
string          g_btn_trail_name             = "OR_Retest_ATR_TrailingToggle";

//+------------------------------------------------------------------+
//| Utility: safe server time / UTC offset                           |
//+------------------------------------------------------------------+
datetime ServerTime()
{
   datetime t=TimeTradeServer();
   if(t<=0)
      t=TimeCurrent();
   return t;
}

int GetServerUtcOffsetSeconds()
{
   datetime srv=ServerTime();
   datetime gmt=TimeGMT();
   if(srv<=0 || gmt<=0)
      return 0;
   return (int)(srv-gmt);
}

//+------------------------------------------------------------------+
//| Utility: start of day (UTC)                                      |
//+------------------------------------------------------------------+
datetime DayStartUTC(const datetime t_gmt)
{
   MqlDateTime dt;
   TimeToStruct(t_gmt,dt);
   dt.hour=0;
   dt.min=0;
   dt.sec=0;
   return StructToTime(dt);
}

//+------------------------------------------------------------------+
//| Utility: pip size                                                |
//+------------------------------------------------------------------+
double PipSize(const string symbol)
{
   const int digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   const double point=SymbolInfoDouble(symbol,SYMBOL_POINT);

   // Common convention: 5/3-digit quotes -> 1 pip = 10 points, otherwise 1 pip = 1 point
   if(digits==3 || digits==5)
      return 10.0*point;

   return point;
}

//+------------------------------------------------------------------+
//| Utility: normalize volume to symbol constraints                   |
//+------------------------------------------------------------------+
double NormalizeVolume(const string symbol,const double volume)
{
   const double vmin = SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN);
   const double vmax = SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX);
   const double vstep= SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP);

   if(vmin<=0.0 || vmax<=0.0 || vstep<=0.0)
      return 0.0;

   double vol=volume;

   // floor to step so risk is not exceeded
   vol = MathFloor(vol/vstep)*vstep;

   if(vol<vmin)
      return 0.0;

   if(vol>vmax)
      vol=vmax;

   vol = NormalizeDouble(vol,g_volume_digits);
   return vol;
}

//+------------------------------------------------------------------+
//| Utility: check if algo trading is allowed                         |
//+------------------------------------------------------------------+
bool IsTradingAllowed()
{
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
      return false;

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      return false;

   if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))
      return false;

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
      return false;

   return true;
}

//+------------------------------------------------------------------+
//| Utility: check if account is hedging mode                         |
//+------------------------------------------------------------------+
bool IsHedgingAccount()
{
   const long mode=AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   return (mode==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
}

//+------------------------------------------------------------------+
//| Utility: find if ANY position exists on this symbol (netting safe)|
//+------------------------------------------------------------------+
bool AnyOpenPositionOnSymbol(const string symbol)
{
   if(!PositionSelect(symbol))
      return false;

   // In netting accounts, PositionSelect(symbol) selects the one position (if any).
   // In hedging accounts, this only selects one position, so we still treat as "exists".
   return true;
}

//+------------------------------------------------------------------+
//| Utility: compute OR from M1 candles                               |
//+------------------------------------------------------------------+
bool ComputeOpeningRange(const string symbol,
                         const datetime start_server,
                         const int or_minutes,
                         double &or_high,
                         double &or_low)
{
   or_high=0.0;
   or_low =0.0;

   if(or_minutes<=0)
      return false;

   MqlRates rates[];
   ArraySetAsSeries(rates,false);

   ResetLastError();
   const int copied=CopyRates(symbol,PERIOD_M1,start_server,or_minutes,rates);
   if(copied!=or_minutes)
   {
      if(EnableLogs)
         PrintFormat("[%s] OR copy failed. start=%s minutes=%d copied=%d err=%d",
                     symbol,TimeToString(start_server,TIME_DATE|TIME_MINUTES),
                     or_minutes,copied,GetLastError());
      return false;
   }

   or_high=rates[0].high;
   or_low =rates[0].low;

   for(int i=1;i<copied;i++)
   {
      if(rates[i].high>or_high) or_high=rates[i].high;
      if(rates[i].low <or_low ) or_low =rates[i].low;
   }

   or_high=NormalizeDouble(or_high,(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS));
   or_low =NormalizeDouble(or_low ,(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS));

   if(or_high<=or_low)
      return false;

   return true;
}

//+------------------------------------------------------------------+
//| Utility: get last closed bar on a timeframe                       |
//+------------------------------------------------------------------+
bool GetLastClosedBar(const string symbol,
                      const ENUM_TIMEFRAMES tf,
                      MqlRates &bar)
{
   MqlRates rates[];
   ArraySetAsSeries(rates,true);

   ResetLastError();
   const int copied=CopyRates(symbol,tf,1,1,rates);
   if(copied!=1)
      return false;

   bar=rates[0];
   return true;
}

//+------------------------------------------------------------------+
//| Utility: ATR value (last closed ATR bar)                          |
//+------------------------------------------------------------------+
double GetATRValue()
{
   if(g_atr_handle==INVALID_HANDLE)
      return 0.0;

   double buf[];
   ArraySetAsSeries(buf,true);

   ResetLastError();
   const int copied=CopyBuffer(g_atr_handle,0,1,1,buf);
   if(copied!=1)
   {
      if(EnableLogs)
         PrintFormat("ATR CopyBuffer failed. copied=%d err=%d",copied,GetLastError());
      return 0.0;
   }

   return buf[0];
}

//+------------------------------------------------------------------+
//| Utility: validate stop constraints for a market order             |
//+------------------------------------------------------------------+
bool ValidateStopsForOrder(const BREAKOUT_DIR dir,
                           const MqlTick &tick,
                           const double sl,
                           const double tp)
{
   const string symbol=_Symbol;
   const double point=SymbolInfoDouble(symbol,SYMBOL_POINT);

   const int stops_level=(int)SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL);
   const int freeze_level=(int)SymbolInfoInteger(symbol,SYMBOL_TRADE_FREEZE_LEVEL);

   const double min_dist = (stops_level>0 ? stops_level*point : 0.0);
   const double frz_dist = (freeze_level>0 ? freeze_level*point : 0.0);

   if(dir==DIR_LONG)
   {
      // For buy positions, stop constraints are checked from Bid (opposite price type)
      const double ref=tick.bid;
      if(sl>=ref || tp<=ref)
         return false;

      const double dsl = ref - sl;
      const double dtp = tp - ref;

      if(min_dist>0.0 && (dsl<min_dist || dtp<min_dist))
         return false;

      if(frz_dist>0.0 && (dsl<frz_dist || dtp<frz_dist))
         return false;

      return true;
   }
   else if(dir==DIR_SHORT)
   {
      // For sell positions, stop constraints are checked from Ask (opposite price type)
      const double ref=tick.ask;
      if(sl<=ref || tp>=ref)
         return false;

      const double dsl = sl - ref;
      const double dtp = ref - tp;

      if(min_dist>0.0 && (dsl<min_dist || dtp<min_dist))
         return false;

      if(frz_dist>0.0 && (dsl<frz_dist || dtp<frz_dist))
         return false;

      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Utility: validate stop constraints for SL modification (trailing) |
//+------------------------------------------------------------------+
bool ValidateStopsForModifySL(const ENUM_POSITION_TYPE pos_type,
                             const MqlTick &tick,
                             const double new_sl)
{
   const string symbol=_Symbol;
   const double point=SymbolInfoDouble(symbol,SYMBOL_POINT);

   const int stops_level=(int)SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL);
   const int freeze_level=(int)SymbolInfoInteger(symbol,SYMBOL_TRADE_FREEZE_LEVEL);

   const double min_dist = (stops_level>0 ? stops_level*point : 0.0);
   const double frz_dist = (freeze_level>0 ? freeze_level*point : 0.0);

   if(pos_type==POSITION_TYPE_BUY)
   {
      const double ref=tick.bid;
      const double dsl = ref - new_sl;
      if(dsl<=0.0)
         return false;

      if(min_dist>0.0 && dsl<min_dist)
         return false;

      if(frz_dist>0.0 && dsl<frz_dist)
         return false;

      return true;
   }
   else if(pos_type==POSITION_TYPE_SELL)
   {
      const double ref=tick.ask;
      const double dsl = new_sl - ref;
      if(dsl<=0.0)
         return false;

      if(min_dist>0.0 && dsl<min_dist)
         return false;

      if(frz_dist>0.0 && dsl<frz_dist)
         return false;

      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Utility: retcode success check                                   |
//+------------------------------------------------------------------+
bool IsTradeRetcodeSuccess(const uint retcode)
{
   if(retcode==TRADE_RETCODE_DONE)
      return true;
   if(retcode==TRADE_RETCODE_DONE_PARTIAL)
      return true;
   if(retcode==TRADE_RETCODE_PLACED)
      return true;

   return false;
}

//+------------------------------------------------------------------+
//| UI: create / update trailing toggle button                        |
//+------------------------------------------------------------------+
void UpdateTrailingButton()
{
   if(ObjectFind(0,g_btn_trail_name)<0)
      return;

   string txt = (g_trailing_enabled ? "Trailing: ON" : "Trailing: OFF");

   ObjectSetString(0,g_btn_trail_name,OBJPROP_TEXT,txt);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_COLOR,(g_trailing_enabled ? clrLime : clrRed));
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_BGCOLOR,clrBlack);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_BORDER_COLOR,clrGray);
   ChartRedraw(0);
}

void CreateTrailingButton()
{
   // Remove any stale object
   if(ObjectFind(0,g_btn_trail_name)>=0)
      ObjectDelete(0,g_btn_trail_name);

   if(!ObjectCreate(0,g_btn_trail_name,OBJ_BUTTON,0,0,0))
   {
      PrintFormat("Failed to create button. err=%d",GetLastError());
      return;
   }

   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_XDISTANCE,10);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_YDISTANCE,20);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_XSIZE,110);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_YSIZE,20);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_FONTSIZE,10);
   ObjectSetString(0,g_btn_trail_name,OBJPROP_FONT,"Arial");
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,g_btn_trail_name,OBJPROP_HIDDEN,true);

   UpdateTrailingButton();
}

//+------------------------------------------------------------------+
//| Session helpers                                                  |
//+------------------------------------------------------------------+
string SessionStateToString(const SESSION_STATE st)
{
   switch(st)
   {
      case BUILD_OR:      return "BUILD_OR";
      case WAIT_BREAKOUT: return "WAIT_BREAKOUT";
      case WAIT_RETEST:   return "WAIT_RETEST";
      case WAIT_CONFIRM:  return "WAIT_CONFIRM";
      case TRADED:        return "TRADED";
      case RESET:         return "RESET";
   }
   return "UNKNOWN";
}

string DirToString(const BREAKOUT_DIR d)
{
   switch(d)
   {
      case DIR_NONE:  return "NONE";
      case DIR_LONG:  return "LONG";
      case DIR_SHORT: return "SHORT";
   }
   return "UNKNOWN";
}

void ResetSession(SessionContext &s)
{
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

//+------------------------------------------------------------------+
//| Convert session UTC times to server times (for current UTC day)   |
//+------------------------------------------------------------------+
void RefreshSessionTimesForToday()
{
   datetime utc_now=TimeGMT();
   if(utc_now<=0)
      return;

   g_last_utc_day_start = DayStartUTC(utc_now);

   for(int i=0;i<3;i++)
   {
      SessionContext &s=g_sessions[i];

      s.start_utc = g_last_utc_day_start + s.utc_hour*3600 + s.utc_minute*60;
      s.start_server = s.start_utc + g_server_utc_offset_sec;

      if(s.enabled)
      {
         ResetSession(s);

         if(EnableLogs)
            PrintFormat("[%s] New UTC day. start_utc=%s start_server=%s offset=%d sec",
                        s.name,
                        TimeToString(s.start_utc,TIME_DATE|TIME_MINUTES),
                        TimeToString(s.start_server,TIME_DATE|TIME_MINUTES),
                        g_server_utc_offset_sec);
      }
      else
      {
         s.state=RESET;
      }
   }
}

//+------------------------------------------------------------------+
//| Update server UTC offset and adjust *future* sessions if needed   |
//+------------------------------------------------------------------+
void RefreshServerOffsetIfNeeded()
{
   const int cur_offset=GetServerUtcOffsetSeconds();
   if(cur_offset==0)
      return;

   if(MathAbs(cur_offset-g_server_utc_offset_sec)<60)
      return; // ignore small drifts

   // Offset changed (likely DST). Update and re-map only sessions that haven't started yet in UTC.
   g_server_utc_offset_sec=cur_offset;

   const datetime utc_now=TimeGMT();
   for(int i=0;i<3;i++)
   {
      SessionContext &s=g_sessions[i];
      if(!s.enabled)
         continue;

      if(utc_now < s.start_utc)
      {
         s.start_server = s.start_utc + g_server_utc_offset_sec;

         if(EnableLogs)
            PrintFormat("[%s] Offset changed -> updated future session start_server=%s (offset=%d sec)",
                        s.name,TimeToString(s.start_server,TIME_DATE|TIME_MINUTES),
                        g_server_utc_offset_sec);
      }
   }
}

//+------------------------------------------------------------------+
//| TRY_ENTER                                                        |
//+------------------------------------------------------------------+
bool TryEnter(SessionContext &s,const BREAKOUT_DIR dir)
{
   if(s.trade_taken)
      return false;

   if(!IsTradingAllowed())
   {
      if(EnableLogs)
         Print("Trading not allowed (terminal/account/MQL settings).");
      return false;
   }

   // Netting safety: if any position already exists on this symbol, skip to avoid unwanted netting merges
   if(!IsHedgingAccount() && AnyOpenPositionOnSymbol(_Symbol))
   {
      if(EnableLogs)
         PrintFormat("[%s] Netting account: position already exists on %s -> skip new entry",
                     s.name,_Symbol);
      return false;
   }

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol,tick))
      return false;

   const double pip=PipSize(_Symbol);
   const double buffer = SL_Buffer_Pips * pip;

   // Entry at market price at the moment confirmation is detected
   double entry = (dir==DIR_LONG ? tick.ask : tick.bid);

   // ATR-based SL
   const double atr = GetATRValue();
   if(atr<=0.0 || !MathIsValidNumber(atr))
   {
      if(EnableLogs)
         PrintFormat("[%s] ATR invalid (atr=%f). Skip.",s.name,atr);
      return false;
   }

   double sl=0.0;
   if(dir==DIR_LONG)
      sl = entry - ATR_Mult*atr - buffer;
   else if(dir==DIR_SHORT)
      sl = entry + ATR_Mult*atr + buffer;
   else
      return false;

   sl = NormalizeDouble(sl,_Digits);

   // Risk distance
   const double R = MathAbs(entry - sl);
   if(R<=0.0)
      return false;

   double tp=0.0;
   if(dir==DIR_LONG)
      tp = entry + RR*R;
   else
      tp = entry - RR*R;

   tp = NormalizeDouble(tp,_Digits);

   // Basic sanity
   if(dir==DIR_LONG)
   {
      if(!(sl<entry && tp>entry))
         return false;
   }
   else
   {
      if(!(sl>entry && tp<entry))
         return false;
   }

   // Broker stop constraints (stops / freeze / min distance)
   if(!ValidateStopsForOrder(dir,tick,sl,tp))
   {
      if(EnableLogs)
         PrintFormat("[%s] Stops/freeze constraint violated. entry=%.*f sl=%.*f tp=%.*f (bid=%.*f ask=%.*f)",
                     s.name,_Digits,entry,_Digits,sl,_Digits,tp,_Digits,tick.bid,_Digits,tick.ask);
      return false;
   }

   // Position sizing: lots = RiskUSD / risk_per_1_lot(stop_distance)
   const ENUM_ORDER_TYPE order_type = (dir==DIR_LONG ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);

   double profit_1lot=0.0;
   ResetLastError();
   if(!OrderCalcProfit(order_type,_Symbol,1.0,entry,sl,profit_1lot))
   {
      if(EnableLogs)
         PrintFormat("[%s] OrderCalcProfit failed. err=%d",s.name,GetLastError());
      return false;
   }

   const double risk_per_1lot = MathAbs(profit_1lot);
   if(risk_per_1lot<=0.0)
      return false;

   double lots = RiskUSD / risk_per_1lot;
   lots = NormalizeVolume(_Symbol,lots);
   if(lots<=0.0)
   {
      if(EnableLogs)
         PrintFormat("[%s] Volume too small after normalization. calcLots=%f",s.name,RiskUSD/risk_per_1lot);
      return false;
   }

   // Optional margin check
   double margin=0.0;
   ResetLastError();
   if(OrderCalcMargin(order_type,_Symbol,lots,entry,margin))
   {
      const double free_margin=AccountInfoDouble(ACCOUNT_FREEMARGIN);
      if(margin>free_margin)
      {
         if(EnableLogs)
            PrintFormat("[%s] Not enough free margin. need=%f free=%f",s.name,margin,free_margin);
         return false;
      }
   }

   // Place market order
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(DeviationPoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_trade.SetMarginMode();

   const string comment=StringFormat("OR_%s",s.name);

   bool sent=false;
   if(dir==DIR_LONG)
      sent = g_trade.Buy(lots,_Symbol,entry,sl,tp,comment);
   else
      sent = g_trade.Sell(lots,_Symbol,entry,sl,tp,comment);

   if(!sent)
   {
      if(EnableLogs)
         PrintFormat("[%s] Order send failed. retcode=%u (%s)",
                     s.name,
                     g_trade.ResultRetcode(),
                     g_trade.ResultRetcodeDescription());

      // per spec: avoid repeated sends
      s.state=RESET;
      return false;
   }

   const uint rc=g_trade.ResultRetcode();
   if(!IsTradeRetcodeSuccess(rc))
   {
      if(EnableLogs)
         PrintFormat("[%s] Trade request not successful. retcode=%u (%s)",
                     s.name,rc,g_trade.ResultRetcodeDescription());

      s.state=RESET;
      return false;
   }

   // Success
   s.trade_taken=true;
   s.state=TRADED;

   if(EnableLogs)
      PrintFormat("[%s] TRADE PLACED %s lots=%.*f entry~%.*f sl=%.*f tp=%.*f",
                  s.name,
                  DirToString(dir),
                  g_volume_digits,lots,
                  _Digits,entry,_Digits,sl,_Digits,tp);

   return true;
}

//+------------------------------------------------------------------+
//| Per-session state machine processing on each new closed bar       |
//+------------------------------------------------------------------+
void ProcessSessionOnClosedBar(SessionContext &s,
                               const MqlRates &bar,
                               const datetime bar_close_time,
                               const datetime now_server)
{
   if(!s.enabled)
      return;

   // If session already done for the day
   if(s.state==TRADED || s.state==RESET)
      return;

   const datetime session_start = s.start_server;
   if(session_start<=0)
      return;

   // If OR not yet built, we can still enforce a hard deadline based on expected window end
   const datetime expected_or_close = session_start + OR_Minutes*60;
   const datetime expected_window_end = expected_or_close + MaxMinutesAfterORClose*60;

   // BUILD_OR
   if(s.state==BUILD_OR)
   {
      if(now_server > expected_window_end && !s.trade_taken)
      {
         s.state=RESET;
         if(EnableLogs)
            PrintFormat("[%s] Deadline passed before OR built -> RESET",s.name);
         return;
      }

      if(now_server < expected_or_close)
         return;

      double orh,orl;
      if(!ComputeOpeningRange(_Symbol,session_start,OR_Minutes,orh,orl))
         return;

      s.ORH=orh;
      s.ORL=orl;

      s.OR_close_time   = expected_or_close;
      s.window_end_time = expected_window_end;

      s.state=WAIT_BREAKOUT;

      if(EnableLogs)
         PrintFormat("[%s] OR built. ORH=%.*f ORL=%.*f OR_close=%s window_end=%s",
                     s.name,_Digits,s.ORH,_Digits,s.ORL,
                     TimeToString(s.OR_close_time,TIME_DATE|TIME_MINUTES),
                     TimeToString(s.window_end_time,TIME_DATE|TIME_MINUTES));

      // Continue processing this closed bar as WAIT_BREAKOUT (important for TF misalignment).
   }

   // Deadline rule (after OR exists)
   if(!s.trade_taken && s.window_end_time>0 && now_server > s.window_end_time)
   {
      s.state=RESET;
      if(EnableLogs)
         PrintFormat("[%s] Entry window expired -> RESET",s.name);
      return;
   }

   // Ignore bars that close on/before OR close time (breakouts start after OR window)
   if(s.OR_close_time>0 && bar_close_time<=s.OR_close_time)
      return;

   // WAIT_BREAKOUT
   if(s.state==WAIT_BREAKOUT)
   {
      if(bar.close > s.ORH)
      {
         s.breakout_dir=DIR_LONG;
         s.breakout_bar_time=bar.time;
         s.retest_seen=false;
         s.state=WAIT_RETEST;

         if(EnableLogs)
            PrintFormat("[%s] Breakout LONG detected. breakout_bar=%s close=%.*f",
                        s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),_Digits,bar.close);
      }
      else if(bar.close < s.ORL)
      {
         s.breakout_dir=DIR_SHORT;
         s.breakout_bar_time=bar.time;
         s.retest_seen=false;
         s.state=WAIT_RETEST;

         if(EnableLogs)
            PrintFormat("[%s] Breakout SHORT detected. breakout_bar=%s close=%.*f",
                        s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),_Digits,bar.close);
      }
      return;
   }

   // WAIT_RETEST / WAIT_CONFIRM
   if(s.state==WAIT_RETEST || s.state==WAIT_CONFIRM)
   {
      // Opposite direction allowed (before trade)
      if(!s.trade_taken)
      {
         if(bar.close > s.ORH && s.breakout_dir!=DIR_LONG)
         {
            s.breakout_dir=DIR_LONG;
            s.breakout_bar_time=bar.time;
            s.retest_seen=false;
            s.state=WAIT_RETEST;

            if(EnableLogs)
               PrintFormat("[%s] Switch breakout -> LONG. breakout_bar=%s",
                           s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES));
            return;
         }

         if(bar.close < s.ORL && s.breakout_dir!=DIR_SHORT)
         {
            s.breakout_dir=DIR_SHORT;
            s.breakout_bar_time=bar.time;
            s.retest_seen=false;
            s.state=WAIT_RETEST;

            if(EnableLogs)
               PrintFormat("[%s] Switch breakout -> SHORT. breakout_bar=%s",
                           s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES));
            return;
         }
      }

      // Retest must be on a later candle than breakout candle
      if(bar.time<=s.breakout_bar_time)
         return;

      // Retest not yet seen
      if(!s.retest_seen)
      {
         if(s.breakout_dir==DIR_LONG)
         {
            if(bar.low <= s.ORH)
            {
               s.retest_seen=true;
               s.retest_start_time=bar.time;
               s.state=WAIT_CONFIRM;

               if(EnableLogs)
                  PrintFormat("[%s] Retest seen (LONG). retest_bar=%s low=%.*f",
                              s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),_Digits,bar.low);

               // If same candle also confirms
               if(bar.close > s.ORH)
                  TryEnter(s,DIR_LONG);
            }
         }
         else if(s.breakout_dir==DIR_SHORT)
         {
            if(bar.high >= s.ORL)
            {
               s.retest_seen=true;
               s.retest_start_time=bar.time;
               s.state=WAIT_CONFIRM;

               if(EnableLogs)
                  PrintFormat("[%s] Retest seen (SHORT). retest_bar=%s high=%.*f",
                              s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),_Digits,bar.high);

               // If same candle also confirms
               if(bar.close < s.ORL)
                  TryEnter(s,DIR_SHORT);
            }
         }

         return;
      }

      // Retest already seen -> wait confirm
      if(s.retest_seen)
      {
         if(s.breakout_dir==DIR_LONG)
         {
            if(bar.close > s.ORH)
               TryEnter(s,DIR_LONG);
         }
         else if(s.breakout_dir==DIR_SHORT)
         {
            if(bar.close < s.ORL)
               TryEnter(s,DIR_SHORT);
         }
      }

      return;
   }
}

//+------------------------------------------------------------------+
//| Trailing management                                              |
//+------------------------------------------------------------------+
void ManageTrailing()
{
   if(!g_trailing_enabled)
      return;

   if(Trail_TriggerProfitPips<=0.0 || Trail_DistancePips<=0.0)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol,tick))
      return;

   const double pip=PipSize(_Symbol);

   const int total=PositionsTotal();
   for(int i=total-1;i>=0;i--)
   {
      const ulong ticket=PositionGetTicket(i);
      if(ticket==0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      const string sym=PositionGetString(POSITION_SYMBOL);
      if(sym!=_Symbol)
         continue;

      const long magic=PositionGetInteger(POSITION_MAGIC);
      if((ulong)magic!=MagicNumber)
         continue;

      const ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      const double entry=PositionGetDouble(POSITION_PRICE_OPEN);
      const double curSL=PositionGetDouble(POSITION_SL);
      const double curTP=PositionGetDouble(POSITION_TP);

      double profit_pips=0.0;
      if(type==POSITION_TYPE_BUY)
         profit_pips=(tick.bid-entry)/pip;
      else if(type==POSITION_TYPE_SELL)
         profit_pips=(entry-tick.ask)/pip;
      else
         continue;

      if(profit_pips<Trail_TriggerProfitPips)
         continue;

      if(type==POSITION_TYPE_BUY)
      {
         const double lockSL = entry + Trail_LockInPips*pip;
         const double trailSL= tick.bid - Trail_DistancePips*pip;

         double baseSL = (curSL>0.0 ? curSL : -DBL_MAX);
         double newSL = MathMax(baseSL,MathMax(lockSL,trailSL));
         newSL=NormalizeDouble(newSL,_Digits);

         // Must improve SL
         if(curSL>0.0 && newSL<=curSL)
            continue;

         if(!ValidateStopsForModifySL(type,tick,newSL))
            continue;

         g_trade.SetExpertMagicNumber(MagicNumber);
         g_trade.SetDeviationInPoints(DeviationPoints);
         g_trade.SetTypeFillingBySymbol(_Symbol);
         g_trade.SetMarginMode();

         if(g_trade.PositionModify(ticket,newSL,curTP))
         {
            const uint rc=g_trade.ResultRetcode();
            if(IsTradeRetcodeSuccess(rc) || rc==TRADE_RETCODE_NO_CHANGES)
            {
               if(EnableLogs)
                  PrintFormat("Trailing BUY modified. ticket=%I64u newSL=%.*f",ticket,_Digits,newSL);
            }
         }
      }
      else if(type==POSITION_TYPE_SELL)
      {
         const double lockSL = entry - Trail_LockInPips*pip;
         const double trailSL= tick.ask + Trail_DistancePips*pip;

         double baseSL = (curSL>0.0 ? curSL : DBL_MAX);
         double newSL = MathMin(baseSL,MathMin(lockSL,trailSL));
         newSL=NormalizeDouble(newSL,_Digits);

         // Must improve SL (downwards)
         if(curSL>0.0 && newSL>=curSL)
            continue;

         if(!ValidateStopsForModifySL(type,tick,newSL))
            continue;

         g_trade.SetExpertMagicNumber(MagicNumber);
         g_trade.SetDeviationInPoints(DeviationPoints);
         g_trade.SetTypeFillingBySymbol(_Symbol);
         g_trade.SetMarginMode();

         if(g_trade.PositionModify(ticket,newSL,curTP))
         {
            const uint rc=g_trade.ResultRetcode();
            if(IsTradeRetcodeSuccess(rc) || rc==TRADE_RETCODE_NO_CHANGES)
            {
               if(EnableLogs)
                  PrintFormat("Trailing SELL modified. ticket=%I64u newSL=%.*f",ticket,_Digits,newSL);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Core loop                                                        |
//+------------------------------------------------------------------+
void MainLoop()
{
   // Detect new UTC day
   const datetime utc_now=TimeGMT();
   if(utc_now>0)
   {
      const datetime day_start=DayStartUTC(utc_now);
      if(g_last_utc_day_start==0 || day_start!=g_last_utc_day_start)
      {
         g_last_utc_day_start=day_start;

         g_server_utc_offset_sec=GetServerUtcOffsetSeconds();
         RefreshSessionTimesForToday();
      }
   }

   // Handle mid-day offset changes (DST) for future sessions
   RefreshServerOffsetIfNeeded();

   // Process new closed Confirm_TF bar
   datetime closed_time=iTime(_Symbol,Confirm_TF,1);
   if(closed_time>0 && closed_time!=g_last_confirm_closed_time)
   {
      g_last_confirm_closed_time=closed_time;

      MqlRates bar;
      if(GetLastClosedBar(_Symbol,Confirm_TF,bar))
      {
         const datetime bar_close_time = bar.time + PeriodSeconds(Confirm_TF);
         const datetime now_server = ServerTime();

         for(int i=0;i<3;i++)
            ProcessSessionOnClosedBar(g_sessions[i],bar,bar_close_time,now_server);
      }
   }

   // Trailing (tick/timer driven)
   ManageTrailing();
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Validate inputs
   if(!(OR_Minutes==5 || OR_Minutes==15 || OR_Minutes==30))
   {
      Print("Invalid OR_Minutes. Allowed values: 5, 15, 30.");
      return INIT_FAILED;
   }

   if(!(Confirm_TF==PERIOD_M1 || Confirm_TF==PERIOD_M3 || Confirm_TF==PERIOD_M5))
   {
      Print("Invalid Confirm_TF. Allowed values: M1, M3, M5.");
      return INIT_FAILED;
   }

   if(ATR_Period<=0 || ATR_Mult<=0.0)
   {
      Print("Invalid ATR inputs.");
      return INIT_FAILED;
   }

   g_volume_digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_VOLUME_DIGITS);

   g_trailing_enabled=TrailingEnabledInitial;

   // Setup sessions
   g_sessions[0].name="MIDNIGHT";
   g_sessions[0].enabled=Trade_Midnight;
   g_sessions[0].utc_hour=Midnight_UTC_Hour;
   g_sessions[0].utc_minute=Midnight_UTC_Minute;

   g_sessions[1].name="LONDON";
   g_sessions[1].enabled=Trade_London;
   g_sessions[1].utc_hour=London_UTC_Hour;
   g_sessions[1].utc_minute=London_UTC_Minute;

   g_sessions[2].name="NY";
   g_sessions[2].enabled=Trade_NY;
   g_sessions[2].utc_hour=NY_UTC_Hour;
   g_sessions[2].utc_minute=NY_UTC_Minute;

   // Setup ATR handle
   g_atr_handle=iATR(_Symbol,ATR_TF,ATR_Period);
   if(g_atr_handle==INVALID_HANDLE)
   {
      PrintFormat("Failed to create ATR handle. err=%d",GetLastError());
      return INIT_FAILED;
   }

   // Setup trade object defaults
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(DeviationPoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_trade.SetMarginMode();

   // Initialize time mapping + session states
   g_server_utc_offset_sec=GetServerUtcOffsetSeconds();
   RefreshSessionTimesForToday();

   // UI
   CreateTrailingButton();

   // Timer to keep trailing logic responsive even on low-tick symbols
   EventSetTimer(1);

   if(EnableLogs)
   {
      PrintFormat("Initialized OR_Retest_ATR on %s | Confirm_TF=%s OR_Minutes=%d | ATR(%s,%d)*%.2f | Magic=%I64u",
                  _Symbol,
                  EnumToString(Confirm_TF),
                  OR_Minutes,
                  EnumToString(ATR_TF),
                  ATR_Period,
                  ATR_Mult,
                  MagicNumber);
   }

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();

   if(g_atr_handle!=INVALID_HANDLE)
   {
      IndicatorRelease(g_atr_handle);
      g_atr_handle=INVALID_HANDLE;
   }

   ObjectDelete(0,g_btn_trail_name);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   MainLoop();
}

//+------------------------------------------------------------------+
//| Timer event                                                     |
//+------------------------------------------------------------------+
void OnTimer()
{
   MainLoop();
}

//+------------------------------------------------------------------+
//| Chart event (button toggle)                                      |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id==CHARTEVENT_OBJECT_CLICK && sparam==g_btn_trail_name)
   {
      g_trailing_enabled = !g_trailing_enabled;

      if(EnableLogs)
         PrintFormat("Trailing toggled -> %s", (g_trailing_enabled ? "ON" : "OFF"));

      UpdateTrailingButton();
   }
}
//+------------------------------------------------------------------+