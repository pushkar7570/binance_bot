//+------------------------------------------------------------------+
//| OR_Retest_Confirm_EA.mq5                                          |
//| Opening Range (M1) breakout + retest + confirm (Confirm_TF)       |
//| 3 UTC sessions (Midnight/London/NY) converted to server time      |
//+------------------------------------------------------------------+
#property copyright "OpenAI (generated)"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- inputs: Sessions
input bool   InpTrade_Midnight = true;     // Trade Midnight session (00:00 UTC)
input bool   InpTrade_London   = true;     // Trade London session   (07:00 UTC)
input bool   InpTrade_NY       = true;     // Trade New York session (12:00 UTC)

//--- inputs: Opening Range + signals
input int             InpOR_Minutes              = 15;        // OR minutes (5,15,30)
input ENUM_TIMEFRAMES InpConfirm_TF              = PERIOD_M5;  // Confirm TF (M1,M3,M5)
input int             InpMaxMinutesAfterORClose  = 105;       // Max minutes after OR close

//--- inputs: Risk & trade params
input double InpRiskUSD        = 100.0;    // Risk per trade (account currency)
input double InpRR             = 2.0;      // Risk:Reward
input double InpSL_Buffer_Pips = 0.0;      // SL buffer in pips (see PipSize())

//--- inputs: Trailing
input bool   InpTrailingEnabledInit     = true;  // Trailing enabled at start (toggle via button)
input double InpTrail_TriggerProfitPips = 10.0;  // Start trailing after profit pips
input double InpTrail_LockInPips        = 4.0;   // Lock-in pips (from entry)
input double InpTrail_DistancePips      = 10.0;  // Trail distance pips (from price)

//--- inputs: Misc
input ulong InpMagicNumber      = 260217; // Magic number
input int   InpSlippagePoints   = 10;     // Slippage in points
input bool  InpPrintDebug       = true;   // Print debug logs

//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
enum ENUM_OR_STATE
{
   STATE_BUILD_OR = 0,
   STATE_WAIT_BREAKOUT,
   STATE_WAIT_RETEST,
   STATE_WAIT_CONFIRM,
   STATE_TRADED,
   STATE_RESET
};

enum ENUM_BREAKOUT_DIR
{
   DIR_NONE = 0,
   DIR_LONG,
   DIR_SHORT
};

//+------------------------------------------------------------------+
//| Session data                                                     |
//+------------------------------------------------------------------+
struct SessionData
{
   string            name;
   bool              enabled;
   int               utc_hour;
   int               utc_min;

   datetime          start_utc;
   datetime          start_server;

   ENUM_OR_STATE     state;

   double            ORH;
   double            ORL;
   datetime          OR_close_time;
   datetime          window_end_time;

   ENUM_BREAKOUT_DIR breakout_dir;
   datetime          breakout_bar_time;     // "closed bar time" (we store bar open time for uniqueness)
   double            breakout_candle_high;  // breakout candle high (Confirm_TF closed bar)
   double            breakout_candle_low;   // breakout candle low  (Confirm_TF closed bar)

   bool              retest_seen;
   datetime          retest_start_time;

   bool              trade_taken;
};

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade      g_trade;

SessionData g_sessions[3];

datetime    g_last_confirm_closed_bar_time = 0;
datetime    g_last_utc_day_start           = 0;
int         g_server_utc_offset_sec        = 0;

bool        g_trailing_enabled             = false;

string      g_btn_trailing_name            = "OR_EA_TrailingToggle";

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
string StateToString(const ENUM_OR_STATE st)
{
   switch(st)
   {
      case STATE_BUILD_OR:      return "BUILD_OR";
      case STATE_WAIT_BREAKOUT: return "WAIT_BREAKOUT";
      case STATE_WAIT_RETEST:   return "WAIT_RETEST";
      case STATE_WAIT_CONFIRM:  return "WAIT_CONFIRM";
      case STATE_TRADED:        return "TRADED";
      case STATE_RESET:         return "RESET";
      default:                  return "UNKNOWN";
   }
}

string DirToString(const ENUM_BREAKOUT_DIR dir)
{
   switch(dir)
   {
      case DIR_LONG:  return "LONG";
      case DIR_SHORT: return "SHORT";
      default:        return "NONE";
   }
}

double PipSize(const string symbol)
{
   const int digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   const double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(digits==3 || digits==5)
      return(point*10.0);
   return(point);
}

double NormalizeVolume(const string symbol,const double volume)
{
   const double vol_min  = SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN);
   const double vol_max  = SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX);
   const double vol_step = SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP);

   if(vol_min<=0.0 || vol_max<=0.0 || vol_step<=0.0)
      return(0.0);

   double v=volume;

   if(v<vol_min)
      return(0.0);
   if(v>vol_max)
      v=vol_max;

   // round down to the nearest step starting from vol_min
   double steps=MathFloor((v-vol_min)/vol_step);
   v=vol_min + steps*vol_step;

   if(v<vol_min)
      return(0.0);

   const int vol_digits=(int)SymbolInfoInteger(symbol,SYMBOL_VOLUME_DIGITS);
   return(NormalizeDouble(v,vol_digits));
}

bool IsConfirmTFAllowed(const ENUM_TIMEFRAMES tf)
{
   return(tf==PERIOD_M1 || tf==PERIOD_M3 || tf==PERIOD_M5);
}

bool IsORMinutesAllowed(const int minutes)
{
   return(minutes==5 || minutes==15 || minutes==30);
}

//+------------------------------------------------------------------+
//| Chart button                                                     |
//+------------------------------------------------------------------+
void UpdateTrailingButton()
{
   if(ObjectFind(0,g_btn_trailing_name)<0)
      return;

   string text = g_trailing_enabled ? "Trailing: ON" : "Trailing: OFF";
   ObjectSetString(0,g_btn_trailing_name,OBJPROP_TEXT,text);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_STATE,g_trailing_enabled);
   ChartRedraw();
}

bool CreateTrailingButton()
{
   if(ObjectFind(0,g_btn_trailing_name)>=0)
      ObjectDelete(0,g_btn_trailing_name);

   if(!ObjectCreate(0,g_btn_trailing_name,OBJ_BUTTON,0,0,0))
   {
      Print(__FUNCTION__,": failed to create button. Error=",GetLastError());
      return(false);
   }

   // Position top-left
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_XDISTANCE,10);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_YDISTANCE,20);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_XSIZE,120);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_YSIZE,22);

   ObjectSetString(0,g_btn_trailing_name,OBJPROP_FONT,"Arial");
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_FONTSIZE,10);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_HIDDEN,true);
   ObjectSetInteger(0,g_btn_trailing_name,OBJPROP_ZORDER,0);

   UpdateTrailingButton();
   return(true);
}

//+------------------------------------------------------------------+
//| Session initialization                                            |
//+------------------------------------------------------------------+
void ResetSession(SessionData &s)
{
   s.ORH=0.0;
   s.ORL=0.0;
   s.OR_close_time=0;
   s.window_end_time=0;

   s.breakout_dir=DIR_NONE;
   s.breakout_bar_time=0;
   s.breakout_candle_high=0.0;
   s.breakout_candle_low=0.0;

   s.retest_seen=false;
   s.retest_start_time=0;

   s.trade_taken=false;

   s.state = s.enabled ? STATE_BUILD_OR : STATE_RESET;
}

void InitSessions()
{
   g_sessions[0].name="Midnight";
   g_sessions[0].utc_hour=0;
   g_sessions[0].utc_min=0;

   g_sessions[1].name="London";
   g_sessions[1].utc_hour=7;
   g_sessions[1].utc_min=0;

   g_sessions[2].name="NewYork";
   g_sessions[2].utc_hour=12;
   g_sessions[2].utc_min=0;
}

void ApplySessionToggles()
{
   g_sessions[0].enabled = InpTrade_Midnight;
   g_sessions[1].enabled = InpTrade_London;
   g_sessions[2].enabled = InpTrade_NY;
}

void ResetSessionsForNewUTCDay(const datetime utc_day_start)
{
   g_server_utc_offset_sec = (int)(TimeTradeServer() - TimeGMT());

   ApplySessionToggles();

   for(int i=0;i<3;i++)
   {
      SessionData &s = g_sessions[i];

      s.start_utc = utc_day_start + s.utc_hour*3600 + s.utc_min*60;
      s.start_server = s.start_utc + g_server_utc_offset_sec;

      ResetSession(s);

      if(InpPrintDebug)
      {
         PrintFormat("New UTC day: session=%s start_utc=%s start_server=%s offset_sec=%d",
                     s.name,
                     TimeToString(s.start_utc,TIME_DATE|TIME_MINUTES),
                     TimeToString(s.start_server,TIME_DATE|TIME_MINUTES),
                     g_server_utc_offset_sec);
      }
   }
}

void CheckForNewUTCDay()
{
   const datetime utc_now = TimeGMT();
   MqlDateTime dt;
   TimeToStruct(utc_now,dt);
   dt.hour=0;
   dt.min=0;
   dt.sec=0;
   const datetime utc_day_start = StructToTime(dt);

   if(utc_day_start != g_last_utc_day_start)
   {
      g_last_utc_day_start = utc_day_start;
      ResetSessionsForNewUTCDay(utc_day_start);
   }
}

//+------------------------------------------------------------------+
//| Opening Range computation                                         |
//+------------------------------------------------------------------+
bool ComputeOpeningRange(const datetime start_server,
                         const int or_minutes,
                         double &out_high,
                         double &out_low)
{
   const datetime from = start_server;
   const datetime to   = start_server + or_minutes*60 - 1;

   MqlRates rates[];
   ResetLastError();
   int copied = CopyRates(_Symbol,PERIOD_M1,from,to,rates);
   if(copied<=0)
   {
      if(InpPrintDebug)
         PrintFormat("ComputeOpeningRange: no M1 rates copied (%s -> %s). Error=%d",
                     TimeToString(from,TIME_DATE|TIME_MINUTES),
                     TimeToString(to,TIME_DATE|TIME_MINUTES),
                     GetLastError());
      return(false);
   }

   double h=rates[0].high;
   double l=rates[0].low;

   for(int i=1;i<copied;i++)
   {
      if(rates[i].high>h) h=rates[i].high;
      if(rates[i].low<l)  l=rates[i].low;
   }

   const int digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   out_high = NormalizeDouble(h,digits);
   out_low  = NormalizeDouble(l,digits);

   return(true);
}

//+------------------------------------------------------------------+
//| Risk / lot sizing                                                 |
//+------------------------------------------------------------------+
double RiskPerLot(const string symbol,const double stop_distance_price)
{
   if(stop_distance_price<=0.0)
      return(0.0);

   double tick_size = SymbolInfoDouble(symbol,SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol,SYMBOL_TRADE_TICK_VALUE_LOSS);
   if(tick_value<=0.0)
      tick_value = SymbolInfoDouble(symbol,SYMBOL_TRADE_TICK_VALUE);

   if(tick_size<=0.0 || tick_value<=0.0)
      return(0.0);

   double ticks = stop_distance_price / tick_size;
   return(ticks * tick_value);
}

double CalculateLotsByRisk(const string symbol,
                           const double risk_money,
                           const double stop_distance_price)
{
   const double risk_per_lot = RiskPerLot(symbol,stop_distance_price);
   if(risk_per_lot<=0.0)
      return(0.0);

   const double lots_raw = risk_money / risk_per_lot;
   return(NormalizeVolume(symbol,lots_raw));
}

//+------------------------------------------------------------------+
//| Stop/Freeze validation                                            |
//+------------------------------------------------------------------+
bool ValidateStopsForMarket(const ENUM_BREAKOUT_DIR dir,
                            const double entry,
                            const double sl,
                            const double tp)
{
   if(dir==DIR_LONG)
   {
      if(!(sl < entry && tp > entry))
         return(false);
   }
   else if(dir==DIR_SHORT)
   {
      if(!(sl > entry && tp < entry))
         return(false);
   }
   else
      return(false);

   const double point = SymbolInfoDouble(_Symbol,SYMBOL_POINT);

   const int stops_level  = (int)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   const int freeze_level = (int)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL);

   const double min_dist = (double)MathMax(stops_level,freeze_level) * point;

   // Use current price as reference (market order)
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol,tick))
      return(false);

   if(dir==DIR_LONG)
   {
      const double ref = tick.bid;
      if((ref - sl) < min_dist) return(false);
      if((tp - ref) < min_dist) return(false);
   }
   else
   {
      const double ref = tick.ask;
      if((sl - ref) < min_dist) return(false);
      if((ref - tp) < min_dist) return(false);
   }

   return(true);
}

//+------------------------------------------------------------------+
//| Check existing position for this session (restart safety)          |
//+------------------------------------------------------------------+
bool SessionHasOpenPosition(const SessionData &s)
{
   // We tag positions with comment "OR_<SessionName>"
   const string tag = "OR_" + s.name;

   // Build a conservative session end time even if OR not computed yet
   datetime end_time = s.window_end_time;
   if(end_time<=0)
      end_time = s.start_server + (InpOR_Minutes + InpMaxMinutesAfterORClose) * 60;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      if(!PositionSelectByIndex(i))
         continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      string cmt = PositionGetString(POSITION_COMMENT);
      if(StringFind(cmt,tag) < 0)
         continue;

      datetime opent = (datetime)PositionGetInteger(POSITION_TIME);
      if(opent >= s.start_server && opent <= end_time)
         return(true);
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Trade entry                                                       |
//+------------------------------------------------------------------+
bool TryEnter(SessionData &s,const ENUM_BREAKOUT_DIR dir)
{
   if(s.trade_taken)
      return(false);

   // Prevent duplicates after restart (same day/session)
   if(SessionHasOpenPosition(s))
   {
      s.trade_taken=true;
      s.state=STATE_TRADED;
      if(InpPrintDebug)
         PrintFormat("Session %s: existing position detected -> marking as TRADED.",s.name);
      return(false);
   }

   // Trading permissions
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      if(InpPrintDebug)
         Print("Trading is not allowed by terminal or EA settings.");
      s.state=STATE_RESET;
      return(false);
   }

   const long trade_mode = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_MODE);
   if(trade_mode==SYMBOL_TRADE_MODE_DISABLED)
   {
      if(InpPrintDebug)
         Print("Symbol trade mode is disabled.");
      s.state=STATE_RESET;
      return(false);
   }

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol,tick))
   {
      s.state=STATE_RESET;
      return(false);
   }

   const int digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   const double pip = PipSize(_Symbol);

   double entry = (dir==DIR_LONG) ? tick.ask : tick.bid;
   entry = NormalizeDouble(entry,digits);

   const double buffer = InpSL_Buffer_Pips * pip;

   // SL Method: Breakout Candle Extreme (Tighter)
   double sl=0.0;
   if(dir==DIR_LONG)
      sl = s.breakout_candle_low - buffer;
   else if(dir==DIR_SHORT)
      sl = s.breakout_candle_high + buffer;
   else
      return(false);

   sl = NormalizeDouble(sl,digits);

   const double stop_distance = MathAbs(entry - sl);
   if(stop_distance <= 0.0)
   {
      s.state=STATE_RESET;
      return(false);
   }

   // TP by RR
   double tp=0.0;
   const double R = stop_distance;
   if(dir==DIR_LONG)
      tp = entry + InpRR * R;
   else
      tp = entry - InpRR * R;

   tp = NormalizeDouble(tp,digits);

   // Broker constraints check
   if(!ValidateStopsForMarket(dir,entry,sl,tp))
   {
      if(InpPrintDebug)
         PrintFormat("Session %s: stops validation failed. entry=%.5f sl=%.5f tp=%.5f",
                     s.name,entry,sl,tp);
      s.state=STATE_RESET;
      return(false);
   }

   // Position sizing by RiskUSD
   double lots = CalculateLotsByRisk(_Symbol,InpRiskUSD,stop_distance);
   if(lots<=0.0)
   {
      if(InpPrintDebug)
         PrintFormat("Session %s: lots calculation failed. risk=%.2f stop=%.5f",
                     s.name,InpRiskUSD,stop_distance);
      s.state=STATE_RESET;
      return(false);
   }

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   const string comment = "OR_" + s.name;

   bool ok=false;
   if(dir==DIR_LONG)
      ok = g_trade.Buy(lots,_Symbol,entry,sl,tp,comment);
   else
      ok = g_trade.Sell(lots,_Symbol,entry,sl,tp,comment);

   if(ok && (g_trade.ResultRetcode()==TRADE_RETCODE_DONE || g_trade.ResultRetcode()==TRADE_RETCODE_PLACED))
   {
      s.trade_taken=true;
      s.state=STATE_TRADED;

      if(InpPrintDebug)
      {
         PrintFormat("TRADE OK: session=%s dir=%s lots=%.2f entry=%.5f sl=%.5f tp=%.5f (retcode=%u)",
                     s.name,DirToString(dir),lots,entry,sl,tp,g_trade.ResultRetcode());
      }
      return(true);
   }

   if(InpPrintDebug)
   {
      PrintFormat("TRADE FAIL: session=%s dir=%s retcode=%u (%s)",
                  s.name,DirToString(dir),g_trade.ResultRetcode(),g_trade.ResultRetcodeDescription());
   }

   s.state=STATE_RESET; // avoid repeated sends
   return(false);
}

//+------------------------------------------------------------------+
//| Trailing management                                               |
//+------------------------------------------------------------------+
void ManageTrailing()
{
   if(!g_trailing_enabled)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol,tick))
      return;

   const double pip = PipSize(_Symbol);
   const int digits = (int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   const double point = SymbolInfoDouble(_Symbol,SYMBOL_POINT);

   const int stops_level  = (int)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   const int freeze_level = (int)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL);
   const double min_dist = (double)MathMax(stops_level,freeze_level) * point;

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      if(!PositionSelectByIndex(i))
         continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      const ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
      const ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double cur_sl = PositionGetDouble(POSITION_SL);
      const double cur_tp = PositionGetDouble(POSITION_TP);

      if(type==POSITION_TYPE_BUY)
      {
         const double profit_pips = (tick.bid - entry) / pip;
         if(profit_pips < InpTrail_TriggerProfitPips)
            continue;

         const double lockSL  = entry + InpTrail_LockInPips * pip;
         const double trailSL = tick.bid - InpTrail_DistancePips * pip;

         double candidate = MathMax(lockSL,trailSL);
         if(cur_sl>0.0)
            candidate = MathMax(candidate,cur_sl);

         double new_sl = NormalizeDouble(candidate,digits);

         bool improves = (cur_sl<=0.0) ? true : (new_sl > cur_sl + point);
         bool allowed  = (tick.bid - new_sl) >= min_dist;

         if(improves && allowed)
         {
            if(!g_trade.PositionModify(ticket,new_sl,cur_tp))
            {
               if(InpPrintDebug)
                  PrintFormat("Trail BUY modify failed (ticket=%I64u). retcode=%u %s",
                              ticket,g_trade.ResultRetcode(),g_trade.ResultRetcodeDescription());
            }
         }
      }
      else if(type==POSITION_TYPE_SELL)
      {
         const double profit_pips = (entry - tick.ask) / pip;
         if(profit_pips < InpTrail_TriggerProfitPips)
            continue;

         const double lockSL  = entry - InpTrail_LockInPips * pip;
         const double trailSL = tick.ask + InpTrail_DistancePips * pip;

         double candidate = MathMin(lockSL,trailSL);
         if(cur_sl>0.0)
            candidate = MathMin(candidate,cur_sl);

         double new_sl = NormalizeDouble(candidate,digits);

         bool improves = (cur_sl<=0.0) ? true : (new_sl < cur_sl - point);
         bool allowed  = (new_sl - tick.ask) >= min_dist;

         if(improves && allowed)
         {
            if(!g_trade.PositionModify(ticket,new_sl,cur_tp))
            {
               if(InpPrintDebug)
                  PrintFormat("Trail SELL modify failed (ticket=%I64u). retcode=%u %s",
                              ticket,g_trade.ResultRetcode(),g_trade.ResultRetcodeDescription());
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Session state machine on each closed Confirm_TF bar                |
//+------------------------------------------------------------------+
void ProcessSessionOnClosedBar(SessionData &s,
                               const datetime now_server,
                               const MqlRates &bar)
{
   if(!s.enabled)
   {
      s.state=STATE_RESET;
      return;
   }

   // Restart safety: if a position exists for this session/day, mark traded
   if(!s.trade_taken && SessionHasOpenPosition(s))
   {
      s.trade_taken=true;
      s.state=STATE_TRADED;
      return;
   }

   // Deadline rule
   if(s.OR_close_time>0 && !s.trade_taken && now_server > s.window_end_time)
   {
      s.state=STATE_RESET;
      return;
   }

   switch(s.state)
   {
      case STATE_BUILD_OR:
      {
         if(now_server >= (s.start_server + InpOR_Minutes*60))
         {
            double orh=0.0,orl=0.0;
            if(ComputeOpeningRange(s.start_server,InpOR_Minutes,orh,orl))
            {
               s.ORH=orh;
               s.ORL=orl;
               s.OR_close_time = s.start_server + InpOR_Minutes*60;
               s.window_end_time = s.OR_close_time + InpMaxMinutesAfterORClose*60;
               s.state=STATE_WAIT_BREAKOUT;

               if(InpPrintDebug)
               {
                  PrintFormat("Session %s OR built: ORH=%.5f ORL=%.5f OR_close=%s window_end=%s",
                              s.name,s.ORH,s.ORL,
                              TimeToString(s.OR_close_time,TIME_DATE|TIME_MINUTES),
                              TimeToString(s.window_end_time,TIME_DATE|TIME_MINUTES));
               }
            }
         }
         break;
      }

      case STATE_WAIT_BREAKOUT:
      {
         if(bar.close > s.ORH)
         {
            s.breakout_dir = DIR_LONG;
            s.breakout_bar_time = bar.time;
            s.breakout_candle_low = bar.low;
            s.breakout_candle_high = bar.high;
            s.retest_seen=false;
            s.state=STATE_WAIT_RETEST;

            if(InpPrintDebug)
               PrintFormat("Session %s breakout LONG at %s (close=%.5f ORH=%.5f)",
                           s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),bar.close,s.ORH);
         }
         else if(bar.close < s.ORL)
         {
            s.breakout_dir = DIR_SHORT;
            s.breakout_bar_time = bar.time;
            s.breakout_candle_low = bar.low;
            s.breakout_candle_high = bar.high;
            s.retest_seen=false;
            s.state=STATE_WAIT_RETEST;

            if(InpPrintDebug)
               PrintFormat("Session %s breakout SHORT at %s (close=%.5f ORL=%.5f)",
                           s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),bar.close,s.ORL);
         }
         break;
      }

      case STATE_WAIT_RETEST:
      case STATE_WAIT_CONFIRM:
      {
         // Opposite direction allowed (before trade)
         if(!s.trade_taken)
         {
            if(bar.close > s.ORH && s.breakout_dir != DIR_LONG)
            {
               s.breakout_dir = DIR_LONG;
               s.breakout_bar_time = bar.time;
               s.breakout_candle_low = bar.low;
               s.breakout_candle_high = bar.high;
               s.retest_seen=false;
               s.state=STATE_WAIT_RETEST;

               if(InpPrintDebug)
                  PrintFormat("Session %s switched breakout -> LONG at %s",s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES));
            }
            else if(bar.close < s.ORL && s.breakout_dir != DIR_SHORT)
            {
               s.breakout_dir = DIR_SHORT;
               s.breakout_bar_time = bar.time;
               s.breakout_candle_low = bar.low;
               s.breakout_candle_high = bar.high;
               s.retest_seen=false;
               s.state=STATE_WAIT_RETEST;

               if(InpPrintDebug)
                  PrintFormat("Session %s switched breakout -> SHORT at %s",s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES));
            }
         }

         // Retest must be on a later candle than breakout candle
         if(bar.time <= s.breakout_bar_time)
            break;

         if(!s.retest_seen)
         {
            if(s.breakout_dir == DIR_LONG)
            {
               if(bar.low <= s.ORH)
               {
                  s.retest_seen=true;
                  s.retest_start_time=bar.time;
                  s.state=STATE_WAIT_CONFIRM;

                  if(InpPrintDebug)
                     PrintFormat("Session %s retest LONG at %s (low=%.5f ORH=%.5f)",
                                 s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),bar.low,s.ORH);

                  if(bar.close > s.ORH)
                     TryEnter(s,DIR_LONG);
               }
            }
            else if(s.breakout_dir == DIR_SHORT)
            {
               if(bar.high >= s.ORL)
               {
                  s.retest_seen=true;
                  s.retest_start_time=bar.time;
                  s.state=STATE_WAIT_CONFIRM;

                  if(InpPrintDebug)
                     PrintFormat("Session %s retest SHORT at %s (high=%.5f ORL=%.5f)",
                                 s.name,TimeToString(bar.time,TIME_DATE|TIME_MINUTES),bar.high,s.ORL);

                  if(bar.close < s.ORL)
                     TryEnter(s,DIR_SHORT);
               }
            }
         }
         else
         {
            if(s.breakout_dir == DIR_LONG)
            {
               if(bar.close > s.ORH)
                  TryEnter(s,DIR_LONG);
            }
            else if(s.breakout_dir == DIR_SHORT)
            {
               if(bar.close < s.ORL)
                  TryEnter(s,DIR_SHORT);
            }
         }
         break;
      }

      case STATE_TRADED:
      case STATE_RESET:
      default:
         break;
   }
}

//+------------------------------------------------------------------+
//| Process new closed bar                                            |
//+------------------------------------------------------------------+
void ProcessOnNewClosedBar()
{
   CheckForNewUTCDay();

   MqlRates bar_arr[];
   ArraySetAsSeries(bar_arr,true);
   if(CopyRates(_Symbol,InpConfirm_TF,1,1,bar_arr)!=1)
      return;

   const MqlRates bar = bar_arr[0];
   const datetime now_server = TimeTradeServer();

   for(int i=0;i<3;i++)
      ProcessSessionOnClosedBar(g_sessions[i],now_server,bar);

   // Status text in the chart comment
   string txt;
   txt = StringFormat("OR EA | %s | Server: %s | UTC: %s | Trailing: %s\n",
                      _Symbol,
                      TimeToString(TimeTradeServer(),TIME_DATE|TIME_SECONDS),
                      TimeToString(TimeGMT(),TIME_DATE|TIME_SECONDS),
                      g_trailing_enabled ? "ON" : "OFF");

   for(int i=0;i<3;i++)
   {
      SessionData &s=g_sessions[i];
      if(!s.enabled)
      {
         txt += StringFormat("%s: DISABLED\n",s.name);
         continue;
      }
      txt += StringFormat("%s: %s | ORH=%.5f ORL=%.5f | Dir=%s | Retest=%s | Taken=%s\n",
                          s.name,
                          StateToString(s.state),
                          s.ORH,s.ORL,
                          DirToString(s.breakout_dir),
                          (s.retest_seen?"Y":"N"),
                          (s.trade_taken?"Y":"N"));
   }
   Comment(txt);
}

//+------------------------------------------------------------------+
//| Detect new closed bar of Confirm_TF                               |
//+------------------------------------------------------------------+
void CheckNewClosedConfirmBar()
{
   // Use the open time of bar #1 (last closed bar)
   const datetime t = iTime(_Symbol,InpConfirm_TF,1);
   if(t<=0)
      return;

   if(t != g_last_confirm_closed_bar_time)
   {
      g_last_confirm_closed_bar_time = t;
      ProcessOnNewClosedBar();
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   if(!IsConfirmTFAllowed(InpConfirm_TF))
   {
      Print("InpConfirm_TF must be M1/M3/M5.");
      return(INIT_FAILED);
   }

   if(!IsORMinutesAllowed(InpOR_Minutes))
   {
      Print("InpOR_Minutes must be 5/15/30.");
      return(INIT_FAILED);
   }

   g_trailing_enabled = InpTrailingEnabledInit;

   if(!SymbolSelect(_Symbol,true))
   {
      Print("Failed to select symbol ",_Symbol);
      return(INIT_FAILED);
   }

   InitSessions();
   CheckForNewUTCDay();

   CreateTrailingButton();

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   EventSetTimer(1);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectDelete(0,g_btn_trailing_name);
   Comment("");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   CheckNewClosedConfirmBar();
   ManageTrailing();
}

//+------------------------------------------------------------------+
//| Timer event handler                                               |
//+------------------------------------------------------------------+
void OnTimer()
{
   CheckNewClosedConfirmBar();
   ManageTrailing();
}

//+------------------------------------------------------------------+
//| Chart event handler                                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id==CHARTEVENT_OBJECT_CLICK && sparam==g_btn_trailing_name)
   {
      g_trailing_enabled = !g_trailing_enabled;
      UpdateTrailingButton();

      if(InpPrintDebug)
         Print("Trailing toggled: ",(g_trailing_enabled?"ON":"OFF"));
   }
}
//+------------------------------------------------------------------+