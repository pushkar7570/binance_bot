//+------------------------------------------------------------------+
//|                                              VENOM_MODEL_BOT.mq5  |
//|  VENOM Model Trading Bot (MT5 / MQL5)                             |
//|  Implements: HTF Bias, Liquidity mapping, Sweep->MSS->FVG+OB+OTE, |
//|  Fixed RRR TP1/TP2, Partial close, BE SL, Daily risk gates, CSV.  |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property description "VENOM MODEL BOT — Sweep + MSS + FVG/OB/OTE confluence (MT5 EA)"

// -------------------------- INPUTS ---------------------------------
// ACCOUNT SETTINGS
input double InpRiskPercentPerTrade       = 0.25;   // RISK_PERCENT_PER_TRADE
input int    InpMaxTradesPerDay           = 2;      // MAX_TRADES_PER_DAY
input double InpMaxDailyRiskPercent       = 0.50;   // MAX_DAILY_RISK_PERCENT
input int    InpMaxDailyLosses            = 2;      // MAX_CONSECUTIVE_LOSSES_DAILY

// INSTRUMENT SETTINGS
input string InpSymbol                    = "";     // If empty => current chart symbol
input double InpPipValuePerLot            = 10.0;   // PIP_VALUE_PER_LOT (USD per pip per 1 lot)
input bool   InpAutoPipSize               = true;   // Auto pip size from symbol digits
input double InpManualPipSize             = 0.01;   // Used only if AutoPipSize=false

// TIMEFRAMES
input ENUM_TIMEFRAMES InpHTF_D1_Timeframe = PERIOD_D1;   // HTF_BIAS_TIMEFRAME_1
input ENUM_TIMEFRAMES InpHTF_H4_Timeframe = PERIOD_H4;   // HTF_BIAS_TIMEFRAME_2
input ENUM_TIMEFRAMES InpStructureTF      = PERIOD_M15;  // STRUCTURE_TIMEFRAME
input ENUM_TIMEFRAMES InpEntryTF          = PERIOD_M1;   // ENTRY_TIMEFRAME

// KILL ZONES (New York Time)
enum ENUM_KILL_ZONE
{
   KZ_LONDON_OPEN   = 0,  // 02:00 - 05:00 NY
   KZ_NEWYORK_OPEN  = 1   // 07:00 - 10:00 NY
};
input ENUM_KILL_ZONE InpActiveKillZone    = KZ_NEWYORK_OPEN;

// SESSION TIMES (New York Time)
input string InpAsianSessionStart         = "19:00"; // ASIAN_SESSION_START (previous day)
input string InpAsianSessionEnd           = "00:00"; // ASIAN_SESSION_END
input string InpPrevDayStart              = "00:00"; // PREVIOUS_DAY_START
input string InpPrevDayEnd                = "23:59"; // PREVIOUS_DAY_END

// STRATEGY PARAMETERS
input double InpFvgEntryLevel             = 0.50;   // FVG_ENTRY_LEVEL
input double InpOTE_FibUpper              = 0.62;   // OTE_FIB_UPPER
input double InpOTE_FibLower              = 0.79;   // OTE_FIB_LOWER
input double InpMinRRR                    = 2.0;    // MIN_RISK_REWARD_RATIO
input double InpTP1_RRR                   = 2.0;    // TP1_RRR
input double InpTP2_RRR                   = 3.0;    // TP2_RRR
input int    InpTP1_ClosePercent          = 60;     // TP1_POSITION_CLOSE_PERCENT
input int    InpTP2_ClosePercent          = 40;     // TP2_POSITION_CLOSE_PERCENT
input double InpSL_BufferPips             = 1.0;    // SL_BUFFER_BEYOND_SWEEP
input bool   InpMoveSLToBEAtTP1           = true;   // MOVE_SL_TO_BREAKEVEN_AT_TP1

// LOOKBACK SETTINGS
input int    InpSwingLookbackHTF          = 50;     // SWING_LOOKBACK_BARS_HTF
input int    InpSwingLookbackStructure    = 30;     // SWING_LOOKBACK_BARS_STRUCTURE
input int    InpSwingStrength             = 3;      // SWING_DETECTION_STRENGTH
input double InpEqualLevelTolerance       = 0.0005; // EQUAL_LEVEL_TOLERANCE
input double InpFVG_MinSize               = 0.50;   // FVG_MIN_SIZE (price units)
input int    InpOB_MaxBarsBack            = 5;      // OB_MAX_BARS_BEFORE_DISPLACEMENT

// EXECUTION / EA SETTINGS
input ulong  InpMagicNumber               = 26032026;
input int    InpSlippagePoints            = 20;
input string InpJournalFileName           = "venom_trade_journal.csv";

// -------------------------- DATA STRUCTS ----------------------------
enum ENUM_DIR
{
   DIR_NEUTRAL = 0,
   DIR_BULLISH = 1,
   DIR_BEARISH = -1,
   DIR_NO_TRADE = 2
};
enum ENUM_ZONE
{
   ZONE_EQUILIBRIUM = 0,
   ZONE_DISCOUNT    = 1,
   ZONE_PREMIUM     = 2
};
enum ENUM_LIQ_TYPE
{
   LIQ_BUY_SIDE  = 0,
   LIQ_SELL_SIDE = 1
};
enum ENUM_SWING_TYPE
{
   SWING_HIGH = 0,
   SWING_LOW  = 1
};
enum ENUM_FVG_DIR
{
   FVG_BULLISH = 0,
   FVG_BEARISH = 1
};
enum ENUM_TRADE_STATUS
{
   TS_NONE = 0,
   TS_PENDING,
   TS_ACTIVE,
   TS_TP1_HIT,
   TS_COMPLETED,
   TS_STOPPED,
   TS_CANCELLED
};
enum ENUM_LIQ_SOURCE
{
   SRC_EQUAL_HIGHS = 0,
   SRC_EQUAL_LOWS,
   SRC_SESSION_HIGH,
   SRC_SESSION_LOW,
   SRC_PREV_DAY_HIGH,
   SRC_PREV_DAY_LOW,
   SRC_SWING_HIGH,
   SRC_SWING_LOW
};

struct Candle
{
   datetime open_time;
   double   open_price;
   double   high_price;
   double   low_price;
   double   close_price;
   long     volume;
   bool     is_bullish;
   bool     is_bearish;
   double   body_size;
   double   upper_wick;
   double   lower_wick;
};

struct SwingPoint
{
   double          price;
   datetime        time;
   ENUM_SWING_TYPE type;
   ENUM_TIMEFRAMES timeframe;
};

struct LiquidityLevel
{
   double           price;
   ENUM_LIQ_TYPE    type;
   ENUM_LIQ_SOURCE  source;
   datetime         time_created;
   bool             is_swept;
   datetime         sweep_time;
};

struct FairValueGap
{
   double        upper_boundary;
   double        lower_boundary;
   double        midpoint;
   ENUM_FVG_DIR  direction;
   double        size;
   datetime      time_created;
   ENUM_TIMEFRAMES timeframe;
   bool          is_mitigated;
};

struct OrderBlock
{
   double        upper_boundary;
   double        lower_boundary;
   ENUM_FVG_DIR  direction;      // bullish/bearish
   Candle        candle_data;
   datetime      time_created;
   ENUM_TIMEFRAMES timeframe;
   bool          is_mitigated;
};

struct OTEZone
{
   double       upper_boundary;
   double       lower_boundary;
   double       fib_swing_high;
   double       fib_swing_low;
   ENUM_DIR     direction;
};

struct Bias
{
   ENUM_DIR  daily_direction;
   ENUM_DIR  four_hour_direction;
   ENUM_DIR  combined_bias;
   ENUM_ZONE price_in_zone;
};

struct SweepData
{
   bool          swept;
   int           level_index;
   LiquidityLevel level;
   Candle        sweep_candle;
   double        sweep_extreme_price;
};

struct MSSData
{
   bool     shifted;
   double   mss_level;
   Candle   mss_candle;
   datetime mss_time;
};

struct SetupData
{
   bool        setup_found;
   string      reason;
   FairValueGap fvg;
   OrderBlock   ob;
   OTEZone      ote;
   double       entry_price;
   double       alternative_entry;
};

struct TradePlan
{
   string            instrument;
   ENUM_DIR          direction;      // DIR_BULLISH => LONG, DIR_BEARISH => SHORT
   double            entry_price;
   double            stop_loss_price;
   double            tp1_price;
   double            tp2_price;
   double            total_lot_size;
   double            tp1_lot_size;
   double            tp2_lot_size;
   double            risk_amount_usd;
   ENUM_TRADE_STATUS status;
   bool              sl_moved_to_be;

   ulong             order_ticket;
   ulong             position_ticket;
   long              position_id;     // DEAL_POSITION_ID mapping
   datetime          created_time;
};

// -------------------------- GLOBAL STATE ----------------------------
string g_symbol;

Bias g_bias;
bool g_daily_prep_done = false;
bool g_skip_today = false;

LiquidityLevel g_liq_buy[];
LiquidityLevel g_liq_sell[];

bool g_sweep_detected = false;
bool g_mss_detected   = false;

SweepData g_sweep;
MSSData   g_mss;

TradePlan g_trade;
bool g_has_trade = false;

// Daily gates
int    g_today_trade_count = 0;
int    g_today_loss_count  = 0;
double g_today_risk_used_usd = 0.0;
double g_day_start_balance = 0.0;

// Time tracking
int      g_last_ny_date_key = 0;
datetime g_last_entry_bar_time = 0;

// -------------------------- UTILITIES -------------------------------
int DaysInMonth(const int year, const int mon)
{
   if(mon==1 || mon==3 || mon==5 || mon==7 || mon==8 || mon==10 || mon==12) return 31;
   if(mon==4 || mon==6 || mon==9 || mon==11) return 30;

   // February
   bool leap = ((year%4==0 && year%100!=0) || (year%400==0));
   return leap ? 29 : 28;
}

void DecrementDate(int &year, int &mon, int &day)
{
   day--;
   if(day>=1) return;
   mon--;
   if(mon>=1)
   {
      day = DaysInMonth(year, mon);
      return;
   }
   year--;
   mon = 12;
   day = DaysInMonth(year, mon);
}

void IncrementDate(int &year, int &mon, int &day)
{
   day++;
   int dim = DaysInMonth(year, mon);
   if(day<=dim) return;
   day = 1;
   mon++;
   if(mon<=12) return;
   mon = 1;
   year++;
}

int DateKey(const int y, const int m, const int d)
{
   return y*10000 + m*100 + d;
}

string DirToString(const ENUM_DIR d)
{
   if(d==DIR_BULLISH) return "BULLISH";
   if(d==DIR_BEARISH) return "BEARISH";
   if(d==DIR_NEUTRAL) return "NEUTRAL";
   return "NO_TRADE";
}

string ZoneToString(const ENUM_ZONE z)
{
   if(z==ZONE_DISCOUNT) return "DISCOUNT";
   if(z==ZONE_PREMIUM) return "PREMIUM";
   return "EQUILIBRIUM";
}

string TradeStatusToString(const ENUM_TRADE_STATUS s)
{
   switch(s)
   {
      case TS_PENDING:   return "PENDING";
      case TS_ACTIVE:    return "ACTIVE";
      case TS_TP1_HIT:   return "TP1_HIT";
      case TS_COMPLETED: return "COMPLETED";
      case TS_STOPPED:   return "STOPPED";
      case TS_CANCELLED: return "CANCELLED";
      default:           return "NONE";
   }
}

double GetMidPrice(const string symbol)
{
   double bid=0, ask=0;
   if(!SymbolInfoDouble(symbol, SYMBOL_BID, bid)) return 0;
   if(!SymbolInfoDouble(symbol, SYMBOL_ASK, ask)) return 0;
   if(bid<=0 || ask<=0) return 0;
   return (bid+ask)*0.5;
}

double GetPipSize(const string symbol)
{
   if(!InpAutoPipSize)
      return InpManualPipSize;

   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);

   // Common FX convention: 5 digits => pip=10*point, 3 digits => pip=10*point
   if(digits==3 || digits==5) return point*10.0;

   // Metals / indices often: pip = point (user can override if needed)
   return point;
}

int ParseHHMMToMinutes(const string hhmm)
{
   if(StringLen(hhmm)<4) return 0;
   int h = (int)StringToInteger(StringSubstr(hhmm,0,2));
   int m = (int)StringToInteger(StringSubstr(hhmm,3,2));
   if(h<0) h=0; if(h>23) h=23;
   if(m<0) m=0; if(m>59) m=59;
   return h*60 + m;
}

// ---------- Time conversion: Server -> UTC -> New York (DST aware) ---
int GetServerUtcOffsetSeconds()
{
   // Updated occasionally (server offset assumed stable intraday)
   static int cached_offset = 0;
   static datetime last_update = 0;

   datetime now = TimeTradeServer();
   if(last_update==0 || (now - last_update) > 300)
   {
      cached_offset = (int)(TimeTradeServer() - TimeGMT());
      last_update = now;
   }
   return cached_offset;
}

datetime USDstStartUtc(const int year)
{
   // 2nd Sunday in March at 07:00 UTC (2:00 EST)
   MqlDateTime dt;
   dt.year=year; dt.mon=3; dt.day=1; dt.hour=0; dt.min=0; dt.sec=0;
   datetime t = StructToTime(dt);
   int dow = TimeDayOfWeek(t); // 0=Sun
   int first_sunday = 1 + ((7 - dow) % 7);
   int second_sunday = first_sunday + 7;
   dt.day = second_sunday;
   dt.hour = 7; dt.min=0; dt.sec=0;
   return StructToTime(dt);
}

datetime USDstEndUtc(const int year)
{
   // 1st Sunday in November at 06:00 UTC (2:00 EDT)
   MqlDateTime dt;
   dt.year=year; dt.mon=11; dt.day=1; dt.hour=0; dt.min=0; dt.sec=0;
   datetime t = StructToTime(dt);
   int dow = TimeDayOfWeek(t);
   int first_sunday = 1 + ((7 - dow) % 7);
   dt.day = first_sunday;
   dt.hour = 6; dt.min=0; dt.sec=0;
   return StructToTime(dt);
}

bool IsNewYorkDST(const datetime utc_time)
{
   MqlDateTime dt;
   TimeToStruct(utc_time, dt);
   int year = dt.year;

   datetime start = USDstStartUtc(year);
   datetime end   = USDstEndUtc(year);

   return (utc_time >= start && utc_time < end);
}

datetime ServerToUTC(const datetime server_time)
{
   int off = GetServerUtcOffsetSeconds();
   return (server_time - off);
}

datetime UTCToNewYork(const datetime utc_time)
{
   int ny_off = IsNewYorkDST(utc_time) ? -4*3600 : -5*3600;
   return (utc_time + ny_off);
}

datetime ServerToNewYork(const datetime server_time)
{
   return UTCToNewYork(ServerToUTC(server_time));
}

int NewYorkDateKeyNow()
{
   datetime ny = ServerToNewYork(TimeTradeServer());
   MqlDateTime dt; TimeToStruct(ny, dt);
   return DateKey(dt.year, dt.mon, dt.day);
}

// -------------------------- MARKET / KILL ZONE ----------------------
bool IsMarketOpenToday()
{
   // Weekend gate (New York)
   datetime ny = ServerToNewYork(TimeTradeServer());
   int dow = TimeDayOfWeek(ny); // 0=Sun ... 6=Sat
   if(dow==0 || dow==6) return false;

   long trade_mode = SymbolInfoInteger(g_symbol, SYMBOL_TRADE_MODE);
   if(trade_mode==SYMBOL_TRADE_MODE_DISABLED) return false;

   MqlTick tick;
   if(!SymbolInfoTick(g_symbol, tick)) return false;
   if(tick.bid<=0 || tick.ask<=0) return false;

   return true;
}

bool IsInKillZone(const datetime server_time)
{
   datetime ny = ServerToNewYork(server_time);
   MqlDateTime dt; TimeToStruct(ny, dt);
   int mins = dt.hour*60 + dt.min;

   int start=0, end=0;
   if(InpActiveKillZone==KZ_LONDON_OPEN)
   {
      start = 2*60;  // 02:00
      end   = 5*60;  // 05:00
   }
   else
   {
      start = 7*60;  // 07:00
      end   = 10*60; // 10:00
   }

   if(start<=end)
      return (mins>=start && mins<=end);

   // Cross-midnight (not used here, but supported)
   return (mins>=start || mins<=end);
}

// -------------------------- FILE LOGGING ----------------------------
bool EnsureJournalHeader()
{
   if(FileIsExist(InpJournalFileName))
      return true;

   int h = FileOpen(InpJournalFileName, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(h==INVALID_HANDLE) return false;

   FileWrite(h,
      "timestamp","event","instrument","direction","entry","stop_loss","tp1","tp2",
      "lot_size","tp1_lots","tp2_lots","risk_usd","status","account_balance"
   );
   FileClose(h);
   return true;
}

void LogTrade(const string event_type, const TradePlan &t)
{
   EnsureJournalHeader();

   string ts = TimeToString(TimeTradeServer(), TIME_DATE|TIME_SECONDS);
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);

   int h = FileOpen(InpJournalFileName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(h!=INVALID_HANDLE)
   {
      FileSeek(h, 0, SEEK_END);
      FileWrite(h,
         ts,
         event_type,
         t.instrument,
         (t.direction==DIR_BULLISH ? "LONG" : "SHORT"),
         DoubleToString(t.entry_price, (int)SymbolInfoInteger(t.instrument,SYMBOL_DIGITS)),
         DoubleToString(t.stop_loss_price, (int)SymbolInfoInteger(t.instrument,SYMBOL_DIGITS)),
         DoubleToString(t.tp1_price, (int)SymbolInfoInteger(t.instrument,SYMBOL_DIGITS)),
         DoubleToString(t.tp2_price, (int)SymbolInfoInteger(t.instrument,SYMBOL_DIGITS)),
         DoubleToString(t.total_lot_size, 2),
         DoubleToString(t.tp1_lot_size, 2),
         DoubleToString(t.tp2_lot_size, 2),
         DoubleToString(t.risk_amount_usd, 2),
         TradeStatusToString(t.status),
         DoubleToString(bal, 2)
      );
      FileClose(h);
   }

   Print("[" + ts + "] " + event_type + " | ", (t.direction==DIR_BULLISH ? "LONG " : "SHORT "), t.instrument);
   Print("  Entry: ", t.entry_price, " | SL: ", t.stop_loss_price);
   Print("  TP1: ", t.tp1_price, " | TP2: ", t.tp2_price);
   Print("  Size: ", t.total_lot_size, " lots | Risk: $", t.risk_amount_usd, " | Status: ", TradeStatusToString(t.status));
}

// -------------------------- CANDLES ---------------------------------
void FillCandleFromRates(const MqlRates &r, Candle &c)
{
   c.open_time   = r.time;
   c.open_price  = r.open;
   c.high_price  = r.high;
   c.low_price   = r.low;
   c.close_price = r.close;
   c.volume      = (long)r.tick_volume;

   c.is_bullish = (c.close_price > c.open_price);
   c.is_bearish = (c.close_price < c.open_price);
   c.body_size  = MathAbs(c.close_price - c.open_price);
   c.upper_wick = c.high_price - MathMax(c.open_price, c.close_price);
   c.lower_wick = MathMin(c.open_price, c.close_price) - c.low_price;
}

bool GetCandles(const string symbol, const ENUM_TIMEFRAMES tf, const int start_shift, const int count, Candle &out[])
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(symbol, tf, start_shift, count, rates);
   if(copied<=0) return false;

   ArrayResize(out, copied);

   // Convert to chronological order (oldest -> newest)
   int idx=0;
   for(int i=copied-1; i>=0; i--)
   {
      Candle c;
      FillCandleFromRates(rates[i], c);
      out[idx++] = c;
   }
   return true;
}

// -------------------------- SWING DETECTION --------------------------
void DetectSwingPoints(const Candle &candles[], const int strength, const ENUM_TIMEFRAMES tf, SwingPoint &swings[])
{
   ArrayResize(swings, 0);
   int n = ArraySize(candles);
   if(n < (2*strength + 3)) return;

   for(int i=strength; i <= n - strength - 1; i++)
   {
      // Swing high
      bool is_high = true;
      for(int j=1; j<=strength; j++)
      {
         if(candles[i-j].high_price >= candles[i].high_price) { is_high=false; break; }
         if(candles[i+j].high_price >= candles[i].high_price) { is_high=false; break; }
      }
      if(is_high)
      {
         int k = ArraySize(swings);
         ArrayResize(swings, k+1);
         swings[k].price = candles[i].high_price;
         swings[k].time  = candles[i].open_time;
         swings[k].type  = SWING_HIGH;
         swings[k].timeframe = tf;
      }

      // Swing low
      bool is_low = true;
      for(int j=1; j<=strength; j++)
      {
         if(candles[i-j].low_price <= candles[i].low_price) { is_low=false; break; }
         if(candles[i+j].low_price <= candles[i].low_price) { is_low=false; break; }
      }
      if(is_low)
      {
         int k = ArraySize(swings);
         ArrayResize(swings, k+1);
         swings[k].price = candles[i].low_price;
         swings[k].time  = candles[i].open_time;
         swings[k].type  = SWING_LOW;
         swings[k].timeframe = tf;
      }
   }
}

bool GetMostRecentSwing(const SwingPoint &swings[], const ENUM_SWING_TYPE type, SwingPoint &out)
{
   int n = ArraySize(swings);
   for(int i=n-1; i>=0; i--)
   {
      if(swings[i].type==type)
      {
         out = swings[i];
         return true;
      }
   }
   return false;
}

// Return last N of given type as MOST RECENT FIRST
int GetLastNSwingsMostRecentFirst(const SwingPoint &swings[], const ENUM_SWING_TYPE type, const int N, SwingPoint &out[])
{
   ArrayResize(out, 0);
   int n = ArraySize(swings);
   int count=0;
   for(int i=n-1; i>=0 && count<N; i--)
   {
      if(swings[i].type==type)
      {
         ArrayResize(out, count+1);
         out[count] = swings[i];
         count++;
      }
   }
   return count;
}

// -------------------------- SESSION RANGE ----------------------------
bool ComputeSessionRangeNY(
   const Candle &candles[],
   const int start_min,
   const int end_min,
   const int start_date_key,
   const int end_date_key,
   const bool crosses_midnight,
   double &session_high,
   double &session_low
)
{
   session_high = -DBL_MAX;
   session_low  =  DBL_MAX;

   int n = ArraySize(candles);
   bool found=false;

   for(int i=0;i<n;i++)
   {
      datetime ny = ServerToNewYork(candles[i].open_time);
      MqlDateTime dt; TimeToStruct(ny, dt);
      int dkey = DateKey(dt.year, dt.mon, dt.day);
      int mins = dt.hour*60 + dt.min;

      bool in_session=false;
      if(!crosses_midnight)
      {
         if(dkey==start_date_key && mins>=start_min && mins<end_min)
            in_session=true;
      }
      else
      {
         if((dkey==start_date_key && mins>=start_min) ||
            (dkey==end_date_key   && mins<end_min))
            in_session=true;
      }

      if(in_session)
      {
         found=true;
         if(candles[i].high_price > session_high) session_high = candles[i].high_price;
         if(candles[i].low_price  < session_low)  session_low  = candles[i].low_price;
      }
   }

   return found;
}

// -------------------------- EQUAL LEVELS -----------------------------
bool LiquidityLevelExists(const LiquidityLevel &levels[], const double price, const ENUM_LIQ_TYPE type, const double tol)
{
   int n = ArraySize(levels);
   for(int i=0;i<n;i++)
   {
      if(levels[i].type==type && MathAbs(levels[i].price - price)<=tol)
         return true;
   }
   return false;
}

void DetectEqualLevels(const SwingPoint &swings[], const double tol, LiquidityLevel &out[])
{
   ArrayResize(out, 0);

   // Collect highs and lows
   SwingPoint highs[], lows[];
   ArrayResize(highs,0); ArrayResize(lows,0);

   int n = ArraySize(swings);
   for(int i=0;i<n;i++)
   {
      if(swings[i].type==SWING_HIGH)
      {
         int k=ArraySize(highs); ArrayResize(highs,k+1); highs[k]=swings[i];
      }
      else
      {
         int k=ArraySize(lows); ArrayResize(lows,k+1); lows[k]=swings[i];
      }
   }

   // Equal highs
   int nh = ArraySize(highs);
   for(int i=0;i<nh-1;i++)
   {
      for(int j=i+1;j<nh;j++)
      {
         if(MathAbs(highs[i].price - highs[j].price) <= tol)
         {
            double avg = (highs[i].price + highs[j].price)*0.5;
            if(!LiquidityLevelExists(out, avg, LIQ_BUY_SIDE, tol))
            {
               int k=ArraySize(out); ArrayResize(out,k+1);
               out[k].price = avg;
               out[k].type  = LIQ_BUY_SIDE;
               out[k].source= SRC_EQUAL_HIGHS;
               out[k].time_created = TimeTradeServer();
               out[k].is_swept=false;
               out[k].sweep_time=0;
            }
         }
      }
   }

   // Equal lows
   int nl = ArraySize(lows);
   for(int i=0;i<nl-1;i++)
   {
      for(int j=i+1;j<nl;j++)
      {
         if(MathAbs(lows[i].price - lows[j].price) <= tol)
         {
            double avg = (lows[i].price + lows[j].price)*0.5;
            if(!LiquidityLevelExists(out, avg, LIQ_SELL_SIDE, tol))
            {
               int k=ArraySize(out); ArrayResize(out,k+1);
               out[k].price = avg;
               out[k].type  = LIQ_SELL_SIDE;
               out[k].source= SRC_EQUAL_LOWS;
               out[k].time_created = TimeTradeServer();
               out[k].is_swept=false;
               out[k].sweep_time=0;
            }
         }
      }
   }
}

// -------------------------- FVG DETECTION ----------------------------
void DetectFairValueGaps(const Candle &candles[], const double min_size, const ENUM_TIMEFRAMES tf, FairValueGap &out[])
{
   ArrayResize(out, 0);
   int n = ArraySize(candles);
   if(n<3) return;

   for(int i=2;i<n;i++)
   {
      const Candle &c1 = candles[i-2];
      const Candle &c2 = candles[i-1];
      const Candle &c3 = candles[i];

      // Bullish FVG: c1.high < c3.low
      if(c1.high_price < c3.low_price)
      {
         double gap = c3.low_price - c1.high_price;
         if(gap >= min_size)
         {
            int k=ArraySize(out); ArrayResize(out,k+1);
            out[k].upper_boundary = c3.low_price;
            out[k].lower_boundary = c1.high_price;
            out[k].midpoint = (out[k].upper_boundary + out[k].lower_boundary)*0.5;
            out[k].direction = FVG_BULLISH;
            out[k].size = gap;
            out[k].time_created = c2.open_time;
            out[k].timeframe = tf;
            out[k].is_mitigated = false;

            // Mitigation check (simple): if any later candle low <= lower => filled
            for(int m=i+1;m<n;m++)
            {
               if(candles[m].low_price <= out[k].lower_boundary)
               {
                  out[k].is_mitigated=true;
                  break;
               }
            }
         }
      }

      // Bearish FVG: c1.low > c3.high
      if(c1.low_price > c3.high_price)
      {
         double gap = c1.low_price - c3.high_price;
         if(gap >= min_size)
         {
            int k=ArraySize(out); ArrayResize(out,k+1);
            out[k].upper_boundary = c1.low_price;
            out[k].lower_boundary = c3.high_price;
            out[k].midpoint = (out[k].upper_boundary + out[k].lower_boundary)*0.5;
            out[k].direction = FVG_BEARISH;
            out[k].size = gap;
            out[k].time_created = c2.open_time;
            out[k].timeframe = tf;
            out[k].is_mitigated = false;

            // Mitigation check (simple): if any later candle high >= upper => filled
            for(int m=i+1;m<n;m++)
            {
               if(candles[m].high_price >= out[k].upper_boundary)
               {
                  out[k].is_mitigated=true;
                  break;
               }
            }
         }
      }
   }
}

// -------------------------- ORDER BLOCK DETECTION --------------------
void DetectOrderBlocks(const Candle &candles[], const int max_bars_back, const ENUM_TIMEFRAMES tf, OrderBlock &out[])
{
   ArrayResize(out, 0);
   int n = ArraySize(candles);
   if(n<5) return;

   for(int i=1;i<n;i++)
   {
      const Candle &cur = candles[i];

      // avg body last 3 candles (if available)
      double avg_body=0.0;
      int cnt=0;
      for(int k=1;k<=3;k++)
      {
         if(i-k>=0)
         {
            avg_body += candles[i-k].body_size;
            cnt++;
         }
      }
      if(cnt>0) avg_body /= cnt;
      if(avg_body<=0) continue;

      // Bullish displacement
      if(cur.is_bullish && cur.body_size > (avg_body*1.5))
      {
         for(int j=1;j<=max_bars_back;j++)
         {
            if(i-j>=0 && candles[i-j].is_bearish)
            {
               const Candle &obc = candles[i-j];
               int k=ArraySize(out); ArrayResize(out,k+1);
               out[k].upper_boundary = obc.open_price;
               out[k].lower_boundary = obc.low_price;
               out[k].direction = FVG_BULLISH;
               out[k].candle_data = obc;
               out[k].time_created = obc.open_time;
               out[k].timeframe = tf;
               out[k].is_mitigated = false;
               break;
            }
         }
      }

      // Bearish displacement
      if(cur.is_bearish && cur.body_size > (avg_body*1.5))
      {
         for(int j=1;j<=max_bars_back;j++)
         {
            if(i-j>=0 && candles[i-j].is_bullish)
            {
               const Candle &obc = candles[i-j];
               int k=ArraySize(out); ArrayResize(out,k+1);
               out[k].upper_boundary = obc.high_price;
               out[k].lower_boundary = obc.open_price;
               out[k].direction = FVG_BEARISH;
               out[k].candle_data = obc;
               out[k].time_created = obc.open_time;
               out[k].timeframe = tf;
               out[k].is_mitigated = false;
               break;
            }
         }
      }
   }
}

// -------------------------- OTE + OVERLAP ----------------------------
OTEZone CalculateOTEZone(const double swing_low, const double swing_high, const ENUM_DIR direction)
{
   OTEZone z;
   z.fib_swing_high = swing_high;
   z.fib_swing_low  = swing_low;
   z.direction = direction;

   double range = swing_high - swing_low;
   if(range<=0) { z.upper_boundary=0; z.lower_boundary=0; return z; }

   if(direction==DIR_BULLISH)
   {
      z.upper_boundary = swing_high - (range * InpOTE_FibUpper);
      z.lower_boundary = swing_high - (range * InpOTE_FibLower);
   }
   else
   {
      z.lower_boundary = swing_low + (range * InpOTE_FibUpper);
      z.upper_boundary = swing_low + (range * InpOTE_FibLower);
   }

   // Ensure upper >= lower
   if(z.upper_boundary < z.lower_boundary)
   {
      double tmp=z.upper_boundary; z.upper_boundary=z.lower_boundary; z.lower_boundary=tmp;
   }

   return z;
}

bool ZonesOverlap(const double a_upper, const double a_lower, const double b_upper, const double b_lower)
{
   return (a_lower <= b_upper && b_lower <= a_upper);
}

// -------------------------- POSITION SIZING --------------------------
int VolumeDigitsFromStep(const double step)
{
   if(step<=0) return 2;
   double x = step;
   int digits=0;
   while(digits<8 && MathAbs(x - MathRound(x)) > 1e-12)
   {
      x *= 10.0;
      digits++;
   }
   return digits;
}

double NormalizeVolumeFloor(const string symbol, double vol)
{
   if(vol<=0) return 0.0;

   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   if(step<=0) step=0.01;

   vol = MathMax(minv, MathMin(maxv, vol));
   vol = MathFloor(vol/step)*step;

   int vd = VolumeDigitsFromStep(step);
   vol = NormalizeDouble(vol, vd);

   if(vol < minv) return 0.0;
   return vol;
}

void SplitVolumes(const string symbol, const double total, double &v_tp1, double &v_tp2)
{
   v_tp1 = 0.0;
   v_tp2 = 0.0;
   if(total<=0) return;

   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   if(step<=0) step=0.01;

   double desired_tp1 = total * ((double)InpTP1_ClosePercent/100.0);
   v_tp1 = NormalizeVolumeFloor(symbol, desired_tp1);
   if(v_tp1 < minv) v_tp1 = 0.0;

   v_tp2 = total - v_tp1;
   v_tp2 = NormalizeVolumeFloor(symbol, v_tp2);

   // If remainder becomes too small, allocate all to TP1
   if(v_tp2 < minv)
   {
      v_tp1 = total;
      v_tp2 = 0.0;
   }

   // Ensure v_tp1 also normalized if all
   v_tp1 = NormalizeVolumeFloor(symbol, v_tp1);
}

double CalculatePositionSizeLots(
   const double account_balance,
   const double risk_percent,
   const double entry_price,
   const double stop_loss_price,
   const double pip_value_per_lot,
   const double pip_size
)
{
   double risk_amount = account_balance * (risk_percent/100.0);
   double sl_dist = MathAbs(entry_price - stop_loss_price);
   if(sl_dist<=0 || pip_size<=0) return 0.0;

   double pips = sl_dist / pip_size;
   double risk_per_lot = pips * pip_value_per_lot;
   if(risk_per_lot<=0) return 0.0;

   return (risk_amount / risk_per_lot);
}

// -------------------------- HTF BIAS --------------------------------
Bias DetermineHTFBias()
{
   Bias b;
   b.daily_direction = DIR_NEUTRAL;
   b.four_hour_direction = DIR_NEUTRAL;
   b.combined_bias = DIR_NO_TRADE;
   b.price_in_zone = ZONE_EQUILIBRIUM;

   // Daily candles
   Candle d1[];
   if(!GetCandles(g_symbol, InpHTF_D1_Timeframe, 1, InpSwingLookbackHTF, d1))
      return b;

   SwingPoint d1_swings[];
   DetectSwingPoints(d1, InpSwingStrength, InpHTF_D1_Timeframe, d1_swings);

   SwingPoint sh, sl;
   if(!GetMostRecentSwing(d1_swings, SWING_HIGH, sh) || !GetMostRecentSwing(d1_swings, SWING_LOW, sl))
      return b;

   double eq = (sh.price + sl.price) * 0.5;
   double cur = GetMidPrice(g_symbol);

   if(cur < eq)
   {
      b.price_in_zone = ZONE_DISCOUNT;
      b.daily_direction = DIR_BULLISH;
   }
   else if(cur > eq)
   {
      b.price_in_zone = ZONE_PREMIUM;
      b.daily_direction = DIR_BEARISH;
   }
   else
   {
      b.price_in_zone = ZONE_EQUILIBRIUM;
      b.daily_direction = DIR_NEUTRAL;
   }

   // 4H candles
   Candle h4[];
   if(!GetCandles(g_symbol, InpHTF_H4_Timeframe, 1, InpSwingLookbackHTF, h4))
      return b;

   SwingPoint h4_swings[];
   DetectSwingPoints(h4, InpSwingStrength, InpHTF_H4_Timeframe, h4_swings);

   SwingPoint last_highs[], last_lows[];
   int nH = GetLastNSwingsMostRecentFirst(h4_swings, SWING_HIGH, 3, last_highs);
   int nL = GetLastNSwingsMostRecentFirst(h4_swings, SWING_LOW, 3, last_lows);

   bool h4_bullish = (nL>=3);
   bool h4_bearish = (nH>=3);

   // Most recent first logic:
   // Bullish = higher lows => low0 > low1 > low2
   if(h4_bullish)
   {
      for(int i=0;i<nL-1;i++)
      {
         if(last_lows[i].price <= last_lows[i+1].price) { h4_bullish=false; break; }
      }
   }

   // Bearish = lower highs => high0 < high1 < high2
   if(h4_bearish)
   {
      for(int i=0;i<nH-1;i++)
      {
         if(last_highs[i].price >= last_highs[i+1].price) { h4_bearish=false; break; }
      }
   }

   if(h4_bullish) b.four_hour_direction = DIR_BULLISH;
   else if(h4_bearish) b.four_hour_direction = DIR_BEARISH;
   else b.four_hour_direction = DIR_NEUTRAL;

   // Combine
   if(b.daily_direction==DIR_BULLISH && b.four_hour_direction==DIR_BULLISH)
      b.combined_bias = DIR_BULLISH;
   else if(b.daily_direction==DIR_BEARISH && b.four_hour_direction==DIR_BEARISH)
      b.combined_bias = DIR_BEARISH;
   else
      b.combined_bias = DIR_NO_TRADE;

   return b;
}

// -------------------------- LIQUIDITY MAPPING ------------------------
void SortLiquidityByPriceAscending(LiquidityLevel &arr[])
{
   int n=ArraySize(arr);
   for(int i=0;i<n-1;i++)
      for(int j=i+1;j<n;j++)
         if(arr[j].price < arr[i].price)
         {
            LiquidityLevel tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
         }
}
void SortLiquidityByPriceDescending(LiquidityLevel &arr[])
{
   int n=ArraySize(arr);
   for(int i=0;i<n-1;i++)
      for(int j=i+1;j<n;j++)
         if(arr[j].price > arr[i].price)
         {
            LiquidityLevel tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
         }
}

void MapLiquidityLevels()
{
   ArrayResize(g_liq_buy, 0);
   ArrayResize(g_liq_sell, 0);

   // Fetch enough M15 candles to cover prev day/session
   int need_bars = MathMax(InpSwingLookbackStructure, 300);
   Candle c[];
   if(!GetCandles(g_symbol, InpStructureTF, 1, need_bars, c))
      return;

   datetime ny_now = ServerToNewYork(TimeTradeServer());
   MqlDateTime dt; TimeToStruct(ny_now, dt);

   int y=dt.year, m=dt.mon, d=dt.day;
   int today_key = DateKey(y,m,d);

   int py=y, pm=m, pd=d;
   DecrementDate(py, pm, pd);
   int prev_key = DateKey(py, pm, pd);

   // For cross-midnight sessions, end date is next day
   int ny2=y, nm2=m, nd2=d;
   // next day key (in case end spills into today)
   int next_key = today_key;
   // (For general cross-midnight sessions)
   // next_key = DateKey(ny2,nm2,nd2); // already today

   // Session minutes
   int asian_start = ParseHHMMToMinutes(InpAsianSessionStart);
   int asian_end   = ParseHHMMToMinutes(InpAsianSessionEnd);
   bool asian_cross = (asian_start > asian_end);

   int prev_start = ParseHHMMToMinutes(InpPrevDayStart);
   int prev_end   = ParseHHMMToMinutes(InpPrevDayEnd);
   if(prev_end==23*60+59) prev_end = 24*60; // make end exclusive

   // Asian session range (start date = prev day, end date = today if crossing)
   double a_high, a_low;
   bool a_found = ComputeSessionRangeNY(
      c,
      asian_start,
      asian_end,
      prev_key,
      today_key,
      asian_cross,
      a_high,
      a_low
   );
   if(a_found)
   {
      LiquidityLevel lh, ll;
      lh.price=a_high; lh.type=LIQ_BUY_SIDE;  lh.source=SRC_SESSION_HIGH;
      lh.time_created=TimeTradeServer(); lh.is_swept=false; lh.sweep_time=0;
      ll.price=a_low;  ll.type=LIQ_SELL_SIDE; ll.source=SRC_SESSION_LOW;
      ll.time_created=TimeTradeServer(); ll.is_swept=false; ll.sweep_time=0;

      int k=ArraySize(g_liq_buy); ArrayResize(g_liq_buy,k+1); g_liq_buy[k]=lh;
      k=ArraySize(g_liq_sell); ArrayResize(g_liq_sell,k+1); g_liq_sell[k]=ll;
   }

   // Previous day range (entire prev day)
   double p_high, p_low;
   bool p_found = ComputeSessionRangeNY(
      c,
      0,
      24*60,
      prev_key,
      prev_key,
      false,
      p_high,
      p_low
   );
   if(p_found)
   {
      LiquidityLevel ph, pl;
      ph.price=p_high; ph.type=LIQ_BUY_SIDE;  ph.source=SRC_PREV_DAY_HIGH;
      ph.time_created=TimeTradeServer(); ph.is_swept=false; ph.sweep_time=0;
      pl.price=p_low;  pl.type=LIQ_SELL_SIDE; pl.source=SRC_PREV_DAY_LOW;
      pl.time_created=TimeTradeServer(); pl.is_swept=false; pl.sweep_time=0;

      int k=ArraySize(g_liq_buy); ArrayResize(g_liq_buy,k+1); g_liq_buy[k]=ph;
      k=ArraySize(g_liq_sell); ArrayResize(g_liq_sell,k+1); g_liq_sell[k]=pl;
   }

   // Swing point levels on structure
   SwingPoint s_swings[];
   DetectSwingPoints(c, InpSwingStrength, InpStructureTF, s_swings);

   int ns = ArraySize(s_swings);
   for(int i=0;i<ns;i++)
   {
      LiquidityLevel l;
      l.price = s_swings[i].price;
      l.time_created = s_swings[i].time;
      l.is_swept=false; l.sweep_time=0;

      if(s_swings[i].type==SWING_HIGH)
      {
         l.type=LIQ_BUY_SIDE;
         l.source=SRC_SWING_HIGH;
         int k=ArraySize(g_liq_buy); ArrayResize(g_liq_buy,k+1); g_liq_buy[k]=l;
      }
      else
      {
         l.type=LIQ_SELL_SIDE;
         l.source=SRC_SWING_LOW;
         int k=ArraySize(g_liq_sell); ArrayResize(g_liq_sell,k+1); g_liq_sell[k]=l;
      }
   }

   // Equal highs/lows from structure swings
   LiquidityLevel eq[];
   DetectEqualLevels(s_swings, InpEqualLevelTolerance, eq);

   int neq=ArraySize(eq);
   for(int i=0;i<neq;i++)
   {
      if(eq[i].type==LIQ_BUY_SIDE)
      {
         int k=ArraySize(g_liq_buy); ArrayResize(g_liq_buy,k+1); g_liq_buy[k]=eq[i];
      }
      else
      {
         int k=ArraySize(g_liq_sell); ArrayResize(g_liq_sell,k+1); g_liq_sell[k]=eq[i];
      }
   }

   // Sort by price
   SortLiquidityByPriceAscending(g_liq_buy);
   SortLiquidityByPriceDescending(g_liq_sell);
}

// -------------------------- SWEEP DETECTION --------------------------
SweepData DetectLiquiditySweep(const Candle &candles[], LiquidityLevel &levels[], const ENUM_DIR bias_dir)
{
   SweepData s;
   s.swept=false;
   s.level_index=-1;
   s.sweep_extreme_price=0.0;

   int n = ArraySize(candles);
   if(n<2) return s;

   const Candle &latest   = candles[n-1];
   const Candle &previous = candles[n-2];

   int ln = ArraySize(levels);
   if(ln<=0) return s;

   if(bias_dir==DIR_BULLISH)
   {
      // sweep SELL_SIDE levels
      for(int i=0;i<ln;i++)
      {
         if(levels[i].is_swept) continue;
         if(levels[i].type!=LIQ_SELL_SIDE) continue;

         bool candle_swept = (latest.low_price < levels[i].price && latest.close_price > levels[i].price);
         bool two_candle_swept = (previous.low_price < levels[i].price && latest.close_price > levels[i].price);

         if(candle_swept || two_candle_swept)
         {
            levels[i].is_swept=true;
            levels[i].sweep_time=latest.open_time;

            s.swept=true;
            s.level_index=i;
            s.level=levels[i];
            s.sweep_candle=latest;
            s.sweep_extreme_price=latest.low_price;
            return s;
         }
      }
   }
   else if(bias_dir==DIR_BEARISH)
   {
      // sweep BUY_SIDE levels
      for(int i=0;i<ln;i++)
      {
         if(levels[i].is_swept) continue;
         if(levels[i].type!=LIQ_BUY_SIDE) continue;

         bool candle_swept = (latest.high_price > levels[i].price && latest.close_price < levels[i].price);
         bool two_candle_swept = (previous.high_price > levels[i].price && latest.close_price < levels[i].price);

         if(candle_swept || two_candle_swept)
         {
            levels[i].is_swept=true;
            levels[i].sweep_time=latest.open_time;

            s.swept=true;
            s.level_index=i;
            s.level=levels[i];
            s.sweep_candle=latest;
            s.sweep_extreme_price=latest.high_price;
            return s;
         }
      }
   }
   return s;
}

// -------------------------- MSS DETECTION ----------------------------
MSSData DetectMarketStructureShift(const Candle &candles[], const ENUM_DIR bias_dir, const datetime sweep_time)
{
   MSSData m;
   m.shifted=false;
   m.mss_level=0.0;
   m.mss_time=0;

   // post-sweep candles
   Candle post[];
   ArrayResize(post,0);
   int n=ArraySize(candles);
   for(int i=0;i<n;i++)
   {
      if(candles[i].open_time >= sweep_time)
      {
         int k=ArraySize(post); ArrayResize(post,k+1); post[k]=candles[i];
      }
   }
   if(ArraySize(post)<3) return m;

   SwingPoint swings[];
   DetectSwingPoints(candles, 2, InpEntryTF, swings);

   if(bias_dir==DIR_BULLISH)
   {
      // find most recent swing high at/before sweep_time
      double mss_level=0.0;
      bool found=false;
      int sn=ArraySize(swings);
      for(int i=sn-1;i>=0;i--)
      {
         if(swings[i].type==SWING_HIGH && swings[i].time <= sweep_time)
         {
            mss_level=swings[i].price;
            found=true;
            break;
         }
      }
      if(!found) return m;

      for(int i=0;i<ArraySize(post);i++)
      {
         if(post[i].close_price > mss_level && post[i].is_bullish && post[i].body_size>0)
         {
            m.shifted=true;
            m.mss_level=mss_level;
            m.mss_candle=post[i];
            m.mss_time=post[i].open_time;
            return m;
         }
      }
   }
   else if(bias_dir==DIR_BEARISH)
   {
      // find most recent swing low at/before sweep_time
      double mss_level=0.0;
      bool found=false;
      int sn=ArraySize(swings);
      for(int i=sn-1;i>=0;i--)
      {
         if(swings[i].type==SWING_LOW && swings[i].time <= sweep_time)
         {
            mss_level=swings[i].price;
            found=true;
            break;
         }
      }
      if(!found) return m;

      for(int i=0;i<ArraySize(post);i++)
      {
         if(post[i].close_price < mss_level && post[i].is_bearish && post[i].body_size>0)
         {
            m.shifted=true;
            m.mss_level=mss_level;
            m.mss_candle=post[i];
            m.mss_time=post[i].open_time;
            return m;
         }
      }
   }
   return m;
}

// -------------------------- SETUP FINDER -----------------------------
void FilterCandlesSince(const Candle &candles[], const datetime since_time, Candle &out[])
{
   ArrayResize(out,0);
   int n=ArraySize(candles);
   for(int i=0;i<n;i++)
   {
      if(candles[i].open_time >= since_time)
      {
         int k=ArraySize(out); ArrayResize(out,k+1); out[k]=candles[i];
      }
   }
}

SetupData FindEntrySetup(const Candle &candles[], const ENUM_DIR bias_dir, const SweepData &sweep, const MSSData &mss)
{
   SetupData sd;
   sd.setup_found=false;
   sd.reason="";

   Candle post[];
   FilterCandlesSince(candles, sweep.sweep_candle.open_time, post);
   if(ArraySize(post)<10)
   {
      sd.reason="Not enough candles after sweep";
      return sd;
   }

   FairValueGap fvgs[];
   DetectFairValueGaps(post, InpFVG_MinSize, InpEntryTF, fvgs);

   // filter fvgs
   FairValueGap vf[];
   ArrayResize(vf,0);
   for(int i=0;i<ArraySize(fvgs);i++)
   {
      if(fvgs[i].is_mitigated) continue;
      if(bias_dir==DIR_BULLISH && fvgs[i].direction==FVG_BULLISH)
      {
         int k=ArraySize(vf); ArrayResize(vf,k+1); vf[k]=fvgs[i];
      }
      if(bias_dir==DIR_BEARISH && fvgs[i].direction==FVG_BEARISH)
      {
         int k=ArraySize(vf); ArrayResize(vf,k+1); vf[k]=fvgs[i];
      }
   }
   if(ArraySize(vf)==0)
   {
      sd.reason="No valid FVG found after MSS";
      return sd;
   }

   OrderBlock obs[];
   DetectOrderBlocks(post, InpOB_MaxBarsBack, InpEntryTF, obs);

   OrderBlock vo[];
   ArrayResize(vo,0);
   for(int i=0;i<ArraySize(obs);i++)
   {
      if(obs[i].is_mitigated) continue;
      if(bias_dir==DIR_BULLISH && obs[i].direction==FVG_BULLISH)
      {
         int k=ArraySize(vo); ArrayResize(vo,k+1); vo[k]=obs[i];
      }
      if(bias_dir==DIR_BEARISH && obs[i].direction==FVG_BEARISH)
      {
         int k=ArraySize(vo); ArrayResize(vo,k+1); vo[k]=obs[i];
      }
   }
   if(ArraySize(vo)==0)
   {
      sd.reason="No valid Order Block found";
      return sd;
   }

   // Fib swing points for OTE
   double fib_low=0.0, fib_high=0.0;
   if(bias_dir==DIR_BULLISH)
   {
      fib_low  = sweep.sweep_extreme_price;
      fib_high = mss.mss_candle.high_price;
   }
   else
   {
      fib_high = sweep.sweep_extreme_price;
      fib_low  = mss.mss_candle.low_price;
   }
   OTEZone ote = CalculateOTEZone(fib_low, fib_high, bias_dir);

   // Find first triple overlap
   for(int i=0;i<ArraySize(vf);i++)
   {
      for(int j=0;j<ArraySize(vo);j++)
      {
         bool fvg_ob = ZonesOverlap(
            vf[i].upper_boundary, vf[i].lower_boundary,
            vo[j].upper_boundary, vo[j].lower_boundary
         );
         if(!fvg_ob) continue;

         bool fvg_ote = ZonesOverlap(
            vf[i].upper_boundary, vf[i].lower_boundary,
            ote.upper_boundary, ote.lower_boundary
         );
         if(!fvg_ote) continue;

         double entry=0.0, alt=0.0;
         if(bias_dir==DIR_BULLISH)
         {
            entry = vf[i].upper_boundary - (vf[i].size * InpFvgEntryLevel);
            alt   = vo[j].upper_boundary;
         }
         else
         {
            entry = vf[i].lower_boundary + (vf[i].size * InpFvgEntryLevel);
            alt   = vo[j].lower_boundary;
         }

         sd.setup_found=true;
         sd.fvg=vf[i];
         sd.ob=vo[j];
         sd.ote=ote;
         sd.entry_price=entry;
         sd.alternative_entry=alt;
         sd.reason="";
         return sd;
      }
   }

   sd.reason="No FVG+OB+OTE triple confluence found";
   return sd;
}

// -------------------------- DAILY RISK LIMITS ------------------------
bool IsWithinRiskLimits()
{
   double bal = (g_day_start_balance>0 ? g_day_start_balance : AccountInfoDouble(ACCOUNT_BALANCE));
   if(g_today_trade_count >= InpMaxTradesPerDay)
      return false;

   if(g_today_loss_count >= InpMaxDailyLosses)
      return false;

   double used_pct = (bal>0 ? (g_today_risk_used_usd / bal * 100.0) : 0.0);
   if(used_pct >= InpMaxDailyRiskPercent)
      return false;

   return true;
}

// -------------------------- TRADE CONSTRUCTION -----------------------
bool ConstructTrade(const SetupData &setup, const ENUM_DIR bias_dir, const SweepData &sweep, TradePlan &out, string &reason)
{
   reason="";

   double pip_size = GetPipSize(g_symbol);
   double buffer   = InpSL_BufferPips * pip_size;

   double entry = setup.entry_price;
   double sl    = 0.0;

   if(bias_dir==DIR_BULLISH)
      sl = sweep.sweep_extreme_price - buffer;
   else
      sl = sweep.sweep_extreme_price + buffer;

   double sl_dist_primary = MathAbs(entry - sl);
   double sl_dist_alt     = MathAbs(setup.alternative_entry - sl);

   if(sl_dist_alt < sl_dist_primary)
      entry = setup.alternative_entry;

   double sl_dist = MathAbs(entry - sl);
   if(sl_dist<=0)
   {
      reason="Invalid SL distance";
      return false;
   }

   double tp1=0.0, tp2=0.0;
   if(bias_dir==DIR_BULLISH)
   {
      tp1 = entry + (sl_dist * InpTP1_RRR);
      tp2 = entry + (sl_dist * InpTP2_RRR);
   }
   else
   {
      tp1 = entry - (sl_dist * InpTP1_RRR);
      tp2 = entry - (sl_dist * InpTP2_RRR);
   }

   if(InpTP1_RRR < InpMinRRR)
   {
      reason = "TP1 RRR is below minimum";
      return false;
   }

   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double lots_raw = CalculatePositionSizeLots(bal, InpRiskPercentPerTrade, entry, sl, InpPipValuePerLot, pip_size);
   double lots = NormalizeVolumeFloor(g_symbol, lots_raw);
   if(lots<=0)
   {
      reason="Position size too small for broker constraints";
      return false;
   }

   double v1, v2;
   SplitVolumes(g_symbol, lots, v1, v2);

   out.instrument = g_symbol;
   out.direction  = bias_dir;
   out.entry_price = entry;
   out.stop_loss_price = sl;
   out.tp1_price = tp1;
   out.tp2_price = tp2;
   out.total_lot_size = lots;
   out.tp1_lot_size = v1;
   out.tp2_lot_size = v2;
   out.risk_amount_usd = bal * (InpRiskPercentPerTrade/100.0);
   out.status = TS_NONE;
   out.sl_moved_to_be=false;
   out.order_ticket=0;
   out.position_ticket=0;
   out.position_id=0;
   out.created_time=TimeTradeServer();

   return true;
}

// -------------------------- TRADE EXECUTION --------------------------
bool RetcodeSuccess(const int rc)
{
   return (rc==TRADE_RETCODE_DONE ||
           rc==TRADE_RETCODE_PLACED ||
           rc==TRADE_RETCODE_DONE_PARTIAL);
}

ENUM_ORDER_TYPE_FILLING BestFillingMode(const string symbol)
{
   long fm = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   // Some brokers return one mode; use as-is
   return (ENUM_ORDER_TYPE_FILLING)fm;
}

bool SendOrder(const MqlTradeRequest &req_in, MqlTradeResult &res)
{
   MqlTradeRequest req=req_in;
   ZeroMemory(res);

   ResetLastError();
   bool ok = OrderSend(req, res);
   if(!ok)
   {
      Print("OrderSend failed. err=", GetLastError());
      return false;
   }
   if(!RetcodeSuccess((int)res.retcode))
   {
      Print("Trade retcode not success: ", res.retcode, " / ", res.comment);
      return false;
   }
   return true;
}

bool PlaceTrade(TradePlan &t)
{
   double bid=0, ask=0;
   SymbolInfoDouble(t.instrument, SYMBOL_BID, bid);
   SymbolInfoDouble(t.instrument, SYMBOL_ASK, ask);
   double mid = (bid+ask)*0.5;

   bool is_long = (t.direction==DIR_BULLISH);

   // Decide order type (pseudocode logic)
   bool use_limit=false;
   ENUM_ORDER_TYPE otype;

   if(is_long)
   {
      if(t.entry_price < mid)
      {
         use_limit=true;
         otype = ORDER_TYPE_BUY_LIMIT;
      }
      else
      {
         use_limit=false;
         otype = ORDER_TYPE_BUY;
      }
   }
   else
   {
      if(t.entry_price > mid)
      {
         use_limit=true;
         otype = ORDER_TYPE_SELL_LIMIT;
      }
      else
      {
         use_limit=false;
         otype = ORDER_TYPE_SELL;
      }
   }

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);

   req.magic   = InpMagicNumber;
   req.symbol  = t.instrument;
   req.volume  = t.total_lot_size;
   req.sl      = t.stop_loss_price;
   req.tp      = 0.0; // TP managed manually for partial closes
   req.deviation = InpSlippagePoints;
   req.type_filling = BestFillingMode(t.instrument);

   string comment = "VENOM_MODEL";
   req.comment = comment;

   if(use_limit)
   {
      req.action = TRADE_ACTION_PENDING;
      req.type   = otype;
      req.price  = t.entry_price;
      req.type_time = ORDER_TIME_GTC;
   }
   else
   {
      req.action = TRADE_ACTION_DEAL;
      req.type   = otype;
      req.price  = is_long ? ask : bid;
   }

   if(!SendOrder(req, res))
      return false;

   t.order_ticket = res.order;
   t.status = use_limit ? TS_PENDING : TS_ACTIVE;
   t.created_time = TimeTradeServer();

   // Count trade for daily limits at placement (as per pseudocode behavior)
   g_today_trade_count++;
   g_today_risk_used_usd += t.risk_amount_usd;

   LogTrade(use_limit ? "ORDER PLACED (PENDING)" : "ORDER PLACED (MARKET)", t);
   return true;
}

// -------------------------- POSITION / ORDER LOOKUP ------------------
ulong FindPositionTicketByMagic(const string symbol, const ulong magic)
{
   int total = PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket==0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long   mg  = PositionGetInteger(POSITION_MAGIC);
      if(sym==symbol && (ulong)mg==magic)
         return ticket;
   }
   return 0;
}

bool CancelPendingOrder(const ulong order_ticket)
{
   if(order_ticket==0) return false;

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req); ZeroMemory(res);

   req.action = TRADE_ACTION_REMOVE;
   req.order  = order_ticket;
   req.symbol = g_symbol;
   req.magic  = InpMagicNumber;

   return SendOrder(req, res);
}

bool ModifyPositionSL(const ulong position_ticket, const double new_sl)
{
   if(position_ticket==0) return false;

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req); ZeroMemory(res);

   req.action   = TRADE_ACTION_SLTP;
   req.position = position_ticket;
   req.symbol   = g_symbol;
   req.magic    = InpMagicNumber;
   req.sl       = new_sl;
   req.tp       = 0.0;

   return SendOrder(req, res);
}

bool ClosePositionPartial(const ulong position_ticket, const double volume)
{
   if(position_ticket==0 || volume<=0) return false;
   if(!PositionSelectByTicket(position_ticket)) return false;

   string sym = PositionGetString(POSITION_SYMBOL);
   long ptype = PositionGetInteger(POSITION_TYPE);
   double bid=0, ask=0;
   SymbolInfoDouble(sym, SYMBOL_BID, bid);
   SymbolInfoDouble(sym, SYMBOL_ASK, ask);

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req); ZeroMemory(res);

   req.action = TRADE_ACTION_DEAL;
   req.position = position_ticket;
   req.symbol = sym;
   req.magic  = InpMagicNumber;
   req.volume = volume;
   req.deviation = InpSlippagePoints;
   req.type_filling = BestFillingMode(sym);

   if(ptype==POSITION_TYPE_BUY)
   {
      req.type  = ORDER_TYPE_SELL;
      req.price = bid;
   }
   else
   {
      req.type  = ORDER_TYPE_BUY;
      req.price = ask;
   }

   return SendOrder(req, res);
}

bool ClosePositionFull(const ulong position_ticket)
{
   if(position_ticket==0) return false;
   if(!PositionSelectByTicket(position_ticket)) return false;

   double vol = PositionGetDouble(POSITION_VOLUME);
   vol = NormalizeVolumeFloor(g_symbol, vol);
   if(vol<=0) return false;

   return ClosePositionPartial(position_ticket, vol);
}

// Determine if trade closed; if so decide STOPPED vs COMPLETED via last deal profit
bool TryUpdateClosedTradeStatus(TradePlan &t)
{
   // If no position exists anymore, look into history deals for this position_id (if known)
   if(t.position_id<=0)
      return false;

   datetime from = t.created_time - 86400; // safe window
   datetime to   = TimeTradeServer();
   if(!HistorySelect(from, to))
      return false;

   double last_profit = 0.0;
   bool   found_out = false;

   int deals = HistoryDealsTotal();
   for(int i=deals-1;i>=0;i--)
   {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket==0) continue;

      long pos_id = (long)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
      if(pos_id != t.position_id) continue;

      long entry = (long)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
      if(entry==DEAL_ENTRY_OUT || entry==DEAL_ENTRY_OUT_BY)
      {
         last_profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
         found_out = true;
         break;
      }
   }

   if(!found_out) return false;

   if(last_profit < 0.0)
   {
      t.status = TS_STOPPED;
      g_today_loss_count++;
      LogTrade("STOPPED OUT", t);
   }
   else
   {
      t.status = TS_COMPLETED;
      LogTrade("TRADE CLOSED", t);
   }

   return true;
}

// -------------------------- TRADE MANAGEMENT -------------------------
void ResetEntryEngine()
{
   g_sweep_detected=false;
   g_mss_detected=false;
   ZeroMemory(g_sweep);
   ZeroMemory(g_mss);
}

void ManageTrade()
{
   if(!g_has_trade) return;

   // Pending order management
   if(g_trade.status==TS_PENDING)
   {
      // If kill zone ended, cancel pending
      if(!IsInKillZone(TimeTradeServer()))
      {
         if(CancelPendingOrder(g_trade.order_ticket))
         {
            g_trade.status = TS_CANCELLED;
            LogTrade("ORDER CANCELLED — Kill zone ended", g_trade);
            g_has_trade=false;
            ResetEntryEngine();
            return;
         }
      }

      // If position appears => filled
      ulong pos_ticket = FindPositionTicketByMagic(g_trade.instrument, InpMagicNumber);
      if(pos_ticket!=0 && PositionSelectByTicket(pos_ticket))
      {
         g_trade.position_ticket = pos_ticket;
         g_trade.position_id = (long)PositionGetInteger(POSITION_IDENTIFIER);
         g_trade.status = TS_ACTIVE;
         LogTrade("ORDER FILLED", g_trade);
      }

      return;
   }

   // Active / TP1 hit management
   if(g_trade.status==TS_ACTIVE || g_trade.status==TS_TP1_HIT)
   {
      // Ensure position still exists
      ulong pos_ticket = FindPositionTicketByMagic(g_trade.instrument, InpMagicNumber);
      if(pos_ticket==0)
      {
         // closed externally or by SL; use history if possible
         if(TryUpdateClosedTradeStatus(g_trade))
         {
            g_has_trade=false;
            ResetEntryEngine();
         }
         return;
      }

      if(!PositionSelectByTicket(pos_ticket)) return;
      g_trade.position_ticket = pos_ticket;
      if(g_trade.position_id<=0)
         g_trade.position_id = (long)PositionGetInteger(POSITION_IDENTIFIER);

      double bid=0, ask=0;
      SymbolInfoDouble(g_trade.instrument, SYMBOL_BID, bid);
      SymbolInfoDouble(g_trade.instrument, SYMBOL_ASK, ask);

      bool is_long = (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY);
      double px = is_long ? bid : ask; // conservative for trigger

      // TP1
      if(g_trade.status==TS_ACTIVE)
      {
         if(is_long && px >= g_trade.tp1_price && g_trade.tp1_lot_size>0)
         {
            if(ClosePositionPartial(g_trade.position_ticket, g_trade.tp1_lot_size))
            {
               g_trade.status = TS_TP1_HIT;
               LogTrade("TP1 HIT — Partial close", g_trade);

               if(InpMoveSLToBEAtTP1 && g_trade.tp2_lot_size>0)
               {
                  if(ModifyPositionSL(g_trade.position_ticket, g_trade.entry_price))
                  {
                     g_trade.sl_moved_to_be=true;
                     LogTrade("SL moved to breakeven", g_trade);
                  }
               }
            }
         }
         else if(!is_long && px <= g_trade.tp1_price && g_trade.tp1_lot_size>0)
         {
            if(ClosePositionPartial(g_trade.position_ticket, g_trade.tp1_lot_size))
            {
               g_trade.status = TS_TP1_HIT;
               LogTrade("TP1 HIT — Partial close", g_trade);

               if(InpMoveSLToBEAtTP1 && g_trade.tp2_lot_size>0)
               {
                  if(ModifyPositionSL(g_trade.position_ticket, g_trade.entry_price))
                  {
                     g_trade.sl_moved_to_be=true;
                     LogTrade("SL moved to breakeven", g_trade);
                  }
               }
            }
         }
      }

      // TP2
      if(g_trade.status==TS_TP1_HIT)
      {
         // If no TP2 portion, finish
         if(g_trade.tp2_lot_size<=0)
         {
            g_trade.status=TS_COMPLETED;
            LogTrade("TRADE COMPLETED — No TP2 portion", g_trade);
            g_has_trade=false;
            ResetEntryEngine();
            return;
         }

         if(is_long && px >= g_trade.tp2_price)
         {
            if(ClosePositionFull(g_trade.position_ticket))
            {
               g_trade.status=TS_COMPLETED;
               LogTrade("TP2 HIT — Trade fully closed", g_trade);
               g_has_trade=false;
               ResetEntryEngine();
            }
         }
         else if(!is_long && px <= g_trade.tp2_price)
         {
            if(ClosePositionFull(g_trade.position_ticket))
            {
               g_trade.status=TS_COMPLETED;
               LogTrade("TP2 HIT — Trade fully closed", g_trade);
               g_has_trade=false;
               ResetEntryEngine();
            }
         }
      }
   }
}

// -------------------------- DAILY RESET / PREP -----------------------
void ResetDailyState()
{
   g_today_trade_count = 0;
   g_today_loss_count  = 0;
   g_today_risk_used_usd = 0.0;
   g_day_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);

   g_daily_prep_done=false;
   g_skip_today=false;

   g_has_trade=false;
   ZeroMemory(g_trade);

   ResetEntryEngine();

   Print("[NEW DAY] State reset (NY date change). Balance start: ", g_day_start_balance);
}

void DailyPreparation()
{
   if(g_daily_prep_done) return;

   if(!IsMarketOpenToday())
   {
      Print("[SKIP] Market is closed today (weekend / no quotes / trade disabled).");
      g_skip_today = true;
      g_daily_prep_done = true;
      return;
   }

   g_bias = DetermineHTFBias();
   Print("[BIAS] Daily: ", DirToString(g_bias.daily_direction));
   Print("[BIAS] 4H: ", DirToString(g_bias.four_hour_direction));
   Print("[BIAS] Combined: ", DirToString(g_bias.combined_bias));
   Print("[BIAS] Price zone: ", ZoneToString(g_bias.price_in_zone));

   if(g_bias.combined_bias==DIR_NO_TRADE)
   {
      Print("[SKIP] Conflicting bias — no trades today.");
      g_skip_today=true;
      g_daily_prep_done=true;
      return;
   }

   MapLiquidityLevels();
   Print("[LIQUIDITY] Buy-side levels: ", ArraySize(g_liq_buy));
   Print("[LIQUIDITY] Sell-side levels: ", ArraySize(g_liq_sell));

   g_daily_prep_done=true;
   Print("[PREP] Daily preparation complete. Waiting for kill zone...");
}

// -------------------------- ENTRY ENGINE (NEW BAR) -------------------
int CountCandlesAfterTime(const Candle &candles[], const datetime t0)
{
   int n=ArraySize(candles);
   int cnt=0;
   for(int i=0;i<n;i++)
      if(candles[i].open_time > t0) cnt++;
   return cnt;
}

void OnNewEntryBar()
{
   if(g_skip_today) return;
   if(!g_daily_prep_done) return;

   // Manage existing trade state handled elsewhere; only look for new if no trade
   if(g_has_trade) return;

   datetime now = TimeTradeServer();

   // Kill zone gate
   if(!IsInKillZone(now))
      return;

   // Risk limits gate
   if(!IsWithinRiskLimits())
   {
      Print("[GATE] Risk limits reached — no new trades.");
      return;
   }

   // Fetch entry candles (closed bars only)
   Candle entry[];
   if(!GetCandles(g_symbol, InpEntryTF, 1, 120, entry))
      return;

   // STEP A: Sweep
   if(!g_sweep_detected)
   {
      // Target liquidity depends on bias
      SweepData s;
      if(g_bias.combined_bias==DIR_BULLISH)
         s = DetectLiquiditySweep(entry, g_liq_sell, g_bias.combined_bias);
      else
         s = DetectLiquiditySweep(entry, g_liq_buy, g_bias.combined_bias);

      if(s.swept)
      {
         g_sweep_detected=true;
         g_sweep = s;
         Print("[SWEEP] Liquidity swept at ", g_sweep.level.price,
               " | Source=", (int)g_sweep.level.source,
               " | Extreme=", g_sweep.sweep_extreme_price);
      }
      else
      {
         return;
      }
   }

   // STEP B: MSS
   if(g_sweep_detected && !g_mss_detected)
   {
      MSSData m = DetectMarketStructureShift(entry, g_bias.combined_bias, g_sweep.sweep_candle.open_time);
      if(m.shifted)
      {
         g_mss_detected=true;
         g_mss = m;
         Print("[MSS] Confirmed. MSS level=", g_mss.mss_level, " | time=", TimeToString(g_mss.mss_time, TIME_DATE|TIME_SECONDS));
      }
      else
      {
         int since = CountCandlesAfterTime(entry, g_sweep.sweep_candle.open_time);
         if(since > 30)
         {
            Print("[INVALID] Too many candles since sweep without MSS — resetting sweep/MSS.");
            ResetEntryEngine();
         }
         return;
      }
   }

   // STEP C: Setup (FVG + OB + OTE)
   if(g_sweep_detected && g_mss_detected)
   {
      SetupData setup = FindEntrySetup(entry, g_bias.combined_bias, g_sweep, g_mss);
      if(!setup.setup_found)
      {
         Print("[WAIT] No triple confluence yet: ", setup.reason);
         return;
      }

      Print("[SETUP] Triple confluence found!");
      Print("  FVG: ", setup.fvg.lower_boundary, " - ", setup.fvg.upper_boundary);
      Print("  OB:  ", setup.ob.lower_boundary,  " - ", setup.ob.upper_boundary);
      Print("  OTE: ", setup.ote.lower_boundary, " - ", setup.ote.upper_boundary);
      Print("  Entry price: ", setup.entry_price);

      // STEP D: Construct trade
      TradePlan tp;
      string reason="";
      if(!ConstructTrade(setup, g_bias.combined_bias, g_sweep, tp, reason))
      {
         Print("[SKIP] Trade invalid: ", reason);
         return;
      }

      Print("[TRADE] Valid trade constructed");
      Print("  Direction: ", (tp.direction==DIR_BULLISH ? "LONG" : "SHORT"));
      Print("  Entry: ", tp.entry_price);
      Print("  SL: ", tp.stop_loss_price);
      Print("  TP1 (1:2): ", tp.tp1_price);
      Print("  TP2 (1:3): ", tp.tp2_price);
      Print("  Lot size: ", tp.total_lot_size);
      Print("  TP1 lots: ", tp.tp1_lot_size, " | TP2 lots: ", tp.tp2_lot_size);
      Print("  Risk: $", tp.risk_amount_usd);

      // STEP E: Execute
      if(PlaceTrade(tp))
      {
         g_trade = tp;
         g_has_trade = true;
         Print("[EXECUTED] Trade placed successfully.");
      }
      else
      {
         Print("[ERROR] Trade execution failed.");
      }
   }
}

// -------------------------- INIT / TICK ------------------------------
int OnInit()
{
   g_symbol = (InpSymbol=="" ? _Symbol : InpSymbol);
   if(!SymbolSelect(g_symbol, true))
   {
      Print("Failed to select symbol: ", g_symbol);
      return INIT_FAILED;
   }

   EnsureJournalHeader();

   g_last_ny_date_key = NewYorkDateKeyNow();
   g_day_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);

   Print("═══════════════════════════════════════");
   Print("  VENOM MODEL BOT — STARTING UP");
   Print("  Symbol: ", g_symbol);
   Print("  Risk: ", DoubleToString(InpRiskPercentPerTrade,2), "% per trade");
   Print("  Kill Zone: ", (InpActiveKillZone==KZ_NEWYORK_OPEN ? "New_York_Open (07:00-10:00 NY)" : "London_Open (02:00-05:00 NY)"));
   Print("  TP1 RRR: 1:", DoubleToString(InpTP1_RRR,2));
   Print("  TP2 RRR: 1:", DoubleToString(InpTP2_RRR,2));
   Print("═══════════════════════════════════════");

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   Print("VENOM MODEL BOT stopped. reason=", reason);
}

void OnTick()
{
   // Daily reset by NY date
   int ny_key = NewYorkDateKeyNow();
   if(ny_key != g_last_ny_date_key)
   {
      g_last_ny_date_key = ny_key;
      ResetDailyState();
   }

   // Daily prep once
   if(!g_daily_prep_done && !g_skip_today)
      DailyPreparation();

   // Manage live trade continuously
   ManageTrade();

   // New-bar logic on ENTRY TF
   datetime bt = iTime(g_symbol, InpEntryTF, 0);
   if(bt==0) return;

   if(g_last_entry_bar_time==0)
   {
      g_last_entry_bar_time = bt;
      return;
   }

   if(bt != g_last_entry_bar_time)
   {
      g_last_entry_bar_time = bt;
      OnNewEntryBar();
   }
}
//+------------------------------------------------------------------+