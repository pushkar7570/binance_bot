ea_code = r'''//+------------------------------------------------------------------+
//|                                          VenomModelEA.mq5        |
//|                        Venom Model - ICT Strategy EA             |
//|                        Hedging Account | MT5                     |
//+------------------------------------------------------------------+
#property copyright "Venom Model EA"
#property link      ""
#property version   "1.00"
#property strict
#property description "ICT Venom Model Trading Strategy EA"
#property description "FVG + Order Block + OTE Fibonacci Confluence"

//+------------------------------------------------------------------+
//| INCLUDES                                                          |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+

//--- Account & Risk
input group "=== ACCOUNT & RISK SETTINGS ==="
input double   InpRiskPercent           = 0.25;    // Risk % per trade idea
input int      InpMaxTradesPerDay       = 2;       // Max trade ideas per day
input double   InpMaxDailyRiskPercent   = 0.50;    // Max daily risk %
input int      InpMaxConsecLossesDaily  = 2;       // Max consecutive losses per day

//--- Instrument
input group "=== INSTRUMENT SETTINGS ==="
input string   InpSymbol                = "XAUUSD"; // Trading Symbol
input double   InpPipValuePerLot        = 10.0;     // Pip value per standard lot

//--- Timeframes
input group "=== TIMEFRAME SETTINGS ==="
input ENUM_TIMEFRAMES InpHTF1           = PERIOD_D1;  // HTF Bias Timeframe 1 (Daily)
input ENUM_TIMEFRAMES InpHTF2           = PERIOD_H4;  // HTF Bias Timeframe 2 (4H)
input ENUM_TIMEFRAMES InpStructureTF    = PERIOD_M15; // Structure Timeframe (15min)
input ENUM_TIMEFRAMES InpEntryTF        = PERIOD_M1;  // Entry Timeframe (1min)

//--- Kill Zone Times (EST)
input group "=== KILL ZONE TIMES (EST) ==="
input int      InpLondonStartHour      = 2;   // London Kill Zone Start Hour (EST)
input int      InpLondonStartMin       = 0;   // London Kill Zone Start Minute
input int      InpLondonEndHour        = 5;   // London Kill Zone End Hour (EST)
input int      InpLondonEndMin         = 0;   // London Kill Zone End Minute
input int      InpNYStartHour          = 7;   // New York Kill Zone Start Hour (EST)
input int      InpNYStartMin           = 0;   // New York Kill Zone Start Minute
input int      InpNYEndHour            = 10;  // New York Kill Zone End Hour (EST)
input int      InpNYEndMin             = 0;   // New York Kill Zone End Minute

//--- Session Times (EST)
input group "=== SESSION TIMES (EST) ==="
input int      InpAsianStartHour       = 19;  // Asian Session Start Hour (EST, prev day)
input int      InpAsianStartMin        = 0;   // Asian Session Start Minute
input int      InpAsianEndHour         = 0;   // Asian Session End Hour (EST)
input int      InpAsianEndMin          = 0;   // Asian Session End Minute

//--- Strategy Parameters
input group "=== STRATEGY PARAMETERS ==="
input double   InpFVGEntryLevel        = 0.50;  // FVG Entry Level (0.0=bottom, 1.0=top)
input double   InpOTEFibUpper          = 0.62;  // OTE Fib Upper (62%)
input double   InpOTEFibLower          = 0.79;  // OTE Fib Lower (79%)
input double   InpMinRRR               = 2.0;   // Minimum Risk:Reward Ratio
input double   InpTP1RRR               = 2.0;   // TP1 RRR multiplier
input double   InpTP2RRR               = 3.0;   // TP2 RRR multiplier
input double   InpTP1ClosePercent      = 60.0;  // TP1 Close % of position
input double   InpSLBuffer             = 1.0;   // SL Buffer beyond sweep (price units)
input bool     InpMoveSLtoBE           = true;  // Move SL to Breakeven at TP1

//--- Lookback Settings
input group "=== LOOKBACK SETTINGS ==="
input int      InpSwingLookbackHTF     = 50;    // Swing lookback bars (HTF)
input int      InpSwingLookbackStruct  = 30;    // Swing lookback bars (Structure)
input int      InpSwingStrength        = 3;     // Swing detection strength
input int      InpSwingStrengthEntry   = 2;     // Swing detection strength (Entry TF)
input double   InpEqualLevelTolerance  = 0.50;  // Equal level tolerance (price units)
input double   InpFVGMinSize           = 0.50;  // Minimum FVG size (price units)
input int      InpOBMaxBarsBack        = 5;     // OB max bars lookback
input int      InpMaxCandlesAfterSweep = 30;    // Max candles after sweep for MSS

//--- EA Identification
input group "=== EA SETTINGS ==="
input int      InpMagicNumber          = 777888; // Magic Number
input int      InpBrokerGMTOffset      = 2;      // Broker Server GMT Offset (hours)

//+------------------------------------------------------------------+
//| ENUMERATIONS                                                      |
//+------------------------------------------------------------------+
enum ENUM_BIAS
{
   BIAS_BULLISH,
   BIAS_BEARISH,
   BIAS_NEUTRAL,
   BIAS_NO_TRADE
};

enum ENUM_LIQUIDITY_TYPE
{
   LIQ_BUY_SIDE,
   LIQ_SELL_SIDE
};

enum ENUM_LIQUIDITY_SOURCE
{
   SRC_EQUAL_HIGHS,
   SRC_EQUAL_LOWS,
   SRC_SESSION_HIGH,
   SRC_SESSION_LOW,
   SRC_PREV_DAY_HIGH,
   SRC_PREV_DAY_LOW,
   SRC_SWING_HIGH,
   SRC_SWING_LOW
};

enum ENUM_DIRECTION
{
   DIR_BULLISH,
   DIR_BEARISH
};

enum ENUM_TRADE_STATUS
{
   STATUS_PENDING,
   STATUS_ACTIVE,
   STATUS_TP1_HIT,
   STATUS_COMPLETED,
   STATUS_STOPPED,
   STATUS_CANCELLED
};

//+------------------------------------------------------------------+
//| DATA STRUCTURES                                                   |
//+------------------------------------------------------------------+
struct SwingPoint
{
   double   price;
   datetime time;
   string   type;  // "SWING_HIGH" or "SWING_LOW"
};

struct LiquidityLevel
{
   double               price;
   ENUM_LIQUIDITY_TYPE  type;
   ENUM_LIQUIDITY_SOURCE source;
   datetime             timeCreated;
   bool                 isSwept;
   datetime             sweepTime;
};

struct FairValueGap
{
   double         upperBoundary;
   double         lowerBoundary;
   double         midpoint;
   ENUM_DIRECTION direction;
   double         size;
   datetime       timeCreated;
   bool           isMitigated;
};

struct OrderBlock
{
   double         upperBoundary;
   double         lowerBoundary;
   ENUM_DIRECTION direction;
   datetime       timeCreated;
   bool           isMitigated;
};

struct OTEZone
{
   double         upperBoundary;
   double         lowerBoundary;
   double         fibSwingHigh;
   double         fibSwingLow;
   ENUM_DIRECTION direction;
};

struct BiasInfo
{
   ENUM_BIAS dailyDirection;
   ENUM_BIAS fourHourDirection;
   ENUM_BIAS combinedBias;
   string    priceZone;  // "PREMIUM" or "DISCOUNT"
};

struct SweepData
{
   bool     swept;
   double   levelPrice;
   string   levelSource;
   datetime sweepCandleTime;
   double   sweepExtremePrice;
};

struct MSSData
{
   bool     shifted;
   double   mssLevel;
   datetime mssTime;
   double   mssCandleHigh;
   double   mssCandleLow;
};

struct EntrySetup
{
   bool        found;
   FairValueGap fvg;
   OrderBlock   ob;
   OTEZone      ote;
   double       entryPrice;
   double       alternativeEntry;
   string       reason;
};

struct TradeIdea
{
   string             symbol;
   string             direction; // "LONG" or "SHORT"
   double             entryPrice;
   double             stopLoss;
   double             tp1Price;
   double             tp2Price;
   double             totalLotSize;
   double             tp1LotSize;
   double             tp2LotSize;
   double             riskAmountUSD;
   ulong              tp1Ticket;
   ulong              tp2Ticket;
   ENUM_TRADE_STATUS  status;
   bool               slMovedToBE;
   datetime           openTime;
};

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
CTrade         g_trade;
CPositionInfo  g_position;
COrderInfo     g_order;

// State variables
TradeIdea      g_activeIdeas[];
int            g_todayTradeCount;
int            g_todayLossCount;
double         g_todayRiskUsed;
BiasInfo       g_bias;
LiquidityLevel g_buySideLiq[];
LiquidityLevel g_sellSideLiq[];
bool           g_sweepDetected;
SweepData      g_sweepData;
bool           g_mssDetected;
MSSData        g_mssData;
bool           g_setupFound;
bool           g_dailyPrepDone;
int            g_lastDay;
datetime       g_lastBarTime;
string         g_tradingSymbol;

// EST offset: EST = GMT - 5
#define EST_GMT_OFFSET (-5)

//+------------------------------------------------------------------+
//| EXPERT INITIALIZATION                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // Validate symbol
   g_tradingSymbol = InpSymbol;
   if(!SymbolSelect(g_tradingSymbol, true))
   {
      Print("ERROR: Symbol ", g_tradingSymbol, " not found or cannot be selected!");
      return(INIT_FAILED);
   }

   // Setup trade object
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviations(10);
   g_trade.SetTypeFilling(ORDER_FILLING_FOK);
   g_trade.SetTypeFillingBySymbol(g_tradingSymbol);

   // Initialize state
   ResetDailyState();
   g_lastDay = -1;
   g_lastBarTime = 0;

   Print("==============================================");
   Print("  VENOM MODEL EA — INITIALIZED");
   Print("  Symbol: ", g_tradingSymbol);
   Print("  Risk: ", DoubleToString(InpRiskPercent, 2), "% per trade idea");
   Print("  TP1 RRR: 1:", DoubleToString(InpTP1RRR, 1));
   Print("  TP2 RRR: 1:", DoubleToString(InpTP2RRR, 1));
   Print("  Magic: ", IntegerToString(InpMagicNumber));
   Print("  Broker GMT Offset: ", IntegerToString(InpBrokerGMTOffset));
   Print("==============================================");

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| EXPERT DEINITIALIZATION                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("VENOM MODEL EA — DEINITIALIZED. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| EXPERT TICK FUNCTION                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only process on new bar close of entry timeframe
   if(!IsNewBar(g_tradingSymbol, InpEntryTF))
      return;

   datetime currentTime = TimeCurrent();
   MqlDateTime dtStruct;
   TimeToStruct(currentTime, dtStruct);

   //=== PHASE 0: DAILY RESET ===
   if(dtStruct.day != g_lastDay)
   {
      ResetDailyState();
      g_lastDay = dtStruct.day;
      Print("[NEW DAY] State reset — ", TimeToString(currentTime, TIME_DATE));
   }

   //=== PHASE 1: PRE-SESSION PREPARATION ===
   if(!g_dailyPrepDone)
   {
      // Check market open (skip weekends)
      if(dtStruct.day_of_week == 0 || dtStruct.day_of_week == 6)
      {
         return; // Saturday or Sunday
      }

      // Determine HTF Bias
      g_bias = DetermineHTFBias();
      Print("[BIAS] Daily: ", BiasToString(g_bias.dailyDirection),
            " | 4H: ", BiasToString(g_bias.fourHourDirection),
            " | Combined: ", BiasToString(g_bias.combinedBias),
            " | Zone: ", g_bias.priceZone);

      if(g_bias.combinedBias == BIAS_NO_TRADE)
      {
         Print("[SKIP] Conflicting bias — no trades today");
         g_dailyPrepDone = true; // Don't keep re-running
         return;
      }

      // Map Liquidity
      MapLiquidityLevels();
      Print("[LIQUIDITY] Buy-side: ", ArraySize(g_buySideLiq),
            " | Sell-side: ", ArraySize(g_sellSideLiq));

      g_dailyPrepDone = true;
      Print("[PREP] Daily preparation complete");
   }

   //=== PHASE 2: MANAGE EXISTING TRADES ===
   ManageActiveIdeas();

   //=== PHASE 3: NEW TRADE DETECTION ===
   // Gate: must be in a kill zone
   if(!IsInKillZone(currentTime))
      return;

   // Gate: risk limits
   if(!IsWithinRiskLimits())
      return;

   // Gate: bias must be valid
   if(g_bias.combinedBias == BIAS_NO_TRADE || g_bias.combinedBias == BIAS_NEUTRAL)
      return;

   // Gate: no more than one pending/active idea at a time
   if(HasActiveIdea())
      return;

   // Get entry timeframe candles
   MqlRates entryRates[];
   ArraySetAsSeries(entryRates, true);
   int copied = CopyRates(g_tradingSymbol, InpEntryTF, 0, 100, entryRates);
   if(copied < 20)
      return;

   //--- STEP A: Detect Liquidity Sweep ---
   if(!g_sweepDetected)
   {
      g_sweepData = DetectLiquiditySweep(entryRates, copied);
      if(g_sweepData.swept)
      {
         g_sweepDetected = true;
         Print("[SWEEP] Liquidity swept at ", DoubleToString(g_sweepData.levelPrice, _Digits),
               " | Source: ", g_sweepData.levelSource,
               " | Extreme: ", DoubleToString(g_sweepData.sweepExtremePrice, _Digits));
      }
      else
         return;
   }

   //--- STEP B: Detect Market Structure Shift ---
   if(g_sweepDetected && !g_mssDetected)
   {
      g_mssData = DetectMSS(entryRates, copied);
      if(g_mssData.shifted)
      {
         g_mssDetected = true;
         Print("[MSS] Confirmed at level ", DoubleToString(g_mssData.mssLevel, _Digits),
               " | Time: ", TimeToString(g_mssData.mssTime));
      }
      else
      {
         // Check if too many candles since sweep
         int candlesSince = CountCandlesSince(entryRates, copied, g_sweepData.sweepCandleTime);
         if(candlesSince > InpMaxCandlesAfterSweep)
         {
            Print("[INVALID] Too many candles since sweep without MSS — resetting");
            g_sweepDetected = false;
         }
         return;
      }
   }

   //--- STEP C: Find Entry Setup ---
   if(g_sweepDetected && g_mssDetected && !g_setupFound)
   {
      EntrySetup setup = FindEntrySetup(entryRates, copied);
      if(setup.found)
      {
         Print("[SETUP] Triple confluence found!");
         Print("  FVG: ", DoubleToString(setup.fvg.lowerBoundary, _Digits),
               " - ", DoubleToString(setup.fvg.upperBoundary, _Digits));
         Print("  OB: ", DoubleToString(setup.ob.lowerBoundary, _Digits),
               " - ", DoubleToString(setup.ob.upperBoundary, _Digits));
         Print("  OTE: ", DoubleToString(setup.ote.lowerBoundary, _Digits),
               " - ", DoubleToString(setup.ote.upperBoundary, _Digits));

         //--- STEP D: Construct & Execute Trade ---
         TradeIdea idea;
         if(ConstructTrade(setup, idea))
         {
            Print("[TRADE] Direction: ", idea.direction,
                  " | Entry: ", DoubleToString(idea.entryPrice, _Digits),
                  " | SL: ", DoubleToString(idea.stopLoss, _Digits),
                  " | TP1(1:2): ", DoubleToString(idea.tp1Price, _Digits),
                  " | TP2(1:3): ", DoubleToString(idea.tp2Price, _Digits),
                  " | Lots: ", DoubleToString(idea.totalLotSize, 2));

            if(ExecuteTrade(idea))
            {
               int idx = ArraySize(g_activeIdeas);
               ArrayResize(g_activeIdeas, idx + 1);
               g_activeIdeas[idx] = idea;
               g_todayTradeCount++;
               g_todayRiskUsed += idea.riskAmountUSD;
               g_setupFound = true;
               Print("[EXECUTED] Trade idea placed successfully");
            }
            else
            {
               Print("[ERROR] Trade execution failed");
            }
         }
         else
         {
            Print("[SKIP] Trade construction invalid: ", setup.reason);
         }
      }
      else
      {
         // Don't reset — keep checking next candles
      }
   }
}

//+------------------------------------------------------------------+
//| HELPER: Check for new bar                                         |
//+------------------------------------------------------------------+
bool IsNewBar(string symbol, ENUM_TIMEFRAMES tf)
{
   datetime barTime[];
   ArraySetAsSeries(barTime, true);
   if(CopyTime(symbol, tf, 0, 1, barTime) < 1)
      return false;

   if(barTime[0] == g_lastBarTime)
      return false;

   g_lastBarTime = barTime[0];
   return true;
}

//+------------------------------------------------------------------+
//| HELPER: Reset daily state                                         |
//+------------------------------------------------------------------+
void ResetDailyState()
{
   g_todayTradeCount  = 0;
   g_todayLossCount   = 0;
   g_todayRiskUsed    = 0.0;
   g_sweepDetected    = false;
   g_mssDetected      = false;
   g_setupFound       = false;
   g_dailyPrepDone    = false;
   ArrayFree(g_activeIdeas);
   ZeroMemory(g_sweepData);
   ZeroMemory(g_mssData);
   ZeroMemory(g_bias);
   ArrayFree(g_buySideLiq);
   ArrayFree(g_sellSideLiq);
}

//+------------------------------------------------------------------+
//| HELPER: Convert broker server time to EST                         |
//+------------------------------------------------------------------+
datetime BrokerTimeToEST(datetime brokerTime)
{
   // EST = GMT - 5
   // BrokerTime = GMT + BrokerOffset
   // So EST = BrokerTime - BrokerOffset - 5
   int totalShiftSeconds = (InpBrokerGMTOffset - EST_GMT_OFFSET) * 3600;
   return brokerTime - totalShiftSeconds;
}

//+------------------------------------------------------------------+
//| HELPER: Get current EST hour and minute                           |
//+------------------------------------------------------------------+
void GetESTTime(int &estHour, int &estMin)
{
   datetime estTime = BrokerTimeToEST(TimeCurrent());
   MqlDateTime dt;
   TimeToStruct(estTime, dt);
   estHour = dt.hour;
   estMin  = dt.min;
}

//+------------------------------------------------------------------+
//| HELPER: Check if time is within a range (handles midnight wrap)   |
//+------------------------------------------------------------------+
bool IsTimeInRange(int curHour, int curMin, int startHour, int startMin, int endHour, int endMin)
{
   int curTotal   = curHour * 60 + curMin;
   int startTotal = startHour * 60 + startMin;
   int endTotal   = endHour * 60 + endMin;

   if(startTotal <= endTotal)
      return (curTotal >= startTotal && curTotal < endTotal);
   else
      return (curTotal >= startTotal || curTotal < endTotal);
}

//+------------------------------------------------------------------+
//| HELPER: Check if currently in any kill zone                       |
//+------------------------------------------------------------------+
bool IsInKillZone(datetime currentTime)
{
   int estH, estM;
   GetESTTime(estH, estM);

   // London kill zone
   if(IsTimeInRange(estH, estM, InpLondonStartHour, InpLondonStartMin, InpLondonEndHour, InpLondonEndMin))
      return true;

   // NY kill zone
   if(IsTimeInRange(estH, estM, InpNYStartHour, InpNYStartMin, InpNYEndHour, InpNYEndMin))
      return true;

   return false;
}

//+------------------------------------------------------------------+
//| HELPER: Check risk limits                                         |
//+------------------------------------------------------------------+
bool IsWithinRiskLimits()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance <= 0) return false;

   if(g_todayTradeCount >= InpMaxTradesPerDay)
   {
      return false;
   }

   if(g_todayLossCount >= InpMaxConsecLossesDaily)
   {
      return false;
   }

   double dailyRiskPct = (g_todayRiskUsed / balance) * 100.0;
   if(dailyRiskPct >= InpMaxDailyRiskPercent)
   {
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| HELPER: Check if there is an active trade idea                    |
//+------------------------------------------------------------------+
bool HasActiveIdea()
{
   for(int i = 0; i < ArraySize(g_activeIdeas); i++)
   {
      if(g_activeIdeas[i].status == STATUS_PENDING ||
         g_activeIdeas[i].status == STATUS_ACTIVE ||
         g_activeIdeas[i].status == STATUS_TP1_HIT)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| HELPER: Bias to string                                            |
//+------------------------------------------------------------------+
string BiasToString(ENUM_BIAS b)
{
   switch(b)
   {
      case BIAS_BULLISH:  return "BULLISH";
      case BIAS_BEARISH:  return "BEARISH";
      case BIAS_NEUTRAL:  return "NEUTRAL";
      case BIAS_NO_TRADE: return "NO_TRADE";
   }
   return "UNKNOWN";
}

//+------------------------------------------------------------------+
//| HELPER: Normalize price                                           |
//+------------------------------------------------------------------+
double NormPrice(double price)
{
   int digits = (int)SymbolInfoInteger(g_tradingSymbol, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
}

//+------------------------------------------------------------------+
//| HELPER: Normalize lot size                                        |
//+------------------------------------------------------------------+
double NormLots(double lots)
{
   double minLot  = SymbolInfoDouble(g_tradingSymbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(g_tradingSymbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(g_tradingSymbol, SYMBOL_VOLUME_STEP);

   if(lotStep <= 0) lotStep = 0.01;

   lots = MathFloor(lots / lotStep) * lotStep;
   if(lots < minLot) lots = 0;
   if(lots > maxLot) lots = maxLot;

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| HELPER: Count candles since a given time                          |
//+------------------------------------------------------------------+
int CountCandlesSince(const MqlRates &rates[], int count, datetime sinceTime)
{
   int cnt = 0;
   for(int i = 0; i < count; i++)
   {
      if(rates[i].time > sinceTime)
         cnt++;
   }
   return cnt;
}

//+------------------------------------------------------------------+
//| CORE: Detect swing points from rate data                          |
//+------------------------------------------------------------------+
int DetectSwingPoints(const MqlRates &rates[], int count, int strength, SwingPoint &swings[])
{
   ArrayFree(swings);

   for(int i = strength; i < count - strength; i++)
   {
      // Check Swing High
      bool isSH = true;
      for(int j = 1; j <= strength; j++)
      {
         if(rates[i - j].high >= rates[i].high || rates[i + j].high >= rates[i].high)
         {
            isSH = false;
            break;
         }
      }
      if(isSH)
      {
         int idx = ArraySize(swings);
         ArrayResize(swings, idx + 1);
         swings[idx].price = rates[i].high;
         swings[idx].time  = rates[i].time;
         swings[idx].type  = "SWING_HIGH";
      }

      // Check Swing Low
      bool isSL = true;
      for(int j = 1; j <= strength; j++)
      {
         if(rates[i - j].low <= rates[i].low || rates[i + j].low <= rates[i].low)
         {
            isSL = false;
            break;
         }
      }
      if(isSL)
      {
         int idx = ArraySize(swings);
         ArrayResize(swings, idx + 1);
         swings[idx].price = rates[i].low;
         swings[idx].time  = rates[i].time;
         swings[idx].type  = "SWING_LOW";
      }
   }
   return ArraySize(swings);
}

//+------------------------------------------------------------------+
//| CORE: Determine HTF Bias                                          |
//+------------------------------------------------------------------+
BiasInfo DetermineHTFBias()
{
   BiasInfo bias;
   bias.combinedBias = BIAS_NO_TRADE;
   bias.priceZone    = "NEUTRAL";

   double currentPrice = SymbolInfoDouble(g_tradingSymbol, SYMBOL_BID);

   //--- Daily Analysis ---
   MqlRates dailyRates[];
   ArraySetAsSeries(dailyRates, true);
   int dailyCopied = CopyRates(g_tradingSymbol, InpHTF1, 0, InpSwingLookbackHTF, dailyRates);
   if(dailyCopied < 10)
   {
      Print("[BIAS] Insufficient daily data");
      return bias;
   }

   // Reverse to chronological for swing detection
   MqlRates dailyChrono[];
   ArrayResize(dailyChrono, dailyCopied);
   for(int i = 0; i < dailyCopied; i++)
      dailyChrono[i] = dailyRates[dailyCopied - 1 - i];

   SwingPoint dailySwings[];
   DetectSwingPoints(dailyChrono, dailyCopied, InpSwingStrength, dailySwings);

   // Find most recent swing high and low
   double recentSH = 0, recentSL = 0;
   bool foundSH = false, foundSL = false;

   for(int i = ArraySize(dailySwings) - 1; i >= 0; i--)
   {
      if(!foundSH && dailySwings[i].type == "SWING_HIGH")
      {
         recentSH = dailySwings[i].price;
         foundSH = true;
      }
      if(!foundSL && dailySwings[i].type == "SWING_LOW")
      {
         recentSL = dailySwings[i].price;
         foundSL = true;
      }
      if(foundSH && foundSL) break;
   }

   if(!foundSH || !foundSL || recentSH <= recentSL)
   {
      Print("[BIAS] Cannot determine dealing range");
      return bias;
   }

   double equilibrium = (recentSH + recentSL) / 2.0;

   if(currentPrice < equilibrium)
   {
      bias.dailyDirection = BIAS_BULLISH;
      bias.priceZone = "DISCOUNT";
   }
   else if(currentPrice > equilibrium)
   {
      bias.dailyDirection = BIAS_BEARISH;
      bias.priceZone = "PREMIUM";
   }
   else
   {
      bias.dailyDirection = BIAS_NEUTRAL;
      bias.priceZone = "EQUILIBRIUM";
   }

   //--- 4H Analysis ---
   MqlRates h4Rates[];
   ArraySetAsSeries(h4Rates, true);
   int h4Copied = CopyRates(g_tradingSymbol, InpHTF2, 0, InpSwingLookbackHTF, h4Rates);
   if(h4Copied < 10)
   {
      Print("[BIAS] Insufficient 4H data");
      return bias;
   }

   MqlRates h4Chrono[];
   ArrayResize(h4Chrono, h4Copied);
   for(int i = 0; i < h4Copied; i++)
      h4Chrono[i] = h4Rates[h4Copied - 1 - i];

   SwingPoint h4Swings[];
   DetectSwingPoints(h4Chrono, h4Copied, InpSwingStrength, h4Swings);

   // Get last 3 swing lows for bullish check
   double lastSLs[];
   ArrayResize(lastSLs, 0);
   for(int i = ArraySize(h4Swings) - 1; i >= 0 && ArraySize(lastSLs) < 3; i--)
   {
      if(h4Swings[i].type == "SWING_LOW")
      {
         int idx = ArraySize(lastSLs);
         ArrayResize(lastSLs, idx + 1);
         lastSLs[idx] = h4Swings[i].price;
      }
   }

   // Get last 3 swing highs for bearish check
   double lastSHs[];
   ArrayResize(lastSHs, 0);
   for(int i = ArraySize(h4Swings) - 1; i >= 0 && ArraySize(lastSHs) < 3; i--)
   {
      if(h4Swings[i].type == "SWING_HIGH")
      {
         int idx = ArraySize(lastSHs);
         ArrayResize(lastSHs, idx + 1);
         lastSHs[idx] = h4Swings[i].price;
      }
   }

   // Check bullish (higher lows): lastSLs[0] > lastSLs[1] > lastSLs[2]
   bool h4Bullish = false;
   if(ArraySize(lastSLs) >= 3)
   {
      if(lastSLs[0] > lastSLs[1] && lastSLs[1] > lastSLs[2])
         h4Bullish = true;
   }

   // Check bearish (lower highs): lastSHs[0] < lastSHs[1] < lastSHs[2]
   bool h4Bearish = false;
   if(ArraySize(lastSHs) >= 3)
   {
      if(lastSHs[0] < lastSHs[1] && lastSHs[1] < lastSHs[2])
         h4Bearish = true;
   }

   if(h4Bullish)
      bias.fourHourDirection = BIAS_BULLISH;
   else if(h4Bearish)
      bias.fourHourDirection = BIAS_BEARISH;
   else
      bias.fourHourDirection = BIAS_NEUTRAL;

   //--- Combine ---
   if(bias.dailyDirection == BIAS_BULLISH && bias.fourHourDirection == BIAS_BULLISH)
      bias.combinedBias = BIAS_BULLISH;
   else if(bias.dailyDirection == BIAS_BEARISH && bias.fourHourDirection == BIAS_BEARISH)
      bias.combinedBias = BIAS_BEARISH;
   else
      bias.combinedBias = BIAS_NO_TRADE;

   return bias;
}

//+------------------------------------------------------------------+
//| CORE: Get session range (high/low between two EST times)          |
//+------------------------------------------------------------------+
void GetSessionRange(int startHour, int startMin, int endHour, int endMin,
                     double &sessionHigh, double &sessionLow)
{
   sessionHigh = -DBL_MAX;
   sessionLow  = DBL_MAX;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   // Get last 200 15-min bars (~50 hours) to cover Asian session from previous day
   int copied = CopyRates(g_tradingSymbol, InpStructureTF, 0, 200, rates);
   if(copied < 5) return;

   for(int i = 0; i < copied; i++)
   {
      datetime estTime = BrokerTimeToEST(rates[i].time);
      MqlDateTime dt;
      TimeToStruct(estTime, dt);
      int h = dt.hour;
      int m = dt.min;

      if(IsTimeInRange(h, m, startHour, startMin, endHour, endMin))
      {
         if(rates[i].high > sessionHigh) sessionHigh = rates[i].high;
         if(rates[i].low < sessionLow)   sessionLow  = rates[i].low;
      }
   }
}

//+------------------------------------------------------------------+
//| CORE: Map all liquidity levels                                    |
//+------------------------------------------------------------------+
void MapLiquidityLevels()
{
   ArrayFree(g_buySideLiq);
   ArrayFree(g_sellSideLiq);

   double currentPrice = SymbolInfoDouble(g_tradingSymbol, SYMBOL_BID);

   //--- Session ranges ---
   double asianHigh, asianLow;
   GetSessionRange(InpAsianStartHour, InpAsianStartMin, InpAsianEndHour, InpAsianEndMin, asianHigh, asianLow);

   if(asianHigh > -DBL_MAX)
      AddLiquidityLevel(asianHigh, LIQ_BUY_SIDE, SRC_SESSION_HIGH);
   if(asianLow < DBL_MAX)
      AddLiquidityLevel(asianLow, LIQ_SELL_SIDE, SRC_SESSION_LOW);

   //--- Previous day high/low ---
   MqlRates dailyRates[];
   ArraySetAsSeries(dailyRates, true);
   if(CopyRates(g_tradingSymbol, PERIOD_D1, 1, 1, dailyRates) >= 1)
   {
      AddLiquidityLevel(dailyRates[0].high, LIQ_BUY_SIDE, SRC_PREV_DAY_HIGH);
      AddLiquidityLevel(dailyRates[0].low, LIQ_SELL_SIDE, SRC_PREV_DAY_LOW);
   }

   //--- Structure timeframe swings ---
   MqlRates structRates[];
   ArraySetAsSeries(structRates, true);
   int structCopied = CopyRates(g_tradingSymbol, InpStructureTF, 0, InpSwingLookbackStruct, structRates);
   if(structCopied > 10)
   {
      MqlRates structChrono[];
      ArrayResize(structChrono, structCopied);
      for(int i = 0; i < structCopied; i++)
         structChrono[i] = structRates[structCopied - 1 - i];

      SwingPoint structSwings[];
      DetectSwingPoints(structChrono, structCopied, InpSwingStrength, structSwings);

      for(int i = 0; i < ArraySize(structSwings); i++)
      {
         if(structSwings[i].type == "SWING_HIGH")
            AddLiquidityLevel(structSwings[i].price, LIQ_BUY_SIDE, SRC_SWING_HIGH);
         else
            AddLiquidityLevel(structSwings[i].price, LIQ_SELL_SIDE, SRC_SWING_LOW);
      }

      // Equal highs and lows
      DetectEqualLevels(structSwings);
   }

   // Sort: buy-side ascending (nearest above first), sell-side descending (nearest below first)
   SortLiquidityBuySide(currentPrice);
   SortLiquiditySellSide(currentPrice);
}

//+------------------------------------------------------------------+
//| HELPER: Add a liquidity level                                     |
//+------------------------------------------------------------------+
void AddLiquidityLevel(double price, ENUM_LIQUIDITY_TYPE type, ENUM_LIQUIDITY_SOURCE source)
{
   LiquidityLevel lv;
   lv.price       = price;
   lv.type        = type;
   lv.source      = source;
   lv.timeCreated = TimeCurrent();
   lv.isSwept     = false;
   lv.sweepTime   = 0;

   if(type == LIQ_BUY_SIDE)
   {
      int idx = ArraySize(g_buySideLiq);
      ArrayResize(g_buySideLiq, idx + 1);
      g_buySideLiq[idx] = lv;
   }
   else
   {
      int idx = ArraySize(g_sellSideLiq);
      ArrayResize(g_sellSideLiq, idx + 1);
      g_sellSideLiq[idx] = lv;
   }
}

//+------------------------------------------------------------------+
//| HELPER: Detect equal highs and lows                               |
//+------------------------------------------------------------------+
void DetectEqualLevels(const SwingPoint &swings[])
{
   int total = ArraySize(swings);

   // Equal highs
   for(int i = 0; i < total - 1; i++)
   {
      if(swings[i].type != "SWING_HIGH") continue;
      for(int j = i + 1; j < total; j++)
      {
         if(swings[j].type != "SWING_HIGH") continue;
         if(MathAbs(swings[i].price - swings[j].price) <= InpEqualLevelTolerance)
         {
            double avgP = (swings[i].price + swings[j].price) / 2.0;
            if(!LiquidityLevelExists(avgP, LIQ_BUY_SIDE))
               AddLiquidityLevel(avgP, LIQ_BUY_SIDE, SRC_EQUAL_HIGHS);
         }
      }
   }

   // Equal lows
   for(int i = 0; i < total - 1; i++)
   {
      if(swings[i].type != "SWING_LOW") continue;
      for(int j = i + 1; j < total; j++)
      {
         if(swings[j].type != "SWING_LOW") continue;
         if(MathAbs(swings[i].price - swings[j].price) <= InpEqualLevelTolerance)
         {
            double avgP = (swings[i].price + swings[j].price) / 2.0;
            if(!LiquidityLevelExists(avgP, LIQ_SELL_SIDE))
               AddLiquidityLevel(avgP, LIQ_SELL_SIDE, SRC_EQUAL_LOWS);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| HELPER: Check if liquidity level already exists                   |
//+------------------------------------------------------------------+
bool LiquidityLevelExists(double price, ENUM_LIQUIDITY_TYPE type)
{
   if(type == LIQ_BUY_SIDE)
   {
      for(int i = 0; i < ArraySize(g_buySideLiq); i++)
         if(MathAbs(g_buySideLiq[i].price - price) <= InpEqualLevelTolerance)
            return true;
   }
   else
   {
      for(int i = 0; i < ArraySize(g_sellSideLiq); i++)
         if(MathAbs(g_sellSideLiq[i].price - price) <= InpEqualLevelTolerance)
            return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| HELPER: Sort buy-side liquidity ascending above current price     |
//+------------------------------------------------------------------+
void SortLiquidityBuySide(double currentPrice)
{
   // Remove levels below current price
   LiquidityLevel temp[];
   for(int i = 0; i < ArraySize(g_buySideLiq); i++)
   {
      if(g_buySideLiq[i].price > currentPrice)
      {
         int idx = ArraySize(temp);
         ArrayResize(temp, idx + 1);
         temp[idx] = g_buySideLiq[i];
      }
   }

   // Bubble sort ascending
   for(int i = 0; i < ArraySize(temp) - 1; i++)
   {
      for(int j = 0; j < ArraySize(temp) - i - 1; j++)
      {
         if(temp[j].price > temp[j+1].price)
         {
            LiquidityLevel swap = temp[j];
            temp[j] = temp[j+1];
            temp[j+1] = swap;
         }
      }
   }

   ArrayFree(g_buySideLiq);
   ArrayResize(g_buySideLiq, ArraySize(temp));
   for(int i = 0; i < ArraySize(temp); i++)
      g_buySideLiq[i] = temp[i];
}

//+------------------------------------------------------------------+
//| HELPER: Sort sell-side liquidity descending below current price   |
//+------------------------------------------------------------------+
void SortLiquiditySellSide(double currentPrice)
{
   LiquidityLevel temp[];
   for(int i = 0; i < ArraySize(g_sellSideLiq); i++)
   {
      if(g_sellSideLiq[i].price < currentPrice)
      {
         int idx = ArraySize(temp);
         ArrayResize(temp, idx + 1);
         temp[idx] = g_sellSideLiq[i];
      }
   }

   // Bubble sort descending
   for(int i = 0; i < ArraySize(temp) - 1; i++)
   {
      for(int j = 0; j < ArraySize(temp) - i - 1; j++)
      {
         if(temp[j].price < temp[j+1].price)
         {
            LiquidityLevel swap = temp[j];
            temp[j] = temp[j+1];
            temp[j+1] = swap;
         }
      }
   }

   ArrayFree(g_sellSideLiq);
   ArrayResize(g_sellSideLiq, ArraySize(temp));
   for(int i = 0; i < ArraySize(temp); i++)
      g_sellSideLiq[i] = temp[i];
}

//+------------------------------------------------------------------+
//| CORE: Detect Liquidity Sweep                                      |
//+------------------------------------------------------------------+
SweepData DetectLiquiditySweep(const MqlRates &rates[], int count)
{
   SweepData result;
   result.swept = false;

   if(count < 3) return result;

   // rates[] is series: [0]=latest, [1]=previous, etc.
   double latestHigh  = rates[0].high;
   double latestLow   = rates[0].low;
   double latestClose = rates[0].close;
   double prevHigh    = rates[1].high;
   double prevLow     = rates[1].low;

   if(g_bias.combinedBias == BIAS_BULLISH)
   {
      // Look for sell-side sweep (price goes below level then closes above)
      for(int i = 0; i < ArraySize(g_sellSideLiq); i++)
      {
         if(g_sellSideLiq[i].isSwept) continue;
         double lvl = g_sellSideLiq[i].price;

         bool singleSweep = (latestLow < lvl && latestClose > lvl);
         bool twoBarSweep = (prevLow < lvl && latestClose > lvl);

         if(singleSweep || twoBarSweep)
         {
            g_sellSideLiq[i].isSwept   = true;
            g_sellSideLiq[i].sweepTime = rates[0].time;

            result.swept             = true;
            result.levelPrice        = lvl;
            result.levelSource       = LiqSourceToString(g_sellSideLiq[i].source);
            result.sweepCandleTime   = rates[0].time;
            result.sweepExtremePrice = latestLow;
            if(twoBarSweep && !singleSweep)
               result.sweepExtremePrice = MathMin(latestLow, prevLow);
            return result;
         }
      }
   }
   else if(g_bias.combinedBias == BIAS_BEARISH)
   {
      // Look for buy-side sweep
      for(int i = 0; i < ArraySize(g_buySideLiq); i++)
      {
         if(g_buySideLiq[i].isSwept) continue;
         double lvl = g_buySideLiq[i].price;

         bool singleSweep = (latestHigh > lvl && latestClose < lvl);
         bool twoBarSweep = (prevHigh > lvl && latestClose < lvl);

         if(singleSweep || twoBarSweep)
         {
            g_buySideLiq[i].isSwept   = true;
            g_buySideLiq[i].sweepTime = rates[0].time;

            result.swept             = true;
            result.levelPrice        = lvl;
            result.levelSource       = LiqSourceToString(g_buySideLiq[i].source);
            result.sweepCandleTime   = rates[0].time;
            result.sweepExtremePrice = latestHigh;
            if(twoBarSweep && !singleSweep)
               result.sweepExtremePrice = MathMax(latestHigh, prevHigh);
            return result;
         }
      }
   }

   return result;
}

//+------------------------------------------------------------------+
//| HELPER: Liquidity source to string                                |
//+------------------------------------------------------------------+
string LiqSourceToString(ENUM_LIQUIDITY_SOURCE src)
{
   switch(src)
   {
      case SRC_EQUAL_HIGHS:   return "EQUAL_HIGHS";
      case SRC_EQUAL_LOWS:    return "EQUAL_LOWS";
      case SRC_SESSION_HIGH:  return "ASIAN_HIGH";
      case SRC_SESSION_LOW:   return "ASIAN_LOW";
      case SRC_PREV_DAY_HIGH: return "PREV_DAY_HIGH";
      case SRC_PREV_DAY_LOW:  return "PREV_DAY_LOW";
      case SRC_SWING_HIGH:    return "SWING_HIGH";
      case SRC_SWING_LOW:     return "SWING_LOW";
   }
   return "UNKNOWN";
}

//+------------------------------------------------------------------+
//| CORE: Detect Market Structure Shift                               |
//+------------------------------------------------------------------+
MSSData DetectMSS(const MqlRates &rates[], int count)
{
   MSSData result;
   result.shifted = false;

   if(count < 10) return result;

   // Build chronological array from sweep time onward
   MqlRates chrono[];
   int chronoSize = 0;

   // rates is series [0]=latest. We need chrono order.
   // First, find all bars and put in chrono order
   int totalBars = MathMin(count, 60); // Look at last 60 bars
   ArrayResize(chrono, totalBars);
   for(int i = 0; i < totalBars; i++)
      chrono[i] = rates[totalBars - 1 - i];
   chronoSize = totalBars;

   // Detect micro swings on chrono data
   SwingPoint microSwings[];
   DetectSwingPoints(chrono, chronoSize, InpSwingStrengthEntry, microSwings);

   if(g_bias.combinedBias == BIAS_BULLISH)
   {
      // Find most recent swing high BEFORE or AT sweep time
      double mssLevel = 0;
      bool foundLevel = false;

      for(int i = ArraySize(microSwings) - 1; i >= 0; i--)
      {
         if(microSwings[i].type == "SWING_HIGH" &&
            microSwings[i].time <= g_sweepData.sweepCandleTime)
         {
            mssLevel = microSwings[i].price;
            foundLevel = true;
            break;
         }
      }

      if(!foundLevel) return result;

      // Check if any candle AFTER sweep closed above this level
      for(int i = 0; i < chronoSize; i++)
      {
         if(chrono[i].time <= g_sweepData.sweepCandleTime) continue;
         if(chrono[i].close > mssLevel && chrono[i].close > chrono[i].open)
         {
            result.shifted       = true;
            result.mssLevel      = mssLevel;
            result.mssTime       = chrono[i].time;
            result.mssCandleHigh = chrono[i].high;
            result.mssCandleLow  = chrono[i].low;
            return result;
         }
      }
   }
   else if(g_bias.combinedBias == BIAS_BEARISH)
   {
      double mssLevel = 0;
      bool foundLevel = false;

      for(int i = ArraySize(microSwings) - 1; i >= 0; i--)
      {
         if(microSwings[i].type == "SWING_LOW" &&
            microSwings[i].time <= g_sweepData.sweepCandleTime)
         {
            mssLevel = microSwings[i].price;
            foundLevel = true;
            break;
         }
      }

      if(!foundLevel) return result;

      for(int i = 0; i < chronoSize; i++)
      {
         if(chrono[i].time <= g_sweepData.sweepCandleTime) continue;
         if(chrono[i].close < mssLevel && chrono[i].close < chrono[i].open)
         {
            result.shifted       = true;
            result.mssLevel      = mssLevel;
            result.mssTime       = chrono[i].time;
            result.mssCandleHigh = chrono[i].high;
            result.mssCandleLow  = chrono[i].low;
            return result;
         }
      }
   }

   return result;
}

//+------------------------------------------------------------------+
//| CORE: Detect Fair Value Gaps                                      |
//+------------------------------------------------------------------+
int DetectFVGs(const MqlRates &chrono[], int count, double minSize,
               ENUM_DIRECTION dir, FairValueGap &fvgs[])
{
   ArrayFree(fvgs);

   for(int i = 2; i < count; i++)
   {
      // Bullish FVG: candle[i-2].high < candle[i].low
      if(dir == DIR_BULLISH)
      {
         if(chrono[i-2].high < chrono[i].low)
         {
            double gapSize = chrono[i].low - chrono[i-2].high;
            if(gapSize >= minSize)
            {
               int idx = ArraySize(fvgs);
               ArrayResize(fvgs, idx + 1);
               fvgs[idx].upperBoundary = chrono[i].low;
               fvgs[idx].lowerBoundary = chrono[i-2].high;
               fvgs[idx].midpoint      = (chrono[i].low + chrono[i-2].high) / 2.0;
               fvgs[idx].direction     = DIR_BULLISH;
               fvgs[idx].size          = gapSize;
               fvgs[idx].timeCreated   = chrono[i-1].time;
               fvgs[idx].isMitigated   = false;
            }
         }
      }

      // Bearish FVG: candle[i-2].low > candle[i].high
      if(dir == DIR_BEARISH)
      {
         if(chrono[i-2].low > chrono[i].high)
         {
            double gapSize = chrono[i-2].low - chrono[i].high;
            if(gapSize >= minSize)
            {
               int idx = ArraySize(fvgs);
               ArrayResize(fvgs, idx + 1);
               fvgs[idx].upperBoundary = chrono[i-2].low;
               fvgs[idx].lowerBoundary = chrono[i].high;
               fvgs[idx].midpoint      = (chrono[i-2].low + chrono[i].high) / 2.0;
               fvgs[idx].direction     = DIR_BEARISH;
               fvgs[idx].size          = gapSize;
               fvgs[idx].timeCreated   = chrono[i-1].time;
               fvgs[idx].isMitigated   = false;
            }
         }
      }
   }

   return ArraySize(fvgs);
}

//+------------------------------------------------------------------+
//| CORE: Detect Order Blocks                                         |
//+------------------------------------------------------------------+
int DetectOrderBlocks(const MqlRates &chrono[], int count, int maxBarsBack,
                      ENUM_DIRECTION dir, OrderBlock &obs[])
{
   ArrayFree(obs);

   for(int i = 1; i < count; i++)
   {
      double curBodySize = MathAbs(chrono[i].close - chrono[i].open);
      bool curBullish = (chrono[i].close > chrono[i].open);
      bool curBearish = (chrono[i].close < chrono[i].open);

      // Calculate average body of previous 3 candles
      double avgBody = 0;
      int avgCount = 0;
      for(int k = 1; k <= 3; k++)
      {
         if(i - k >= 0)
         {
            avgBody += MathAbs(chrono[i-k].close - chrono[i-k].open);
            avgCount++;
         }
      }
      if(avgCount > 0) avgBody /= avgCount;
      if(avgBody <= 0) continue;

      // Bullish OB: current candle is strongly bullish, find last bearish candle before it
      if(dir == DIR_BULLISH && curBullish && curBodySize > avgBody * 1.5)
      {
         for(int j = 1; j <= maxBarsBack; j++)
         {
            if(i - j < 0) break;
            if(chrono[i-j].close < chrono[i-j].open) // bearish candle
            {
               int idx = ArraySize(obs);
               ArrayResize(obs, idx + 1);
               obs[idx].upperBoundary = chrono[i-j].open;
               obs[idx].lowerBoundary = chrono[i-j].low;
               obs[idx].direction     = DIR_BULLISH;
               obs[idx].timeCreated   = chrono[i-j].time;
               obs[idx].isMitigated   = false;
               break;
            }
         }
      }

      // Bearish OB: current candle is strongly bearish
      if(dir == DIR_BEARISH && curBearish && curBodySize > avgBody * 1.5)
      {
         for(int j = 1; j <= maxBarsBack; j++)
         {
            if(i - j < 0) break;
            if(chrono[i-j].close > chrono[i-j].open) // bullish candle
            {
               int idx = ArraySize(obs);
               ArrayResize(obs, idx + 1);
               obs[idx].upperBoundary = chrono[i-j].high;
               obs[idx].lowerBoundary = chrono[i-j].open;
               obs[idx].direction     = DIR_BEARISH;
               obs[idx].timeCreated   = chrono[i-j].time;
               obs[idx].isMitigated   = false;
               break;
            }
         }
      }
   }

   return ArraySize(obs);
}

//+------------------------------------------------------------------+
//| CORE: Calculate OTE Zone                                          |
//+------------------------------------------------------------------+
OTEZone CalculateOTEZone(double swingLow, double swingHigh, ENUM_DIRECTION dir)
{
   OTEZone ote;
   double range = swingHigh - swingLow;

   ote.fibSwingHigh = swingHigh;
   ote.fibSwingLow  = swingLow;
   ote.direction    = dir;

   if(dir == DIR_BULLISH)
   {
      ote.upperBoundary = swingHigh - (range * InpOTEFibUpper); // 62%
      ote.lowerBoundary = swingHigh - (range * InpOTEFibLower); // 79%
   }
   else
   {
      ote.lowerBoundary = swingLow + (range * InpOTEFibUpper);
      ote.upperBoundary = swingLow + (range * InpOTEFibLower);
   }

   return ote;
}

//+------------------------------------------------------------------+
//| HELPER: Check if two zones overlap                                |
//+------------------------------------------------------------------+
bool ZonesOverlap(double aUpper, double aLower, double bUpper, double bLower)
{
   return (aLower <= bUpper && bLower <= aUpper);
}

//+------------------------------------------------------------------+
//| CORE: Find Entry Setup (FVG + OB + OTE triple confluence)         |
//+------------------------------------------------------------------+
EntrySetup FindEntrySetup(const MqlRates &rates[], int count)
{
   EntrySetup setup;
   setup.found  = false;
   setup.reason = "";

   ENUM_DIRECTION dir = (g_bias.combinedBias == BIAS_BULLISH) ? DIR_BULLISH : DIR_BEARISH;

   // Build chrono array from sweep candle onward
   MqlRates chrono[];
   int chronoSize = 0;

   // rates is series [0]=latest. Convert relevant portion to chrono.
   for(int i = count - 1; i >= 0; i--)
   {
      if(rates[i].time >= g_sweepData.sweepCandleTime)
      {
         ArrayResize(chrono, chronoSize + 1);
         chrono[chronoSize] = rates[i];
         chronoSize++;
      }
   }

   if(chronoSize < 5)
   {
      setup.reason = "Insufficient candles after sweep";
      return setup;
   }

   // Detect FVGs
   FairValueGap fvgs[];
   DetectFVGs(chrono, chronoSize, InpFVGMinSize, dir, fvgs);
   if(ArraySize(fvgs) == 0)
   {
      setup.reason = "No valid FVG found after MSS";
      return setup;
   }

   // Detect Order Blocks
   OrderBlock obs[];
   DetectOrderBlocks(chrono, chronoSize, InpOBMaxBarsBack, dir, obs);
   if(ArraySize(obs) == 0)
   {
      setup.reason = "No valid Order Block found";
      return setup;
   }

   // Calculate OTE Zone
   double fibLow, fibHigh;
   if(dir == DIR_BULLISH)
   {
      fibLow  = g_sweepData.sweepExtremePrice;
      fibHigh = g_mssData.mssCandleHigh;
   }
   else
   {
      fibHigh = g_sweepData.sweepExtremePrice;
      fibLow  = g_mssData.mssCandleLow;
   }

   if(fibHigh <= fibLow)
   {
      setup.reason = "Invalid fib range";
      return setup;
   }

   OTEZone ote = CalculateOTEZone(fibLow, fibHigh, dir);

   // Find FVG + OB + OTE overlap
   for(int f = 0; f < ArraySize(fvgs); f++)
   {
      for(int o = 0; o < ArraySize(obs); o++)
      {
         // FVG overlaps OB?
         if(!ZonesOverlap(fvgs[f].upperBoundary, fvgs[f].lowerBoundary,
                          obs[o].upperBoundary, obs[o].lowerBoundary))
            continue;

         // FVG overlaps OTE?
         if(!ZonesOverlap(fvgs[f].upperBoundary, fvgs[f].lowerBoundary,
                          ote.upperBoundary, ote.lowerBoundary))
            continue;

         // Triple confluence found!
         setup.found = true;
         setup.fvg   = fvgs[f];
         setup.ob    = obs[o];
         setup.ote   = ote;

         if(dir == DIR_BULLISH)
         {
            setup.entryPrice      = fvgs[f].upperBoundary - (fvgs[f].size * InpFVGEntryLevel);
            setup.alternativeEntry = obs[o].upperBoundary;
         }
         else
         {
            setup.entryPrice      = fvgs[f].lowerBoundary + (fvgs[f].size * InpFVGEntryLevel);
            setup.alternativeEntry = obs[o].lowerBoundary;
         }

         return setup;
      }
   }

   setup.reason = "No FVG+OB+OTE triple confluence found";
   return setup;
}

//+------------------------------------------------------------------+
//| CORE: Construct Trade                                             |
//+------------------------------------------------------------------+
bool ConstructTrade(EntrySetup &setup, TradeIdea &idea)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance <= 0) return false;

   bool isBull = (g_bias.combinedBias == BIAS_BULLISH);
   idea.symbol = g_tradingSymbol;
   idea.direction = isBull ? "LONG" : "SHORT";

   // Entry price
   idea.entryPrice = NormPrice(setup.entryPrice);

   // Stop loss
   if(isBull)
      idea.stopLoss = NormPrice(g_sweepData.sweepExtremePrice - InpSLBuffer);
   else
      idea.stopLoss = NormPrice(g_sweepData.sweepExtremePrice + InpSLBuffer);

   double slDistPrimary = MathAbs(idea.entryPrice - idea.stopLoss);

   // Check alternative entry for tighter stop
   double altEntry = NormPrice(setup.alternativeEntry);
   double slDistAlt = MathAbs(altEntry - idea.stopLoss);

   if(slDistAlt < slDistPrimary && slDistAlt > 0)
   {
      idea.entryPrice = altEntry;
      slDistPrimary = slDistAlt;
   }

   if(slDistPrimary <= 0)
   {
      setup.reason = "SL distance is zero";
      return false;
   }

   // TP1 and TP2 using fixed RRR
   if(isBull)
   {
      idea.tp1Price = NormPrice(idea.entryPrice + (slDistPrimary * InpTP1RRR));
      idea.tp2Price = NormPrice(idea.entryPrice + (slDistPrimary * InpTP2RRR));
   }
   else
   {
      idea.tp1Price = NormPrice(idea.entryPrice - (slDistPrimary * InpTP1RRR));
      idea.tp2Price = NormPrice(idea.entryPrice - (slDistPrimary * InpTP2RRR));
   }

   // Validate RRR
   if(InpTP1RRR < InpMinRRR)
   {
      setup.reason = "TP1 RRR below minimum";
      return false;
   }

   // Position sizing
   idea.riskAmountUSD = balance * (InpRiskPercent / 100.0);
   double tickSize  = SymbolInfoDouble(g_tradingSymbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(g_tradingSymbol, SYMBOL_TRADE_TICK_VALUE);

   if(tickSize <= 0 || tickValue <= 0)
   {
      setup.reason = "Cannot get tick size/value";
      return false;
   }

   double slDistTicks = slDistPrimary / tickSize;
   double riskPerLot  = slDistTicks * tickValue;

   if(riskPerLot <= 0)
   {
      setup.reason = "Risk per lot is zero";
      return false;
   }

   idea.totalLotSize = NormLots(idea.riskAmountUSD / riskPerLot);
   if(idea.totalLotSize <= 0)
   {
      setup.reason = "Lot size too small";
      return false;
   }

   // Split into TP1 and TP2 portions
   idea.tp1LotSize = NormLots(idea.totalLotSize * (InpTP1ClosePercent / 100.0));
   idea.tp2LotSize = NormLots(idea.totalLotSize - idea.tp1LotSize);

   // If either portion is zero due to min lot, assign all to TP1
   if(idea.tp1LotSize <= 0)
   {
      idea.tp1LotSize = idea.totalLotSize;
      idea.tp2LotSize = 0;
   }
   if(idea.tp2LotSize <= 0 && idea.tp1LotSize < idea.totalLotSize)
   {
      idea.tp1LotSize = idea.totalLotSize;
      idea.tp2LotSize = 0;
   }

   idea.tp1Ticket   = 0;
   idea.tp2Ticket   = 0;
   idea.status      = STATUS_PENDING;
   idea.slMovedToBE = false;
   idea.openTime    = TimeCurrent();

   return true;
}

//+------------------------------------------------------------------+
//| CORE: Execute Trade (two separate positions)                      |
//+------------------------------------------------------------------+
bool ExecuteTrade(TradeIdea &idea)
{
   double currentPrice = SymbolInfoDouble(g_tradingSymbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(g_tradingSymbol, SYMBOL_ASK);

   bool isLong = (idea.direction == "LONG");

   //--- Position 1: TP1 portion ---
   bool tp1Success = false;
   if(idea.tp1LotSize > 0)
   {
      if(isLong)
      {
         if(idea.entryPrice < ask)
         {
            // Buy limit
            tp1Success = g_trade.BuyLimit(idea.tp1LotSize, idea.entryPrice, g_tradingSymbol,
                                           idea.stopLoss, idea.tp1Price,
                                           ORDER_TIME_DAY, 0, "Venom_TP1");
         }
         else
         {
            // Market buy
            tp1Success = g_trade.Buy(idea.tp1LotSize, g_tradingSymbol,
                                      0, idea.stopLoss, idea.tp1Price, "Venom_TP1");
         }
      }
      else
      {
         if(idea.entryPrice > currentPrice)
         {
            // Sell limit
            tp1Success = g_trade.SellLimit(idea.tp1LotSize, idea.entryPrice, g_tradingSymbol,
                                            idea.stopLoss, idea.tp1Price,
                                            ORDER_TIME_DAY, 0, "Venom_TP1");
         }
         else
         {
            // Market sell
            tp1Success = g_trade.Sell(idea.tp1LotSize, g_tradingSymbol,
                                       0, idea.stopLoss, idea.tp1Price, "Venom_TP1");
         }
      }

      if(tp1Success)
      {
         idea.tp1Ticket = g_trade.ResultOrder();
         if(idea.tp1Ticket == 0)
            idea.tp1Ticket = g_trade.ResultDeal();
         Print("[ORDER] TP1 position placed. Ticket: ", idea.tp1Ticket,
               " | Lots: ", DoubleToString(idea.tp1LotSize, 2));
      }
      else
      {
         Print("[ERROR] TP1 order failed: ", g_trade.ResultRetcodeDescription());
         return false;
      }
   }

   //--- Position 2: TP2 portion ---
   bool tp2Success = false;
   if(idea.tp2LotSize > 0)
   {
      if(isLong)
      {
         if(idea.entryPrice < ask)
         {
            tp2Success = g_trade.BuyLimit(idea.tp2LotSize, idea.entryPrice, g_tradingSymbol,
                                           idea.stopLoss, idea.tp2Price,
                                           ORDER_TIME_DAY, 0, "Venom_TP2");
         }
         else
         {
            tp2Success = g_trade.Buy(idea.tp2LotSize, g_tradingSymbol,
                                      0, idea.stopLoss, idea.tp2Price, "Venom_TP2");
         }
      }
      else
      {
         if(idea.entryPrice > currentPrice)
         {
            tp2Success = g_trade.SellLimit(idea.tp2LotSize, idea.entryPrice, g_tradingSymbol,
                                            idea.stopLoss, idea.tp2Price,
                                            ORDER_TIME_DAY, 0, "Venom_TP2");
         }
         else
         {
            tp2Success = g_trade.Sell(idea.tp2LotSize, g_tradingSymbol,
                                       0, idea.stopLoss, idea.tp2Price, "Venom_TP2");
         }
      }

      if(tp2Success)
      {
         idea.tp2Ticket = g_trade.ResultOrder();
         if(idea.tp2Ticket == 0)
            idea.tp2Ticket = g_trade.ResultDeal();
         Print("[ORDER] TP2 position placed. Ticket: ", idea.tp2Ticket,
               " | Lots: ", DoubleToString(idea.tp2LotSize, 2));
      }
      else
      {
         Print("[ERROR] TP2 order failed: ", g_trade.ResultRetcodeDescription());
         // TP1 was placed, so we still continue with TP1 only
         idea.tp2LotSize = 0;
         idea.tp2Ticket  = 0;
      }
   }

   idea.status = STATUS_PENDING;
   return true;
}

//+------------------------------------------------------------------+
//| CORE: Manage active trade ideas                                   |
//+------------------------------------------------------------------+
void ManageActiveIdeas()
{
   for(int i = ArraySize(g_activeIdeas) - 1; i >= 0; i--)
   {
      ManageSingleIdea(g_activeIdeas[i]);
   }
}

//+------------------------------------------------------------------+
//| CORE: Manage a single trade idea                                  |
//+------------------------------------------------------------------+
void ManageSingleIdea(TradeIdea &idea)
{
   if(idea.status == STATUS_COMPLETED || idea.status == STATUS_STOPPED || idea.status == STATUS_CANCELLED)
      return;

   //--- Check TP1 position status ---
   bool tp1Alive  = false;
   bool tp1Filled = false;
   bool tp1Exists = false;

   if(idea.tp1Ticket > 0)
   {
      // Check if it's a pending order
      if(OrderSelect(idea.tp1Ticket))
      {
         tp1Exists = true;
         tp1Alive  = true;

         // Cancel pending orders if kill zone ended and still not filled
         if(!IsInKillZone(TimeCurrent()) && idea.status == STATUS_PENDING)
         {
            g_trade.OrderDelete(idea.tp1Ticket);
            Print("[CANCEL] TP1 pending order cancelled — kill zone ended");
            tp1Alive = false;
         }
      }
      // Check if it's an open position
      else if(PositionSelectByTicket(idea.tp1Ticket))
      {
         tp1Filled = true;
         tp1Alive  = true;
      }
      else
      {
         // Neither pending nor position — it was closed (TP/SL hit or cancelled)
         tp1Alive = false;
      }
   }

   //--- Check TP2 position status ---
   bool tp2Alive  = false;
   bool tp2Filled = false;
   bool tp2Exists = false;

   if(idea.tp2Ticket > 0)
   {
      if(OrderSelect(idea.tp2Ticket))
      {
         tp2Exists = true;
         tp2Alive  = true;

         if(!IsInKillZone(TimeCurrent()) && idea.status == STATUS_PENDING)
         {
            g_trade.OrderDelete(idea.tp2Ticket);
            Print("[CANCEL] TP2 pending order cancelled — kill zone ended");
            tp2Alive = false;
         }
      }
      else if(PositionSelectByTicket(idea.tp2Ticket))
      {
         tp2Filled = true;
         tp2Alive  = true;
      }
      else
      {
         tp2Alive = false;
      }
   }

   //--- Update idea status ---
   // If both pending orders got cancelled
   if(!tp1Alive && !tp2Alive && idea.status == STATUS_PENDING)
   {
      idea.status = STATUS_CANCELLED;
      Print("[STATUS] Trade idea CANCELLED");
      return;
   }

   // At least one position is filled
   if(tp1Filled || tp2Filled)
   {
      if(idea.status == STATUS_PENDING)
      {
         idea.status = STATUS_ACTIVE;
         Print("[STATUS] Trade idea now ACTIVE");
      }
   }

   //--- Handle TP1 hit: if TP1 position is gone (TP hit by broker) but was filled before ---
   if(idea.status == STATUS_ACTIVE && !tp1Alive && idea.tp1Ticket > 0)
   {
      // TP1 position closed — could be TP or SL
      // Check trade history to determine outcome
      if(IsPositionClosedByTP(idea.tp1Ticket))
      {
         idea.status = STATUS_TP1_HIT;
         Print("[TP1 HIT] TP1 position closed at profit (1:2 RRR)");

         // Move TP2 SL to breakeven
         if(InpMoveSLtoBE && tp2Filled && idea.tp2Ticket > 0 && !idea.slMovedToBE)
         {
            if(PositionSelectByTicket(idea.tp2Ticket))
            {
               g_trade.PositionModify(idea.tp2Ticket, idea.entryPrice, idea.tp2Price);
               idea.slMovedToBE = true;
               Print("[BE] TP2 SL moved to breakeven: ", DoubleToString(idea.entryPrice, _Digits));
            }
         }
      }
      else
      {
         // TP1 hit SL
         Print("[SL HIT] TP1 position stopped out");
         g_todayLossCount++;
         // If TP2 is still pending, cancel it
         if(tp2Exists && !tp2Filled)
         {
            g_trade.OrderDelete(idea.tp2Ticket);
         }
         idea.status = STATUS_STOPPED;
         // Reset sweep/mss for potential new setup in next kill zone
         g_sweepDetected = false;
         g_mssDetected   = false;
         g_setupFound    = false;
         return;
      }
   }

   //--- Handle TP2 done ---
   if(idea.status == STATUS_TP1_HIT)
   {
      if(idea.tp2LotSize <= 0 || idea.tp2Ticket == 0)
      {
         idea.status = STATUS_COMPLETED;
         Print("[COMPLETED] Trade idea fully completed (no TP2)");
         g_sweepDetected = false;
         g_mssDetected   = false;
         g_setupFound    = false;
         return;
      }

      if(!tp2Alive)
      {
         if(IsPositionClosedByTP(idea.tp2Ticket))
         {
            idea.status = STATUS_COMPLETED;
            Print("[TP2 HIT] Trade idea fully completed at 1:3 RRR");
         }
         else
         {
            idea.status = STATUS_COMPLETED;
            Print("[TP2 CLOSED] TP2 position closed (BE or SL)");
         }
         // Reset for potential new setup
         g_sweepDetected = false;
         g_mssDetected   = false;
         g_setupFound    = false;
      }
   }

   //--- Handle case where active trade gets fully stopped ---
   if(idea.status == STATUS_ACTIVE && !tp1Alive && !tp2Alive)
   {
      idea.status = STATUS_STOPPED;
      g_todayLossCount++;
      Print("[STOPPED] Both positions closed — trade stopped");
      g_sweepDetected = false;
      g_mssDetected   = false;
      g_setupFound    = false;
   }
}

//+------------------------------------------------------------------+
//| HELPER: Check if a position was closed by TP (profit)             |
//+------------------------------------------------------------------+
bool IsPositionClosedByTP(ulong ticket)
{
   // Search in deal history
   datetime fromTime = TimeCurrent() - 86400; // Last 24 hours
   datetime toTime   = TimeCurrent() + 3600;

   if(!HistorySelect(fromTime, toTime))
      return false;

   int totalDeals = HistoryDealsTotal();
   for(int i = totalDeals - 1; i >= 0; i--)
   {
      ulong dealTicket = HistoryDealGetTicket(i);
      if(dealTicket == 0) continue;

      ulong dealPosition = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
      // Match by position ticket
      if(dealPosition == ticket || dealTicket == ticket)
      {
         long dealEntry = HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
         if(dealEntry == DEAL_ENTRY_OUT || dealEntry == DEAL_ENTRY_OUT_BY)
         {
            double dealProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
            double dealSwap   = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
            if((dealProfit + dealSwap) > 0)
               return true;
            else
               return false;
         }
      }
   }

   return false;
}

//+------------------------------------------------------------------+
//| END OF EXPERT ADVISOR                                             |
//+------------------------------------------------------------------+
'''

# Save to file
filepath = '/mnt/data/VenomModelEA.mq5'
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(ea_code)

print("VenomModelEA.mq5 created successfully!")
print(f"File size: {len(ea_code)} characters")
print(f"Approximate lines: {ea_code.count(chr(10))}")