//+------------------------------------------------------------------+
//| MLTradingEA.mq5                                                  |
//| Machine Learning Trading Expert Advisor                          |
//| Connects to Python server, receives trading signals, executes    |
//| basic market orders with risk management                         |
//+------------------------------------------------------------------+

#property copyright "ML Trading System"
#property link "https://github.com/kuniyoshi/trade"
#property version "1.0"
#property strict

#include <ZeroMQ.mqh>
#include <PositionManager.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+

input string PythonHost = "127.0.0.1";        // Python server host
input int PythonPort = 5555;                   // Python server port
input double MinConfidence = 0.65;             // Minimum signal confidence
input double RiskPercent = 1.0;                // Risk per trade (%)
input int MagicNumber = 20260126;              // EA magic number
input string TradeComment = "ML_EA";           // Trade comment
input bool EnableTrading = true;               // Enable actual trading
input int SignalCooldownSeconds = 60;          // Cooldown between signals

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+

CZeroMQ g_zmq;                                 // ZeroMQ connection
CPositionManager g_posManager;                 // Position manager instance
datetime g_lastSignalTime = 0;                 // Last signal request time
bool g_initialized = false;                    // Initialization flag
int g_connectionFailures = 0;                  // Connection failure counter
int g_consecutiveOrderFailures = 0;            // Consecutive order failures
datetime g_lastReconnectAttempt = 0;           // Last reconnect attempt time
datetime g_tradingPausedUntil = 0;             // Trading pause until this time

// Error handling constants
const int MAX_CONNECTION_FAILURES = 5;         // Max failures before alert
const int MAX_ORDER_FAILURES = 3;              // Max order failures before pause
const int RECONNECT_COOLDOWN_SECONDS = 30;     // Seconds between reconnect attempts
const int TRADING_PAUSE_SECONDS = 300;         // Pause trading for 5 minutes after failures
const int MAX_RECONNECT_ATTEMPTS = 10;         // Max reconnect attempts before stopping

// Position management constants
const int MAX_POSITIONS = 3;                   // Max concurrent positions (from risk.yaml)
const double TRAILING_ACTIVATION_ATR = 1.5;    // Activation threshold (ATR multiplier)
const double TRAILING_DISTANCE_ATR = 1.0;      // Trailing distance (ATR multiplier)
const double TRAILING_STEP_PIPS = 5.0;         // Trailing step size in pips

// Error tracking
int g_totalReconnectAttempts = 0;              // Total reconnect attempts in session
int g_totalOrderErrors = 0;                    // Total order errors in session
int g_totalSignalErrors = 0;                   // Total signal request errors

//+------------------------------------------------------------------+
//| Expert Advisor Initialization                                    |
//+------------------------------------------------------------------+

int OnInit()
{
   Print("=== MLTradingEA Initialization Started ===");
   
   // Validate input parameters
   if(RiskPercent <= 0 || RiskPercent > 10)
   {
      Print("ERROR: RiskPercent must be between 0 and 10");
      return INIT_FAILED;
   }
   
   if(SignalCooldownSeconds < 1)
   {
      Print("ERROR: SignalCooldownSeconds must be at least 1");
      return INIT_FAILED;
   }
   
   // Connect to Python server
   Print(StringFormat("Connecting to Python server at %s:%d", PythonHost, PythonPort));
   
   if(!g_zmq.Connect(PythonHost, PythonPort))
   {
      Print("ERROR: Failed to connect to Python server");
      return INIT_FAILED;
   }
   
   // Test connection with PING
   Print("Testing connection with PING...");
   if(!g_zmq.Ping())
   {
      Print("ERROR: Python server not responding to PING");
      g_zmq.Disconnect();
      return INIT_FAILED;
   }
   
    // Initialize position manager
    g_posManager.Init(MagicNumber, MAX_POSITIONS);
    
    // Initialize variables
    g_lastSignalTime = 0;
    g_connectionFailures = 0;
    g_initialized = true;
    
    Print("=== MLTradingEA Initialized Successfully ===");
    Print(StringFormat("Settings: MinConfidence=%.2f, RiskPercent=%.2f%%, Cooldown=%d sec", 
                       MinConfidence, RiskPercent, SignalCooldownSeconds));
    Print(StringFormat("Trading Enabled: %s, Magic Number: %d", 
                       EnableTrading ? "YES" : "NO", MagicNumber));
    Print(StringFormat("Position Manager: Max=%d, Trailing: Activation=%.1f ATR, Distance=%.1f ATR, Step=%.0f pips",
                       MAX_POSITIONS, TRAILING_ACTIVATION_ATR, TRAILING_DISTANCE_ATR, TRAILING_STEP_PIPS));
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Advisor Deinitialization                                  |
//+------------------------------------------------------------------+

void OnDeinit(const int reason)
{
   Print("=== MLTradingEA Deinitialization ===");
   
   // Print session error summary
   Print("=== Session Error Summary ===");
   PrintFormat("  Total reconnect attempts: %d", g_totalReconnectAttempts);
   PrintFormat("  Total order errors: %d", g_totalOrderErrors);
   PrintFormat("  Total signal errors: %d", g_totalSignalErrors);
   
   if(g_initialized)
   {
      g_zmq.Disconnect();
      g_initialized = false;
      Print("Disconnected from Python server");
   }
   
   string reason_text = "";
   switch(reason)
   {
      case REASON_ACCOUNT:
         reason_text = "Account changed";
         break;
      case REASON_CHARTCHANGE:
         reason_text = "Chart changed";
         break;
      case REASON_CHARTCLOSE:
         reason_text = "Chart closed";
         break;
      case REASON_PARAMETERS:
         reason_text = "Parameters changed";
         break;
      case REASON_RECOMPILE:
         reason_text = "Recompiled";
         break;
      case REASON_REMOVE:
         reason_text = "EA removed";
         break;
      case REASON_TEMPLATE:
         reason_text = "Template changed";
         break;
      default:
         reason_text = "Unknown reason";
   }
   
   Print(StringFormat("Reason: %s (%d)", reason_text, reason));
   Print("=== MLTradingEA Shutdown Complete ===");
}

//+------------------------------------------------------------------+
//| Expert Advisor Main Tick Handler                                 |
//+------------------------------------------------------------------+

void OnTick()
{
   // Check if EA is initialized
   if(!g_initialized)
   {
      Print("WARNING: EA not initialized");
      return;
   }
   
   // Check if trading is enabled
   if(!EnableTrading)
   {
      return;
   }
   
   // Check if trading is paused due to errors
   if(IsTradingPaused())
   {
      return;
   }
   
   // Check connection status and attempt reconnection if needed
   if(!g_zmq.IsConnected())
   {
      if(!AttemptReconnection())
      {
         return;  // Reconnection failed or on cooldown
      }
   }
   
   // Reset connection failure counter on successful connection
   if(g_connectionFailures > 0)
   {
      g_connectionFailures = 0;
      Print("Connection restored to Python server");
   }
   
   // Check signal cooldown
   if(TimeCurrent() - g_lastSignalTime < SignalCooldownSeconds)
   {
      return;
   }
   
   // Update trailing stops for active positions
   UpdateTrailingStops();
   
   // Request signal from Python server
   RequestSignal();
}

//+------------------------------------------------------------------+
//| Check if Trading is Paused Due to Errors                         |
//+------------------------------------------------------------------+

bool IsTradingPaused()
{
   if(g_tradingPausedUntil > 0 && TimeCurrent() < g_tradingPausedUntil)
   {
      static datetime lastPauseLog = 0;
      // Log pause status every 60 seconds
      if(TimeCurrent() - lastPauseLog >= 60)
      {
         int remainingSeconds = (int)(g_tradingPausedUntil - TimeCurrent());
         PrintFormat("INFO: Trading paused for %d more seconds due to previous errors", 
                     remainingSeconds);
         lastPauseLog = TimeCurrent();
      }
      return true;
   }
   
   // Reset pause
   if(g_tradingPausedUntil > 0)
   {
      Print("INFO: Trading pause ended, resuming normal operation");
      g_tradingPausedUntil = 0;
      g_consecutiveOrderFailures = 0;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Pause Trading Due to Errors                                      |
//+------------------------------------------------------------------+

void PauseTrading(string reason)
{
   g_tradingPausedUntil = TimeCurrent() + TRADING_PAUSE_SECONDS;
   PrintFormat("WARNING: Trading paused for %d seconds - Reason: %s", 
               TRADING_PAUSE_SECONDS, reason);
}

//+------------------------------------------------------------------+
//| Attempt Reconnection to Python Server                            |
//+------------------------------------------------------------------+

bool AttemptReconnection()
{
   // Check reconnect cooldown
   if(TimeCurrent() - g_lastReconnectAttempt < RECONNECT_COOLDOWN_SECONDS)
   {
      return false;
   }
   
   g_lastReconnectAttempt = TimeCurrent();
   g_totalReconnectAttempts++;
   
   // Check if we've exceeded max reconnect attempts
   if(g_totalReconnectAttempts > MAX_RECONNECT_ATTEMPTS)
   {
      static bool maxAttemptsLogged = false;
      if(!maxAttemptsLogged)
      {
         Print("CRITICAL: Max reconnection attempts exceeded. Manual intervention required.");
         maxAttemptsLogged = true;
      }
      return false;
   }
   
   PrintFormat("INFO: Attempting reconnection to Python server (attempt %d/%d)...", 
               g_totalReconnectAttempts, MAX_RECONNECT_ATTEMPTS);
   
   // Disconnect first (clean up)
   g_zmq.Disconnect();
   
   // Wait briefly before reconnecting
   Sleep(500);
   
   // Attempt reconnection
   if(!g_zmq.Connect(PythonHost, PythonPort))
   {
      g_connectionFailures++;
      PrintFormat("WARNING: Reconnection failed (failures: %d)", g_connectionFailures);
      
      if(g_connectionFailures >= MAX_CONNECTION_FAILURES)
      {
         PauseTrading("Multiple connection failures");
      }
      
      return false;
   }
   
   // Test connection with PING
   if(!g_zmq.Ping())
   {
      Print("WARNING: Connected but PING failed");
      g_zmq.Disconnect();
      return false;
   }
   
   // Reset error counters on successful reconnection
   g_connectionFailures = 0;
   g_totalReconnectAttempts = 0;
   Print("SUCCESS: Reconnected to Python server");
   
   return true;
}

//+------------------------------------------------------------------+
//| Request Trading Signal from Python Server                        |
//+------------------------------------------------------------------+

void RequestSignal()
{
   // Build features JSON (simplified - Python server handles complex features)
   string features = BuildFeaturesJSON();
   
   // Request signal with retry logic
   string response;
   int retries = 0;
   const int maxRetries = 3;
   bool success = false;
   
   while(retries < maxRetries && !success)
   {
      if(g_zmq.GetSignal(_Symbol, "H1", features, response))
      {
         success = true;
      }
      else
      {
         retries++;
         g_totalSignalErrors++;
         
         if(retries < maxRetries)
         {
            PrintFormat("WARNING: Signal request failed (retry %d/%d)", retries, maxRetries);
            Sleep(100 * retries);  // Exponential backoff
         }
      }
   }
   
   if(!success)
   {
      PrintFormat("ERROR: Failed to get signal after %d attempts", maxRetries);
      g_connectionFailures++;
      return;
   }
   
   // Reset signal errors on success
   if(retries > 0)
   {
      PrintFormat("INFO: Signal request succeeded after %d retries", retries);
   }
   
   // Parse and execute signal
   ExecuteSignal(response);
   
   // Update last signal time
   g_lastSignalTime = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Build Features JSON for Signal Request                           |
//+------------------------------------------------------------------+

string BuildFeaturesJSON()
{
   // Get current OHLCV data
   double open = iOpen(_Symbol, PERIOD_H1, 0);
   double high = iHigh(_Symbol, PERIOD_H1, 0);
   double low = iLow(_Symbol, PERIOD_H1, 0);
   double close = iClose(_Symbol, PERIOD_H1, 0);
   long volume = iVolume(_Symbol, PERIOD_H1, 0);
   
   // Build JSON string
   string json = StringFormat(
      "{\"open\": %.5f, \"high\": %.5f, \"low\": %.5f, \"close\": %.5f, \"volume\": %d}",
      open, high, low, close, volume
   );
   
   return json;
}

//+------------------------------------------------------------------+
//| Execute Trading Signal                                           |
//+------------------------------------------------------------------+

void ExecuteSignal(string response)
{
   // Parse direction from response
   int buy_pos = StringFind(response, "\"direction\": \"buy\"");
   int sell_pos = StringFind(response, "\"direction\": \"sell\"");
   
   if(buy_pos < 0 && sell_pos < 0)
   {
      Print("WARNING: No valid direction in signal response");
      return;
   }
   
   // Parse confidence
   double confidence = ParseConfidence(response);
   if(confidence < MinConfidence)
   {
      PrintFormat("Signal confidence %.2f below minimum %.2f - skipping", 
                  confidence, MinConfidence);
      return;
   }
   
   // Parse stop loss and take profit
   double sl = ParseStopLoss(response);
   double tp = ParseTakeProfit(response);
   
   if(sl <= 0 || tp <= 0)
   {
      Print("ERROR: Invalid stop loss or take profit in signal");
      return;
   }
   
   // Execute trade based on direction
   if(buy_pos >= 0)
   {
      PrintFormat("BUY signal received - Confidence: %.2f%%", confidence * 100);
      OpenBuyOrder(sl, tp);
   }
   else if(sell_pos >= 0)
   {
      PrintFormat("SELL signal received - Confidence: %.2f%%", confidence * 100);
      OpenSellOrder(sl, tp);
   }
}

//+------------------------------------------------------------------+
//| Parse Confidence from Signal Response                            |
//+------------------------------------------------------------------+

double ParseConfidence(string response)
{
   int pos = StringFind(response, "\"confidence\":");
   if(pos < 0)
      return 0.0;
   
   // Find the number after "confidence":
   int start = pos + 14;  // Length of "\"confidence\":"
   int end = StringFind(response, ",", start);
   if(end < 0)
      end = StringFind(response, "}", start);
   
   if(end < 0)
      return 0.0;
   
   string value_str = StringSubstr(response, start, end - start);
   value_str = StringTrim(value_str);
   
   return StringToDouble(value_str);
}

//+------------------------------------------------------------------+
//| Parse Stop Loss from Signal Response                             |
//+------------------------------------------------------------------+

double ParseStopLoss(string response)
{
   int pos = StringFind(response, "\"stop_loss\":");
   if(pos < 0)
      return 0.0;
   
   int start = pos + 13;  // Length of "\"stop_loss\":"
   int end = StringFind(response, ",", start);
   if(end < 0)
      end = StringFind(response, "}", start);
   
   if(end < 0)
      return 0.0;
   
   string value_str = StringSubstr(response, start, end - start);
   value_str = StringTrim(value_str);
   
   return StringToDouble(value_str);
}

//+------------------------------------------------------------------+
//| Parse Take Profit from Signal Response                           |
//+------------------------------------------------------------------+

double ParseTakeProfit(string response)
{
   int pos = StringFind(response, "\"take_profit\":");
   if(pos < 0)
      return 0.0;
   
   int start = pos + 15;  // Length of "\"take_profit\":"
   int end = StringFind(response, ",", start);
   if(end < 0)
      end = StringFind(response, "}", start);
   
   if(end < 0)
      return 0.0;
   
   string value_str = StringSubstr(response, start, end - start);
   value_str = StringTrim(value_str);
   
   return StringToDouble(value_str);
}

//+------------------------------------------------------------------+
//| Trim Whitespace from String                                      |
//+------------------------------------------------------------------+

string StringTrim(string str)
{
   int len = StringLen(str);
   int start = 0;
   int end = len - 1;
   
   // Find first non-whitespace character
   while(start < len && (str[start] == ' ' || str[start] == '\t' || 
                         str[start] == '\n' || str[start] == '\r'))
      start++;
   
   // Find last non-whitespace character
   while(end >= start && (str[end] == ' ' || str[end] == '\t' || 
                          str[end] == '\n' || str[end] == '\r'))
      end--;
   
   if(start > end)
      return "";
   
   return StringSubstr(str, start, end - start + 1);
}

//+------------------------------------------------------------------+
//| Open Buy Order                                                   |
//+------------------------------------------------------------------+

bool OpenBuyOrder(double sl, double tp)
{
    // Get current ask price
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    if(price <= 0)
    {
       Print("ERROR: Failed to get ASK price");
       return false;
    }
    
    // Check if new position can be opened (position manager enforces limits)
    if(!g_posManager.CanOpenPosition(_Symbol))
    {
       Print("WARNING: Cannot open position - position limits enforced");
       return false;
    }
   
   // Calculate lot size
   double slPoints = MathAbs(price - sl) / _Point;
   double lots = CalculateLotSize(slPoints);
   
   if(lots <= 0)
   {
      Print("ERROR: Invalid lot size calculated");
      return false;
   }
   
   // Prepare trade request
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lots;
   request.type = ORDER_TYPE_BUY;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 20;
   request.magic = MagicNumber;
   request.comment = TradeComment;
   
   // Send order with retry logic
   bool orderSent = SendOrderWithRetry(request, result);
   
   if(!orderSent)
   {
      return false;
   }
   
   // Reset consecutive failures on success
   g_consecutiveOrderFailures = 0;
   
   // Log successful trade
   PrintFormat("BUY order opened - Ticket: %d, Lots: %.2f, Price: %.5f, SL: %.5f, TP: %.5f",
               result.order, lots, price, sl, tp);
   
   return true;
}

//+------------------------------------------------------------------+
//| Open Sell Order                                                  |
//+------------------------------------------------------------------+

bool OpenSellOrder(double sl, double tp)
{
    // Get current bid price
    double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    if(price <= 0)
    {
       Print("ERROR: Failed to get BID price");
       return false;
    }
    
    // Check if new position can be opened (position manager enforces limits)
    if(!g_posManager.CanOpenPosition(_Symbol))
    {
       Print("WARNING: Cannot open position - position limits enforced");
       return false;
    }
   
   // Calculate lot size
   double slPoints = MathAbs(sl - price) / _Point;
   double lots = CalculateLotSize(slPoints);
   
   if(lots <= 0)
   {
      Print("ERROR: Invalid lot size calculated");
      return false;
   }
   
   // Prepare trade request
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lots;
   request.type = ORDER_TYPE_SELL;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 20;
   request.magic = MagicNumber;
   request.comment = TradeComment;
   
   // Send order with retry logic
   bool orderSent = SendOrderWithRetry(request, result);
   
   if(!orderSent)
   {
      return false;
   }
   
   // Reset consecutive failures on success
   g_consecutiveOrderFailures = 0;
   
   // Log successful trade
   PrintFormat("SELL order opened - Ticket: %d, Lots: %.2f, Price: %.5f, SL: %.5f, TP: %.5f",
               result.order, lots, price, sl, tp);
   
   return true;
}

//+------------------------------------------------------------------+
//| Send Order with Retry Logic                                      |
//+------------------------------------------------------------------+

bool SendOrderWithRetry(MqlTradeRequest &request, MqlTradeResult &result)
{
   const int maxRetries = 3;
   int retries = 0;
   
   while(retries < maxRetries)
   {
      // Update price for retries (price may have moved)
      if(retries > 0)
      {
         if(request.type == ORDER_TYPE_BUY)
         {
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         }
         else
         {
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         }
      }
      
      if(OrderSend(request, result))
      {
         // Check if order was actually placed
         if(result.retcode == TRADE_RETCODE_DONE || 
            result.retcode == TRADE_RETCODE_PLACED)
         {
            return true;
         }
      }
      
      retries++;
      g_totalOrderErrors++;
      
      // Handle specific error codes
      switch(result.retcode)
      {
         case TRADE_RETCODE_REQUOTE:
            // Requote - retry immediately with new price
            PrintFormat("INFO: Requote received, retrying with new price (attempt %d)", retries);
            break;
            
         case TRADE_RETCODE_PRICE_CHANGED:
         case TRADE_RETCODE_PRICE_OFF:
            // Price changed - retry with delay
            PrintFormat("INFO: Price changed, retrying (attempt %d)", retries);
            Sleep(100);
            break;
            
         case TRADE_RETCODE_NO_MONEY:
            // Insufficient funds - don't retry
            Print("ERROR: Insufficient funds for trade");
            g_consecutiveOrderFailures++;
            return false;
            
         case TRADE_RETCODE_MARKET_CLOSED:
            // Market closed - don't retry
            Print("WARNING: Market is closed");
            return false;
            
         case TRADE_RETCODE_TRADE_DISABLED:
            // Trading disabled - don't retry
            Print("ERROR: Trading is disabled");
            return false;
            
         case TRADE_RETCODE_CONNECTION:
         case TRADE_RETCODE_TIMEOUT:
            // Connection issues - retry with longer delay
            PrintFormat("WARNING: Connection issue (code %d), retrying (attempt %d)", 
                        result.retcode, retries);
            Sleep(500 * retries);
            break;
            
         default:
            // Other errors - retry with delay
            PrintFormat("WARNING: Order failed (code %d: %s), retrying (attempt %d)", 
                        result.retcode, result.comment, retries);
            Sleep(200 * retries);
            break;
      }
   }
   
   // All retries exhausted
   g_consecutiveOrderFailures++;
   PrintFormat("ERROR: Order failed after %d attempts - Code: %d, Message: %s", 
               maxRetries, result.retcode, result.comment);
   
   // Check if we should pause trading
   if(g_consecutiveOrderFailures >= MAX_ORDER_FAILURES)
   {
      PauseTrading("Multiple consecutive order failures");
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Calculate Lot Size Based on Risk                                 |
//+------------------------------------------------------------------+

double CalculateLotSize(double stopLossPoints)
{
   if(stopLossPoints <= 0)
   {
      Print("ERROR: Invalid stop loss points");
      return 0.0;
   }
   
   // Get account balance
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(accountBalance <= 0)
   {
      Print("ERROR: Invalid account balance");
      return 0.0;
   }
   
   // Calculate risk amount
   double riskAmount = accountBalance * RiskPercent / 100.0;
   
   // Get symbol tick value
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tickValue <= 0)
   {
      Print("ERROR: Invalid tick value");
      return 0.0;
   }
   
   // Calculate lot size
   double lotSize = riskAmount / (stopLossPoints * tickValue);
   
   // Get symbol lot constraints
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   if(minLot <= 0 || maxLot <= 0 || lotStep <= 0)
   {
      Print("ERROR: Invalid lot constraints");
      return 0.0;
   }
   
   // Normalize to symbol's lot step
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   
   // Clamp to min/max
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   
   PrintFormat("Lot calculation - Risk: %.2f, SL Points: %.0f, Lot Size: %.2f",
               riskAmount, stopLossPoints, lotSize);
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| Update Trailing Stops for Active Positions                       |
//+------------------------------------------------------------------+

void UpdateTrailingStops()
{
    // Calculate ATR for trailing stop distance
    double atr = CalculateATR(_Symbol, PERIOD_H1, 14);
    if(atr <= 0)
    {
       return;  // ATR calculation failed, skip trailing stop update
    }
    
    // Calculate trailing distances in price units
    double activationDistance = atr * TRAILING_ACTIVATION_ATR;
    double trailDistance = atr * TRAILING_DISTANCE_ATR;
    
    // Update trailing stop for current symbol
    g_posManager.UpdateTrailingStop(_Symbol, trailDistance, TRAILING_STEP_PIPS);
}

//+------------------------------------------------------------------+
//| Calculate ATR for Trailing Stop Distance                         |
//+------------------------------------------------------------------+

double CalculateATR(string symbol, ENUM_TIMEFRAMES timeframe, int period)
{
    // Create ATR handle
    int atrHandle = iATR(symbol, timeframe, period);
    if(atrHandle == INVALID_HANDLE)
    {
       PrintFormat("ERROR: Failed to create ATR handle for %s", symbol);
       return 0.0;
    }
    
    // Get ATR value
    double atrBuffer[];
    if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) <= 0)
    {
       PrintFormat("ERROR: Failed to copy ATR buffer for %s", symbol);
       IndicatorRelease(atrHandle);
       return 0.0;
    }
    
    double atr = atrBuffer[0];
    IndicatorRelease(atrHandle);
    
    return atr;
}

//+------------------------------------------------------------------+
//| End of MLTradingEA.mq5                                           |
//+------------------------------------------------------------------+
