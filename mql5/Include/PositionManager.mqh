//+------------------------------------------------------------------+
//| PositionManager.mqh                                              |
//| Position Management Class for ML Trading EA                      |
//| Handles position limits, trailing stops, and position tracking   |
//+------------------------------------------------------------------+

#ifndef __POSITION_MANAGER_MQH__
#define __POSITION_MANAGER_MQH__

//+------------------------------------------------------------------+
//| CPositionManager Class                                           |
//+------------------------------------------------------------------+

class CPositionManager
{
private:
   int m_magicNumber;           // Magic number for position filtering
   int m_maxPositions;          // Maximum concurrent positions
   
public:
   // Constructor
   CPositionManager(int magicNumber = 0, int maxPositions = 3)
   {
      m_magicNumber = magicNumber;
      m_maxPositions = maxPositions;
   }
   
   //+------------------------------------------------------------------+
   //| Initialize Position Manager                                     |
   //+------------------------------------------------------------------+
   
   void Init(int magicNumber, int maxPositions)
   {
      m_magicNumber = magicNumber;
      m_maxPositions = maxPositions;
      PrintFormat("PositionManager initialized - Magic: %d, Max Positions: %d", 
                  m_magicNumber, m_maxPositions);
   }
   
   //+------------------------------------------------------------------+
   //| Check if Position Exists for Symbol                             |
   //+------------------------------------------------------------------+
   
   bool HasPosition(string symbol)
   {
      // Iterate through all positions
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(PositionSelect(i))
         {
            // Check if position matches symbol and magic number
            if(PositionGetString(POSITION_SYMBOL) == symbol &&
               PositionGetInteger(POSITION_MAGIC) == m_magicNumber)
            {
               return true;
            }
         }
      }
      return false;
   }
   
   //+------------------------------------------------------------------+
   //| Get Position Details                                            |
   //+------------------------------------------------------------------+
   
   bool GetPosition(string symbol, double &lots, double &openPrice, 
                    double &sl, double &tp, long &type)
   {
      // Iterate through all positions
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(PositionSelect(i))
         {
            // Check if position matches symbol and magic number
            if(PositionGetString(POSITION_SYMBOL) == symbol &&
               PositionGetInteger(POSITION_MAGIC) == m_magicNumber)
            {
               lots = PositionGetDouble(POSITION_VOLUME);
               openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
               sl = PositionGetDouble(POSITION_SL);
               tp = PositionGetDouble(POSITION_TP);
               type = PositionGetInteger(POSITION_TYPE);
               
               PrintFormat("Position found - Symbol: %s, Lots: %.2f, Open: %.5f, SL: %.5f, TP: %.5f",
                           symbol, lots, openPrice, sl, tp);
               return true;
            }
         }
      }
      
      PrintFormat("WARNING: No position found for symbol %s", symbol);
      return false;
   }
   
   //+------------------------------------------------------------------+
   //| Count Total Positions with Magic Number                         |
   //+------------------------------------------------------------------+
   
   int CountPositions()
   {
      int count = 0;
      
      // Iterate through all positions
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(PositionSelect(i))
         {
            // Count positions with matching magic number
            if(PositionGetInteger(POSITION_MAGIC) == m_magicNumber)
            {
               count++;
            }
         }
      }
      
      return count;
   }
   
   //+------------------------------------------------------------------+
   //| Check if New Position Can Be Opened                             |
   //+------------------------------------------------------------------+
   
   bool CanOpenPosition(string symbol)
   {
      // Check if position already exists for this symbol
      if(HasPosition(symbol))
      {
         PrintFormat("WARNING: Position already exists for %s (max 1 per symbol)", symbol);
         return false;
      }
      
      // Check if total position limit reached
      int currentCount = CountPositions();
      if(currentCount >= m_maxPositions)
      {
         PrintFormat("WARNING: Position limit reached (%d/%d)", currentCount, m_maxPositions);
         return false;
      }
      
      PrintFormat("Position allowed - Current: %d/%d, Symbol: %s", 
                  currentCount, m_maxPositions, symbol);
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Close Position by Ticket                                        |
   //+------------------------------------------------------------------+
   
   bool ClosePosition(ulong ticket)
   {
      // Select position by ticket
      if(!PositionSelectByTicket(ticket))
      {
         PrintFormat("ERROR: Position ticket %d not found", ticket);
         return false;
      }
      
      // Get position details
      string symbol = PositionGetString(POSITION_SYMBOL);
      long type = PositionGetInteger(POSITION_TYPE);
      double volume = PositionGetDouble(POSITION_VOLUME);
      
      // Prepare close request
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_DEAL;
      request.position = ticket;
      request.symbol = symbol;
      request.volume = volume;
      request.type = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      request.deviation = 20;
      request.magic = m_magicNumber;
      request.comment = "PositionManager Close";
      
      // Get current price
      if(type == POSITION_TYPE_BUY)
      {
         request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
      }
      else
      {
         request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
      }
      
      // Send close order
      if(!OrderSend(request, result))
      {
         PrintFormat("ERROR: Close failed - Ticket: %d, Code: %d, Message: %s", 
                     ticket, result.retcode, result.comment);
         return false;
      }
      
      PrintFormat("Position closed - Ticket: %d, Symbol: %s, Volume: %.2f", 
                  ticket, symbol, volume);
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Modify Position SL/TP                                           |
   //+------------------------------------------------------------------+
   
   bool ModifyPosition(ulong ticket, double sl, double tp)
   {
      // Select position by ticket
      if(!PositionSelectByTicket(ticket))
      {
         PrintFormat("ERROR: Position ticket %d not found", ticket);
         return false;
      }
      
      // Get position details
      string symbol = PositionGetString(POSITION_SYMBOL);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      
      // Check if modification is needed
      if(currentSL == sl && currentTP == tp)
      {
         PrintFormat("INFO: No modification needed - SL: %.5f, TP: %.5f", sl, tp);
         return true;
      }
      
      // Prepare modify request
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_SLTP;
      request.position = ticket;
      request.sl = sl;
      request.tp = tp;
      request.magic = m_magicNumber;
      
      // Send modify order
      if(!OrderSend(request, result))
      {
         PrintFormat("ERROR: Modify failed - Ticket: %d, Code: %d, Message: %s", 
                     ticket, result.retcode, result.comment);
         return false;
      }
      
      PrintFormat("Position modified - Ticket: %d, Symbol: %s, SL: %.5f, TP: %.5f", 
                  ticket, symbol, sl, tp);
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Update Trailing Stop for Symbol                                 |
   //+------------------------------------------------------------------+
   
   bool UpdateTrailingStop(string symbol, double trailDistance, double stepPips)
   {
      // Find position for symbol
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(PositionSelect(i))
         {
            // Check if position matches symbol and magic number
            if(PositionGetString(POSITION_SYMBOL) == symbol &&
               PositionGetInteger(POSITION_MAGIC) == m_magicNumber)
            {
               ulong ticket = PositionGetInteger(POSITION_TICKET);
               long type = PositionGetInteger(POSITION_TYPE);
               double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
               double currentSL = PositionGetDouble(POSITION_SL);
               double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
               
               // Calculate new SL based on position type
               double newSL = 0.0;
               double profit = 0.0;
               
               if(type == POSITION_TYPE_BUY)
               {
                  profit = currentPrice - openPrice;
                  
                  // Only trail if profit >= activation threshold
                  if(profit >= trailDistance)
                  {
                     newSL = currentPrice - trailDistance;
                     
                     // Only move SL up (never down for buy)
                     if(newSL > currentSL)
                     {
                        // Check if move is at least stepPips
                        double slDiff = (newSL - currentSL) / _Point;
                        if(slDiff >= stepPips)
                        {
                           return ModifyPosition(ticket, newSL, 0);
                        }
                     }
                  }
               }
               else if(type == POSITION_TYPE_SELL)
               {
                  profit = openPrice - currentPrice;
                  
                  // Only trail if profit >= activation threshold
                  if(profit >= trailDistance)
                  {
                     newSL = currentPrice + trailDistance;
                     
                     // Only move SL down (never up for sell)
                     if(newSL < currentSL || currentSL == 0)
                     {
                        // Check if move is at least stepPips
                        double slDiff = (currentSL - newSL) / _Point;
                        if(slDiff >= stepPips || currentSL == 0)
                        {
                           return ModifyPosition(ticket, newSL, 0);
                        }
                     }
                  }
               }
               
               return true;
            }
         }
      }
      
      // No position found for symbol (not an error, just no update needed)
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Calculate ATR for Trailing Stop Distance                        |
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
};

#endif // __POSITION_MANAGER_MQH__
