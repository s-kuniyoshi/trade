//+------------------------------------------------------------------+
//| ZeroMQ.mqh                                                        |
//| MQL5 wrapper library for ZeroMQ communication                     |
//| Provides simple interface for sending requests to Python server   |
//| and receiving responses using REQ/REP socket pattern              |
//+------------------------------------------------------------------+

#ifndef __ZEROMQ_MQH__
#define __ZEROMQ_MQH__

//+------------------------------------------------------------------+
//| ZeroMQ DLL imports                                               |
//+------------------------------------------------------------------+

#import "libzmq.dll"
   int zmq_ctx_new();
   int zmq_socket(int context, int type);
   int zmq_connect(int socket, string endpoint);
   int zmq_send(int socket, uchar &data[], int size, int flags);
   int zmq_recv(int socket, uchar &data[], int size, int flags);
   int zmq_close(int socket);
   int zmq_ctx_destroy(int context);
   int zmq_setsockopt(int socket, int option, int &value, int size);
#import

//+------------------------------------------------------------------+
//| ZeroMQ Constants                                                 |
//+------------------------------------------------------------------+

#define ZMQ_REQ 3
#define ZMQ_RCVTIMEO 27
#define ZMQ_SNDTIMEO 28

//+------------------------------------------------------------------+
//| CZeroMQ Class                                                    |
//| Wrapper for ZeroMQ REQ socket communication with Python server   |
//+------------------------------------------------------------------+

class CZeroMQ
{
private:
   int m_context;           // ZMQ context
   int m_socket;            // ZMQ socket (REQ type)
   string m_host;           // Server host
   int m_port;              // Server port
   int m_timeout;           // Timeout in milliseconds
   bool m_connected;        // Connection status

public:
   //--- Constructor and Destructor
   CZeroMQ();
   ~CZeroMQ();

   //--- Connection management
   bool Connect(string host, int port);
   void Disconnect();
   bool IsConnected() const { return m_connected; }

   //--- Low-level communication
   bool SendRequest(string json);
   bool ReceiveResponse(string &response);

   //--- High-level communication
   bool Ping();
   bool GetSignal(string symbol, string timeframe, string features, string &response);

private:
   //--- Helper methods
   void LogError(string message);
   void LogInfo(string message);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+

CZeroMQ::CZeroMQ()
{
   m_context = 0;
   m_socket = 0;
   m_host = "127.0.0.1";
   m_port = 5555;
   m_timeout = 5000;  // 5 seconds default
   m_connected = false;
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+

CZeroMQ::~CZeroMQ()
{
   Disconnect();
}

//+------------------------------------------------------------------+
//| Connect to ZeroMQ server                                         |
//+------------------------------------------------------------------+

bool CZeroMQ::Connect(string host, int port)
{
   // Store connection parameters
   m_host = host;
   m_port = port;

   // Create ZMQ context
   m_context = zmq_ctx_new();
   if(m_context == 0)
   {
      LogError("Failed to create ZMQ context");
      return false;
   }

   // Create REQ socket
   m_socket = zmq_socket(m_context, ZMQ_REQ);
   if(m_socket == 0)
   {
      LogError("Failed to create ZMQ socket");
      zmq_ctx_destroy(m_context);
      m_context = 0;
      return false;
   }

   // Set receive timeout
   int timeout = m_timeout;
   if(zmq_setsockopt(m_socket, ZMQ_RCVTIMEO, timeout, sizeof(int)) != 0)
   {
      LogError("Failed to set receive timeout");
      zmq_close(m_socket);
      zmq_ctx_destroy(m_context);
      m_socket = 0;
      m_context = 0;
      return false;
   }

   // Set send timeout
   if(zmq_setsockopt(m_socket, ZMQ_SNDTIMEO, timeout, sizeof(int)) != 0)
   {
      LogError("Failed to set send timeout");
      zmq_close(m_socket);
      zmq_ctx_destroy(m_context);
      m_socket = 0;
      m_context = 0;
      return false;
   }

   // Build endpoint string
   string endpoint = StringFormat("tcp://%s:%d", host, port);

   // Connect to server
   if(zmq_connect(m_socket, endpoint) != 0)
   {
      LogError(StringFormat("Failed to connect to %s", endpoint));
      zmq_close(m_socket);
      zmq_ctx_destroy(m_context);
      m_socket = 0;
      m_context = 0;
      return false;
   }

   m_connected = true;
   LogInfo(StringFormat("Connected to ZeroMQ server at %s", endpoint));
   return true;
}

//+------------------------------------------------------------------+
//| Disconnect from ZeroMQ server                                    |
//+------------------------------------------------------------------+

void CZeroMQ::Disconnect()
{
   if(m_socket != 0)
   {
      zmq_close(m_socket);
      m_socket = 0;
   }

   if(m_context != 0)
   {
      zmq_ctx_destroy(m_context);
      m_context = 0;
   }

   m_connected = false;
   LogInfo("Disconnected from ZeroMQ server");
}

//+------------------------------------------------------------------+
//| Send JSON request to server                                      |
//+------------------------------------------------------------------+

bool CZeroMQ::SendRequest(string json)
{
   if(!m_connected || m_socket == 0)
   {
      LogError("Not connected to server");
      return false;
   }

   // Convert string to byte array
   uchar data[];
   int len = StringLen(json);
   ArrayResize(data, len);
   StringToCharArray(json, data, 0, len);

   // Send request
   int sent = zmq_send(m_socket, data, len, 0);
   if(sent < 0)
   {
      LogError("Failed to send request (timeout or error)");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Receive JSON response from server                                |
//+------------------------------------------------------------------+

bool CZeroMQ::ReceiveResponse(string &response)
{
   if(!m_connected || m_socket == 0)
   {
      LogError("Not connected to server");
      return false;
   }

   // Receive response
   uchar data[4096];
   int received = zmq_recv(m_socket, data, ArraySize(data), 0);

   if(received < 0)
   {
      LogError("Failed to receive response (timeout or error)");
      return false;
   }

   // Convert byte array to string
   response = CharArrayToString(data, 0, received);
   return true;
}

//+------------------------------------------------------------------+
//| Send PING request and expect PONG response                       |
//+------------------------------------------------------------------+

bool CZeroMQ::Ping()
{
   if(!m_connected)
   {
      LogError("Not connected to server");
      return false;
   }

   // Build PING request
   string request = "{\"type\": \"PING\"}";

   // Send request
   if(!SendRequest(request))
   {
      LogError("Failed to send PING request");
      return false;
   }

   // Receive response
   string response;
   if(!ReceiveResponse(response))
   {
      LogError("Failed to receive PING response");
      return false;
   }

   // Check if response contains PONG
   if(StringFind(response, "PONG") >= 0)
   {
      LogInfo("PING successful - server is responding");
      return true;
   }

   LogError(StringFormat("Invalid PING response: %s", response));
   return false;
}

//+------------------------------------------------------------------+
//| Request trading signal from server                               |
//+------------------------------------------------------------------+

bool CZeroMQ::GetSignal(string symbol, string timeframe, string features, string &response)
{
   if(!m_connected)
   {
      LogError("Not connected to server");
      return false;
   }

   // Get current price
   double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);
   if(current_price == 0)
   {
      LogError(StringFormat("Failed to get price for symbol %s", symbol));
      return false;
   }

   // Get current timestamp
   string timestamp = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);

   // Build GET_SIGNAL request
   string request = StringFormat(
      "{\"type\": \"GET_SIGNAL\", \"data\": {\"symbol\": \"%s\", \"timeframe\": \"%s\", \"current_price\": %.5f, \"features\": %s}, \"timestamp\": \"%s\"}",
      symbol, timeframe, current_price, features, timestamp
   );

   // Send request
   if(!SendRequest(request))
   {
      LogError("Failed to send GET_SIGNAL request");
      return false;
   }

   // Receive response
   if(!ReceiveResponse(response))
   {
      LogError("Failed to receive GET_SIGNAL response");
      return false;
   }

   // Check if response indicates success
   if(StringFind(response, "\"status\": \"success\"") >= 0)
   {
      LogInfo(StringFormat("Signal received for %s %s", symbol, timeframe));
      return true;
   }

   // Log error if present
   int error_pos = StringFind(response, "\"error\":");
   if(error_pos >= 0)
   {
      LogError(StringFormat("Server error: %s", response));
   }
   else
   {
      LogError(StringFormat("Invalid signal response: %s", response));
   }

   return false;
}

//+------------------------------------------------------------------+
//| Log error message                                                |
//+------------------------------------------------------------------+

void CZeroMQ::LogError(string message)
{
   PrintFormat("[ZeroMQ ERROR] %s", message);
}

//+------------------------------------------------------------------+
//| Log info message                                                 |
//+------------------------------------------------------------------+

void CZeroMQ::LogInfo(string message)
{
   PrintFormat("[ZeroMQ INFO] %s", message);
}

#endif // __ZEROMQ_MQH__
