using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class SocketServer : MonoBehaviour
{
    public int serverPort = 12345;

    private TcpListener listener;
    private TcpClient client;
    private NetworkStream stream;

    void Start()
    {
        
        try
        {
            listener = new TcpListener(IPAddress.Any, serverPort);
            listener.Start();
            Debug.Log("Server started, waiting for connections...");
            
            // Accept client connection
            client = listener.AcceptTcpClient();
            stream = client.GetStream();

            // Example: Send data to Python client
            //SendData("Hello from Unity!");

        }
        catch (Exception e)
        {
            Debug.Log($"Server error: {e}");
        }
    }

    public void SendData(string data)
    {
        try
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(data);
            stream.Write(bytes, 0, bytes.Length);
        }
        catch (Exception e)
        {
            Debug.Log($"Error sending data: {e}");
        }
    }

    async Task HandleClientAsync(TcpClient client)
    {
        try
        {
            using (NetworkStream stream = client.GetStream())
            {
                byte[] buffer = new byte[1024];
                int bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);
                string requestData = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                Debug.Log("Request received: " + requestData);

                // Example: Send response back to Python client
                string responseData = "Hello from Unity!";
                byte[] responseBuffer = Encoding.UTF8.GetBytes(responseData);
                await stream.WriteAsync(responseBuffer, 0, responseBuffer.Length);
                Debug.Log("Response sent: " + responseData);
            }
        }
        catch (Exception e)
        {
            Debug.Log($"Error handling client: {e}");
        }
        finally
        {
            client.Close();
        }
    }
}