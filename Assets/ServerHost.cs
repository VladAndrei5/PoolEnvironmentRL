using System;
using System.Net;
using System.Net.Sockets;
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
            SendData("Hello from Unity!");
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
}