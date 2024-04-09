using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class ServerHost : MonoBehaviour
{
    public int serverPort = 12345;
    public Environment environment; // Reference to the Environment GameObject

    TcpListener server = null;
    TcpClient client = null;
    NetworkStream stream = null;
    Thread thread;

    private void Start()
    {
        thread = new Thread(new ThreadStart(SetupServer));
        thread.Start();
    }

    private void Update()
    {
        // test sentting hello
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SendMessageToClient("Hello");
        }
    }

    private void SetupServer()
    {
        try
        {
            server = new TcpListener(IPAddress.Any, serverPort);
            server.Start();
            Debug.Log("Server started.");

            byte[] buffer = new byte[1024];
            string data = null;

            while (true)
            {
                Debug.Log("Waiting for connection...");
                client = server.AcceptTcpClient();
                Debug.Log("Connected!");

                data = null;
                stream = client.GetStream();

                int i;

                while ((i = stream.Read(buffer, 0, buffer.Length)) != 0)
                {
                    data = Encoding.UTF8.GetString(buffer, 0, i);
                    //Debug.Log("Received: " + data);

                    // Parse the data
                    string[] stringArray = data.Split(',');
                    (float, float) action = (float.Parse(stringArray[0]), float.Parse(stringArray[1]));
                    Debug.Log("Received action: " + action);

                    // Pass the action data to the Environment GameObject
                    environment.ProcessReceivedData(action);

                    //string response = "Server response: " + data.ToString();
                    //SendMessageToClient(message: response);
                }
                client.Close();
            }
        }
        catch (SocketException e)
        {
            Debug.Log("SocketException: " + e);
        }
        finally
        {
            server.Stop();
        }
    }

    private void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
        server.Stop();
        thread.Abort();
    }

    public void SendMessageToClient(string message)
    {
        byte[] msg = Encoding.UTF8.GetBytes(message);
        stream.Write(msg, 0, msg.Length);
        Debug.Log("Sent: " + message);
    }
}