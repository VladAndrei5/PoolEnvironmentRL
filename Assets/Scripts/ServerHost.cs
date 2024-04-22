using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

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

                    // Parse the data
                    string[] stringArray = data.Split(',');
                    (float, float) action = (float.Parse(stringArray[0]), float.Parse(stringArray[1]));
                    Debug.Log("Received action: " + action);


                    //// Responses back to the client
                    //string response = "Server received action: " + data.ToString();
                    //SendResponseDataToClient(response);

                    // Pass the action data to the Environment GameObject
                    environment.ProcessReceivedData(action);

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
        if (stream != null)
        {
            stream.Close();
        }
        if (client != null)
        {
            client.Close();
        }
        if (server != null)
        {
            server.Stop();
        }
        //if(thread != null)
        //{
        //    thread.Abort();
        //}
    }

    public void ParseAndSendDataToClient(float[] state, float reward, bool terminal)
    {
        Debug.Log("parsing state: " + string.Join(",", state));
        Debug.Log("parsing reward: " + reward);
        Debug.Log("parsing terminal: " + terminal);
        SendResponseDataToClient(string.Join(",", state) + "||" + reward + "," + terminal);
    }

    public void SendResponseDataToClient(string message)
    {
        byte[] msg = Encoding.UTF8.GetBytes(message);
        stream.Write(msg, 0, msg.Length);
        Debug.Log("Sent: " + message);
    }
}