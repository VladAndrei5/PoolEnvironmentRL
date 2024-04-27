using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class ServerHost : MonoBehaviour
{
    private TcpListener server;
    private TcpClient client;
    private Thread clientThread;

    public Environment env;
    public bool resetTheLevel = false;
    private void Start()
    {
        resetTheLevel = false;
        server = new TcpListener(IPAddress.Parse("127.0.0.1"), 8888);
        server.Start();
        Debug.Log("Server started. Waiting for a connection...");

        clientThread = new Thread(new ThreadStart(ListenForClients));
        clientThread.Start();
    }

    private void ListenForClients()
    {
        client = server.AcceptTcpClient();
        Debug.Log("Client connected!");
        while (true)
        {   
            while(resetTheLevel){
                Thread.Sleep(500);
            }
            try{
                NetworkStream stream = client.GetStream();

                byte[] buffer = new byte[1024];
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                string dataReceived = Encoding.ASCII.GetString(buffer, 0, bytesRead);

                string[] instructions = dataReceived.Split(',');
                float instruction1 = float.Parse(instructions[0]);
                float instruction2 = float.Parse(instructions[1]);

                Debug.Log($"Received instructions: ({instruction1}, {instruction2})");
                //StartCoroutine(env.Step((instruction1, instruction2)));
                env.TakeAction((instruction1, instruction2));
                //Debug.Log(env.IsStateUpdated());
                // Wait until a condition becomes true
                while (!env.IsStateUpdated())
                {
                    //Debug.Log("Sleeping");
                    //Debug.Log(env.IsStateUpdated());
                    byte[] responseBytesWait = Encoding.ASCII.GetBytes("WAIT");
                    stream.Write(responseBytesWait, 0, responseBytesWait.Length);
                    Thread.Sleep(500);
                }

                float[] state = env.GetState();
                int reward = env.GetReward();
                bool terminal = env.IsTerminal();
                //Debug.Log("terminal " + terminal);

                string response = $"{string.Join(",", state)},{reward},{terminal}";
                byte[] responseBytes = Encoding.ASCII.GetBytes(response);
                stream.Write(responseBytes, 0, responseBytes.Length);

                Debug.Log($"Sent response: {response}");

                //reset the game if game is over
                if(env.gameOver == true){
                    resetTheLevel = true;
                }
            }
            catch (Exception e)
            {
                // Catching any exception that occurs
                Debug.LogError("An error occurred: " + e.Message);
                server.Stop();
            }
        }
    }

    /*
    private bool SomeCondition()
    {
        // Replace with your actual condition
        return true;
    }

    private bool IsTerminal()
    {
        // Replace with your actual terminal condition
        return false;
    }

    private int GetReward()
    {
        // Replace with your actual reward calculation
        return 10;
    }

    private float[] GetState()
    {
        // Replace with your actual state retrieval
        return new float[] { 1.0f, 2.0f, 3.0f };
    }
    */

    private void OnApplicationQuit()
    {
        if (clientThread != null)
        {
            clientThread.Abort();
        }

        if (client != null)
        {
            client.Close();
        }

        if (server != null)
        {
            server.Stop();
        }
    }
}