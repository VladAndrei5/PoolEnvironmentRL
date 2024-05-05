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

    private void SendStateBack(NetworkStream stream){

        // Get the updated state, reward, and terminal flag
        float[] state = env.GetState();
        int reward = env.GetReward();
        bool terminal = env.IsTerminal();

        // Construct response
        string response = $"{string.Join(",", state)},{reward},{terminal}";
        byte[] responseBytes = Encoding.ASCII.GetBytes(response);

        stream.Write(responseBytes, 0, responseBytes.Length);

        Debug.Log($"Sent state: {response}");
    }

    private void SendWaitCommand(NetworkStream stream){

        byte[] responseBytes = Encoding.ASCII.GetBytes("WAIT");
        stream.Write(responseBytes, 0, responseBytes.Length);
        Debug.Log("waiting");
    }

    private void SendReadyToGo(NetworkStream stream){

        byte[] responseBytes = Encoding.ASCII.GetBytes("READYTOGO");
        stream.Write(responseBytes, 0, responseBytes.Length);
    }


    private void ListenForClients()
    {
        client = server.AcceptTcpClient();
        Debug.Log("Client connected!");
        while (true)
        {   
            while(resetTheLevel){
                Thread.Sleep(50);
            }
            try{
                
                NetworkStream stream = client.GetStream();

                while (!env.IsStateUpdated())
                {
                    SendWaitCommand(stream);
                    Thread.Sleep(50);
                }

                SendStateBack(stream);

                byte[] buffer = new byte[1024];
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                string dataReceived = Encoding.ASCII.GetString(buffer, 0, bytesRead);

                if (dataReceived.StartsWith("RESET"))
                {
                    // Reset the environment
                    resetTheLevel = true;
                    while (resetTheLevel)
                    {
                        SendWaitCommand(stream);
                        Thread.Sleep(50);
                    }
                    while (!env.IsStateUpdated())
                    {
                        SendWaitCommand(stream);
                        Thread.Sleep(50);
                    }
                    
                    SendStateBack(stream);
                }
                else if (dataReceived.StartsWith("INSTRUCTION"))
                {
                    string[] instructions = dataReceived.Split(',');
                    float instruction1 = float.Parse(instructions[1]);
                    float instruction2 = float.Parse(instructions[2]);

                    // Execute instructions
                    env.TakeAction((instruction1, instruction2));

                    // Wait until the state is updated
                    while (!env.IsStateUpdated())
                    {
                        SendWaitCommand(stream);
                        Thread.Sleep(50);
                    }

                    SendStateBack(stream);

                    // Check if the game is over
                    if (env.gameOver)
                    {
                        resetTheLevel = true;
                    }
                }
                else{
                    while (!env.IsStateUpdated())
                    {
                        SendWaitCommand(stream);
                        Thread.Sleep(50);
                    }
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