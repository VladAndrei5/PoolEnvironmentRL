using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
#if UNITY_EDITOR
using UnityEditor;
#endif
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
        //Debug.Log("Server started. Waiting for a connection...");

        clientThread = new Thread(new ThreadStart(ListenForClients));
        clientThread.Start();
    }

    private void SendStateBack(NetworkStream stream){
        Debug.Log("sendstateback");
        // Get the updated state, reward, and terminal flag
        float[] state = env.GetState();
        float reward = env.GetReward();
        bool terminal = env.IsTerminal();

        // Construct response
        string response = $"{string.Join(",", state)},{reward},{terminal}";
        byte[] responseBytes = Encoding.ASCII.GetBytes(response);
        stream.Write(responseBytes, 0, responseBytes.Length);
    }

    private void SendWaitCommand(NetworkStream stream){
       // Debug.Log("wait");
        byte[] responseBytes = Encoding.ASCII.GetBytes("WAIT");
        stream.Write(responseBytes, 0, responseBytes.Length);
        //Debug.Log("waiting");
    }

    private void ListenForClients()
    {
        client = server.AcceptTcpClient();
       // Debug.Log("Client connected!");
        while (true)
        {   
            while(resetTheLevel){
                Thread.Sleep(20);
                //Debug.Log("sleep1");
            }
            try{
                
                NetworkStream stream = client.GetStream();

                while (!env.IsStateUpdated())
                {
                    //Debug.Log("hi");
                    SendWaitCommand(stream);
                    Thread.Sleep(20);
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
                        Thread.Sleep(20);
                    }
                    while (!env.IsStateUpdated())
                    {
                        SendWaitCommand(stream);
                        Thread.Sleep(20);
                    }

                    SendStateBack(stream);
                }
                else if (dataReceived.StartsWith("INSTRUCTION"))
                {
                    string[] instructions = dataReceived.Split(',');
                    //Debug.Log(instructions);
                    float instruction1 = float.Parse(instructions[1]);
                    float instruction2 = float.Parse(instructions[2]);
                    float instruction3 = float.Parse(instructions[3]);
                    //Debug.Log(instruction1);
                    //Debug.Log(instruction2);
                    // Execute instructions
                    env.TakeAction((instruction1, instruction2, instruction3));

                    // Wait until the state is updated
                    while (!env.IsStateUpdated())
                    {
                        SendWaitCommand(stream);
                        Thread.Sleep(20);
                    }

                    SendStateBack(stream);

                    // Check if the game is over
                    //if (env.gameOver)
                    //{
                        //resetTheLevel = true;
                    //}
                }
                else if (dataReceived.StartsWith("DISCONNECT"))
                {
                    Debug.Log("Stop");
                    StopPlaying();
                    server.Stop();
                    StopPlaying();
                    server.Stop();
                }
                else{
                    while (!env.IsStateUpdated())
                    {
                        SendWaitCommand(stream);
                        Thread.Sleep(20);
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

    public void StopPlaying()
    {
        Debug.Log("stop");
        #if UNITY_EDITOR
        if(EditorApplication.isPlaying)
        {
        UnityEditor.EditorApplication.isPlaying = false;
        }
        #endif
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