using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Net;
using System.Net.Sockets;

public class Environment : MonoBehaviour
{
    public string serverIP = "127.0.0.1";
    public int serverPort = 12345;
    private TcpClient client;
    private NetworkStream stream;


    public MoveWhiteBall whiteBallControls;
    private (float, float) action;
    public float angle = 0.2f;
    //public float speed;

    public float maxSpeed = 20f;

    // value between 0 - 1
    private float power = 1f;

    void Start(){
         try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
        }
        catch (System.Exception e)
        {
            Debug.Log($"Socket connection error: {e}");
        }

        Step((angle, power));
        //ExampleFunction();
    }

    public void SendData(string data)
    {
        try
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(data);
            stream.Write(bytes, 0, bytes.Length);
        }
        catch (System.Exception e)
        {
            Debug.Log($"Error sending data: {e}");
        }
    }

    public string ReceiveData()
    {
        string receivedData = "";
        try
        {
            byte[] buffer = new byte[1024];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            receivedData = System.Text.Encoding.UTF8.GetString(buffer, 0, bytesRead);
        }
        catch (System.Exception e)
        {
            Debug.Log($"Error receiving data: {e}");
        }
        return receivedData;
    }

    // Example function to use SendData and ReceiveData
    public void ExampleFunction()
    {
        // Send data to Python
        SendData("Hello from Unity!");

        // Receive data from Python
        string receivedData = ReceiveData();
        Debug.Log("Received from Python: " + receivedData);
    }


    private void Step((float, float) action){
        whiteBallControls.MoveBall(action.Item1, action.Item2 * maxSpeed);
    }

    private void ResetEnv(){
        SceneManager.LoadScene(0);
    }

}
