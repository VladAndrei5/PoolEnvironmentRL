using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Net;
using System.Net.Sockets;
using TMPro;

public class Environment : MonoBehaviour
{
    public ServerHost serverhost;
    public MoveWhiteBall whiteBallControls;
    private (float, float) action;
    public float angle = 0f;
    //public float speed;
    public float maxVelocity = 20f;
    // value between 0 - 1
    private float power = 1f;
    public float gameSpeed = 1f;

    private bool stationaryBalls = true;

    public int currentPlayer = 0;
    public int currentReward;
    public int currentState;

    public bool gameOver = false;
    public bool resetWhiteBall = false;
    public bool changePlayer = false;
    public GameObject[] ballsArray;

    public TextMeshProUGUI playerNumbText;

    void Start(){
        ballsArray = GameObject.FindGameObjectsWithTag("Ball");
        stationaryBalls = true;
        gameOver =false;
        resetWhiteBall = false;
        changePlayer = false;
        playerNumbText.text = "1";
        playerNumbText.color = Color.red;
        SetGameSpeed(gameSpeed);
    }

    IEnumerator Step((float, float) action){
        stationaryBalls = false;
        float randomAngleAdd = Random.Range((float)-0.03, (float)0.03);
        float randomPowerAdd = 0f;
        whiteBallControls.MoveBall(action.Item1 + randomAngleAdd, (action.Item2 + randomPowerAdd) * maxVelocity);
        //check if all balls are not moving
        
        while(!stationaryBalls){
            stationaryBalls = true;
            foreach (GameObject ball in ballsArray){
                Rigidbody2D rb = ball.GetComponent<Rigidbody2D>();
                if (  (rb.velocity.magnitude > 0.01f || Mathf.Abs(rb.angularVelocity) > 0.01f) && rb.simulated == true ){
                    stationaryBalls = false;
                }
            }
            yield return null;
        }
        //substract one reward for taking a turn
        currentReward--;

        //return reward and state
        yield break;
    }

    void Update()
    {
        // Check if new data has been received from the client
        if (action != default)
        {
            Debug.Log("taking action..");
            TakeAction(action);

            // Reset action to default so that it's processed only once
            action = default;
            Debug.Log("Finished the action");
        }
    }

    public void TakeAction((float, float) _action)
    {
        // Process the received action
        // TODO:use a try catch to avoid invalid receivedAction
        if (stationaryBalls)
        {
            Debug.Log("Taking action.");
            StartCoroutine(Step(_action));
            Debug.Log("currentReward: " + currentReward);

            // Send response data back to the client
            serverhost.SendResponseDataToClient(currentReward.ToString());

            //currentReward = 0;
            if (gameOver)
            {
                ResetEnv();
            }
            if (resetWhiteBall)
            {
                resetWhiteBall = false;
                whiteBallControls.Reset();
            }
            if (changePlayer)
            {
                changePlayer = false;
                if (currentPlayer == 1)
                {
                    playerNumbText.text = "1";
                    playerNumbText.color = Color.red;
                    currentPlayer = 0;
                }
                else
                {
                    playerNumbText.text = "2";
                    playerNumbText.color = Color.yellow;
                    currentPlayer = 1;
                }
            }
        }
    }

    // Method to process data received from the Server GameObject
    public void ProcessReceivedData((float, float) receivedAction)
    {

        Debug.Log("ProcessReceivedData...");
        action = receivedAction;


    }

    private void ResetEnv(){
        SceneManager.LoadScene(0);
    }

    private void SetGameSpeed(float speed){
        Time.timeScale = speed;
    }


}
