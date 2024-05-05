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
    public float maxVelocity = 25f;
    // value between 0 - 1
    private float power = 1f;
    public float gameSpeed = 1f;

    private bool stationaryBalls = true;

    public int currentPlayer = 0;
    public int reward;
    public int currentState;

    public bool gameOver = false;
    public bool changePlayer = false;
    public GameObject[] ballsArray;

    public TextMeshProUGUI playerNumbText;

    public float[] state;
    public int currentPlayerColour;

    public int rewardPerWrongBall = -5;
    public int rewardPerCorrectBall = 5;
    public int rewardPerBlackBall = -200;
    public int rewardPerWin = 100;
    public int rewardPerLose = -100;
    public int rewardPerSkipTurn = -2;

    public bool updatedState;
    //!!!!!!!!!!!!!!!!!!!!
    //0 is red , 1 is yellow, 2 is black, 3 is white

    void Start(){
        //current player is Yellow
        //TODO change this
        updatedState = false;
        currentPlayerColour = 1;
        stationaryBalls = true;
        gameOver = false;
        changePlayer = false;
        playerNumbText.text = "1";
        playerNumbText.color = Color.red;
        reward = 0;
        SetGameSpeed(gameSpeed);
        UpdateState();
    }

    public IEnumerator Step((float, float) action){
        updatedState = false;
        reward = -1;

        stationaryBalls = false;
        float randomAngleAdd = Random.Range((float)-0.03, (float)0.03);
        float randomPowerAdd = 0f;

        Debug.Log("taking action..");
        whiteBallControls.MoveBall(action.Item1 + randomAngleAdd, (action.Item2 + randomPowerAdd) * maxVelocity);
        //check if all balls are not moving
        
        while(!stationaryBalls){
            //In this loop the reward is updated
            stationaryBalls = true;
            foreach (GameObject ball in ballsArray){
                Rigidbody2D rb = ball.GetComponent<Rigidbody2D>();
                if (  (rb.velocity.magnitude > 0.1f || Mathf.Abs(rb.angularVelocity) > 0.1f) && rb.simulated == true ){
                    stationaryBalls = false;
                }
            }
            yield return null;
        }
        UpdateState();
        yield break;
    }
    
    
    void Update()
    {
        // Check if new data has been received from the client
        if (action != default)
        {
            StartCoroutine(Step(action));
            // Reset action to default so that it's processed only once
            action = default;
        }

        if(serverhost.resetTheLevel == true){
            Debug.Log("reseting env..");
            ResetEnv();
            serverhost.resetTheLevel = false; 
        }

    }
    

    public bool IsStateUpdated(){
        return updatedState;
    }
    
    public void ResetWhiteBall(){
        whiteBallControls.Reset();
    }

    public void UpdateState(){
        List<float> stateList = new List<float>();
        //string[] stateStrList = new string[ballsArray.Length * 4];

        foreach (GameObject ball in ballsArray){
            BallScript ballScript = ball.GetComponent<BallScript>();

            stateList.Add(ballScript.GetPositionX());
            stateList.Add(ballScript.GetPositionY());
            stateList.Add((float)ballScript.GetBallColour());
            stateList.Add((float)ballScript.GetBallActive());

        }

        stateList.Add(currentPlayerColour);

        state = stateList.ToArray();
        updatedState = true;

        Debug.Log("UpdateState");
    }

    public bool CheckIfRedWon(){
        bool didWin = true;
        foreach (GameObject ball in ballsArray){
            BallScript ballScript = ball.GetComponent<BallScript>();
            if(ballScript.GetBallColour() == 0 && ballScript.GetBallActive() == 1){
                didWin = false;
                break;
            }
        }

        if(didWin && currentPlayerColour == 0){
            UpdateReward(rewardPerWin);
            gameOver = true;
        }
        else if(didWin){
            UpdateReward(rewardPerLose);
            gameOver = true;
        }

        return didWin;
    }

    public bool CheckIfYellowWon(){
        bool didWin = true;
        foreach (GameObject ball in ballsArray){
            BallScript ballScript = ball.GetComponent<BallScript>();
            if(ballScript.GetBallColour() == 1 && ballScript.GetBallActive() == 1){
                didWin = false;
                break;
            }
        }

        if(didWin && currentPlayerColour == 1){
            UpdateReward(rewardPerWin);
            gameOver = true;
        }
        else if(didWin){
            UpdateReward(rewardPerLose);
            gameOver = true;
        }

        return didWin;
    }

    public void UpdateReward(int newReward){
        reward = reward + newReward;
    }

    public void ResetReward(){
        reward = 0;
    }

    // Method to process data received from the Server GameObject
    public void ProcessReceivedData((float, float) receivedAction)
    {
        Debug.Log("ProcessReceivedData...");
        action = receivedAction;
    }

    public void ResetEnv(){
        Debug.Log("Resetting Enviornment");
        updatedState = false;
        reward = 0;
        gameOver = false;
        foreach (GameObject ball in ballsArray){
            BallScript ballScript = ball.GetComponent<BallScript>();
            ballScript.ResetBall();
        }
        currentPlayerColour = 1;
        stationaryBalls = true;
        changePlayer = false;
        playerNumbText.text = "1";
        playerNumbText.color = Color.red;
        Debug.Log("Enviornment Reset");
        UpdateState();
    }

    private void SetGameSpeed(float speed){
        Time.timeScale = speed;
    }

    public bool IsTerminal()
    {
        return gameOver;
    }

    public int GetReward()
    {
        return reward;
    }

    public float[] GetState()
    {
        return state;
    }

    public void TakeAction((float, float) action){
        Debug.Log(action);
        updatedState = false;
        this.action = action;
    }


}
