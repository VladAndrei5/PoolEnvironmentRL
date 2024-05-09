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
    private (float, float, float) action;
    public float angle = 0f;
    //public float speed;
    public float maxVelocity = 80f;
    // value between 0 - 1
    private float power = 1f;

    public float gameSpeed = 1f;
    private float previousGameSpeed = 1f;
 
    private bool stationaryBalls = true;

    public int currentPlayer = 0;
    public float reward;
    public int currentState;

    public bool gameOver = false;
    public bool changePlayer = false;
    public GameObject[] ballsArray;

    public TextMeshProUGUI playerNumbText;

    public float[] state;
    public int currentPlayerColour;

    public float rewardPerWrongBall = -10f;
    public float rewardPerCorrectBall = 10f;
    public float rewardPerBlackBall = -200f;
    public float rewardPerWin = 100f;
    public float rewardPerLose = -100f;
    public float rewardPerSkipTurn = -2f;
    public float rewardPerHittingCorrectBall = 3f;
    public float rewardPerHittingWrongBall = -3f;
    public float rewardPerNotHittingBall = -2;
    public float rewardPerTimeStep = -3f;

    public bool updatedState;

    private bool newActionRec = false;
    //!!!!!!!!!!!!!!!!!!!!
    //0 is red , 1 is yellow, 2 is black, 3 is white

    private void OnValidate()
    {
        // Check if the value of myVariable has changed
        if (gameSpeed != previousGameSpeed)
        {
            SetGameSpeed(gameSpeed);
            previousGameSpeed = gameSpeed;
        }
    }

    void Start(){
        previousGameSpeed = gameSpeed;
        //current player is Yellow
        //TODO change this
        updatedState = false;
        currentPlayerColour = 1;
        stationaryBalls = true;
        gameOver = false;
        changePlayer = false;
        playerNumbText.text = "1";
        playerNumbText.color = Color.red;
        ResetReward();
        SetGameSpeed(gameSpeed);
        UpdateState();
    }

    public IEnumerator Step((float, float, float) action){
        Debug.Log(action);
        updatedState = false;
        ResetReward();
        UpdateReward(rewardPerTimeStep);

        stationaryBalls = false;
        float randomAngleAdd = 0f;
        float randomPowerAdd = 0f;

        //Debug.Log("taking action..");
        //Debug.Log(action.Item1, action.Item2, action.Item3);
        whiteBallControls.MoveBall(action.Item1, action.Item2, action.Item3 * maxVelocity);
        //check if all balls are not moving
        
        bool checkedWhiteBall = false;
        while(!stationaryBalls){
            //In this loop the reward is updated
            stationaryBalls = true;
   
            foreach (GameObject ball in ballsArray){
                Rigidbody2D rb = ball.GetComponent<Rigidbody2D>();
                if (  (rb.velocity.magnitude > 0.2f || Mathf.Abs(rb.angularVelocity) > 0.2f) && rb.simulated == true ){
                    stationaryBalls = false;
                }
                else{
                    
                    if(ball.CompareTag("WhiteBall") && checkedWhiteBall == false){
                        whiteBallControls.CheckIfItHitReward();
                        checkedWhiteBall = true;
                    }
                    
                    rb.velocity = Vector2.zero;
                    rb.angularVelocity = 0f;
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
        if (newActionRec)
        {
            StartCoroutine(Step(action));
            // Reset action to default so that it's processed only once
            newActionRec = false;
        }

        if(serverhost.resetTheLevel == true){
            //Debug.Log("reseting env..");
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
            //stateList.Add((float)ballScript.GetBallColour());
            stateList.Add((float)ballScript.GetBallActive());


        }
        /*
        stateList.Add(22f);
        stateList.Add(12f);




        stateList.Add(-11f);
        stateList.Add(-6f);

        stateList.Add(-11f);
        stateList.Add(6f);

        stateList.Add(0f);
        stateList.Add(-6f);

        stateList.Add(0f);
        stateList.Add(6f);

        stateList.Add(11f);
        stateList.Add(-6f);

        stateList.Add(11f);
        stateList.Add(6f);
        */



        //stateList.Add(currentPlayerColour);
        //stateList.Add(currentPlayerColour);

        state = stateList.ToArray();
        //Debug.Log("reward " + reward);

        updatedState = true;
    }

    public bool CheckIfRedWon(){

        if(gameOver){
            return false;
        }

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
        if(gameOver){
            return false;
        }

        bool didWin = true;
        foreach (GameObject ball in ballsArray){
            BallScript ballScript = ball.GetComponent<BallScript>();
            if(ballScript.GetBallColour() == 1 && ballScript.GetBallActive() == 1){
                didWin = false;
                break;
            }
        }

        if(didWin && currentPlayerColour == 1){
            //Debug.Log("won");
            UpdateReward(rewardPerWin);
            gameOver = true;
        }
        else if(didWin){
            UpdateReward(rewardPerLose);
            gameOver = true;
        }

        return didWin;
    }

    public void UpdateReward(float newReward){
        reward = reward + newReward;
    }

    public void ResetReward(){
        reward = 0;
    }

    // Method to process data received from the Server GameObject
    public void ProcessReceivedData((float, float, float) receivedAction)
    {
        //Debug.Log("ProcessReceivedData...");
        action = receivedAction;
    }

    public void ResetEnv(){
        //Debug.Log("Resetting Enviornment");
        newActionRec = false;
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
        //Debug.Log("Enviornment Reset");
        UpdateState();
    }

    private void SetGameSpeed(float speed){
        //Debug.Log(speed);
        Time.timeScale = speed;
    }

    public bool IsTerminal()
    {
        return gameOver;
    }

    public float GetReward()
    {
        return reward;
    }

    public float[] GetState()
    {
        return state;
    }

    public void TakeAction((float, float, float) action){
        //Debug.Log(action);
        newActionRec = true;
        //Debug.Log(action);
        updatedState = false;
        //Debug.Log(this.action);
        this.action = action;
    }


}
