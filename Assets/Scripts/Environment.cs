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
        updatedState = true;
        currentPlayerColour = 1;
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
        updatedState = false;
        currentReward = -1;

        stationaryBalls = false;
        float randomAngleAdd = Random.Range((float)-0.03, (float)0.03);
        float randomPowerAdd = 0f;
        whiteBallControls.MoveBall(action.Item1 + randomAngleAdd, (action.Item2 + randomPowerAdd) * maxVelocity);
        //check if all balls are not moving
        
        while(!stationaryBalls){
            //In this loop the reward is updated
            stationaryBalls = true;
            foreach (GameObject ball in ballsArray){
                Rigidbody2D rb = ball.GetComponent<Rigidbody2D>();
                if (  (rb.velocity.magnitude > 0.01f || Mathf.Abs(rb.angularVelocity) > 0.01f) && rb.simulated == true ){
                    stationaryBalls = false;
                }
            }
            yield return null;
        }

        UpdateState();
        // send state, send reward, send gameOver


        if(gameOver){
            ResetEnv();
        }

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
        if (updatedState)
        {
            Debug.Log("Taking action.");
            StartCoroutine(Step(_action));
            Debug.Log("currentReward: " + currentReward);

            // Send response data back to the client
            serverhost.SendResponseDataToClient(state.ToString() + currentReward.ToString());

        }
    }
    

    /*
    void Update(){
        if(updatedState){
            StartCoroutine(Step( (0.1f, 1f) ));
        }
    }
    */

    public void ReseWhitetBall(){
        whiteBallControls.Reset();
    }

    public void UpdateState(){
        Debug.Log("hERE");
        List<float> stateList = new List<float>();

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
        currentReward = currentReward + newReward;
    }

    public void ResetReward(){
        currentReward = 0;
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
