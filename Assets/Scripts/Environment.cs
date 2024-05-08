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

    [SerializeField] 
    private float reward;
    public int currentState;

    public bool gameOver = false;
    public GameObject[] ballsArray;

    public TextMeshProUGUI playerNumbText;

    public float[] state;
    public float rewardPerWhiteBall = 10f;
    public float rewardPerTimeStep = -1f;
    public GameObject hole; // Reference to the second object.
    public GameObject whiteBall;
    public float maxDistance = 1f; // Threshold distance for maximum reward.
    public float rewardPerProximity = 1f; // Maximum reward for proximity.

    public bool updatedState;

    private bool newActionRec = false;
    //!!!!!!!!!!!!!!!!!!!!
    //0 is red , 1 is yellow, 2 is black, 3 is white


    public void RewardProximity(){
        float distance = Vector3.Distance(whiteBall.transform.position, hole.transform.position);
        float rewardProx;

        // If within the max distance, calculate the reward proportionally.
        if (distance <= maxDistance)
        {
            rewardProx = rewardPerProximity * (1f - distance / maxDistance);
        }
        else
        {
            // If outside the max distance, reward is zero.
            rewardProx = 0f;
        }

        UpdateReward(rewardProx);
    }

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
        stationaryBalls = true;
        gameOver = false;
        playerNumbText.text = "1";
        playerNumbText.color = Color.red;
        ResetEnv();
        SetGameSpeed(gameSpeed);
        UpdateState();
    }

    public IEnumerator Step((float, float, float) action){
        //Debug.Log(action);
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
                    rb.velocity = Vector2.zero;
                    rb.angularVelocity = 0f;
                }
            }
            yield return null;
        }
        RewardProximity();
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
        }

        stateList.Add(hole.transform.position.x);
        stateList.Add(hole.transform.position.y);

        state = stateList.ToArray();
        Debug.Log("reward " + reward);
    

        updatedState = true;
    }

    public void UpdateReward(float newReward){
        reward = reward + newReward;
    }

    public void ResetReward(){
        reward = 0f;
    }

    public void ResetEnv(){
        Debug.Log("Resetting Enviornment");
        newActionRec = false;
        updatedState = false;
        ResetReward();
        gameOver = false;
        foreach (GameObject ball in ballsArray){
            BallScript ballScript = ball.GetComponent<BallScript>();
            ballScript.ResetBall();
        }

        stationaryBalls = true;
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
