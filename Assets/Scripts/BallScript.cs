using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallScript : MonoBehaviour
{
    // Start is called before the first frame update
    private bool isActive;
    private bool isMoving;

    //0 is red , 1 is yellow, 2 is black, 3 is white
    public int ballColour;
    public Environment env;

    private Vector2 originalPosition;
    void Awake()
    {
        isActive = true;
        isMoving = false;
        originalPosition = new Vector2(transform.position.x, transform.position.y);
    }

    public void ResetBall(){
        transform.position = new Vector3(originalPosition.x, originalPosition.y, transform.position.z);
        isActive = true;
        isMoving = false;
        GetComponent<SpriteRenderer>().enabled = true;
        GetComponent<CircleCollider2D>().enabled = true;
        GetComponent<Rigidbody2D>().simulated = true;
    }
    private void DisableBall(){
        isActive = false;
        GetComponent<SpriteRenderer>().enabled = false;
        GetComponent<CircleCollider2D>().enabled = false;
        GetComponent<Rigidbody2D>().simulated = false;
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        // Check if the collider that the rigid body entered is a trigger collider
        if (other.isTrigger)
        {   

            //Give rewards if ball falls in pocket based on its colour 
            if(isActive){
                if( ballColour == 2){
                    env.UpdateReward(env.rewardPerBlackBall);
                    env.gameOver = true;
                }
                else if(ballColour == 3){
                    env.UpdateReward(env.rewardPerSkipTurn);
                    env.ResetWhiteBall();
                }
                else if(env.currentPlayer == ballColour){
                    env.UpdateReward(env.rewardPerCorrectBall);
                    env.CheckIfRedWon();
                    env.CheckIfYellowWon();
                    DisableBall();
                }
                else if(env.currentPlayer != ballColour){
                    env.UpdateReward(env.rewardPerWrongBall);
                    env.CheckIfRedWon();
                    env.CheckIfYellowWon();
                    //env.changePlayer = true;
                    DisableBall();
                }
                
            }

        }
    }

    public float GetPositionX(){
        return transform.position.x;
    }

    public float GetPositionY(){
        return transform.position.y;
    }

    public int GetBallColour(){
        return ballColour;
    }

    public int GetBallActive(){
        if(isActive){
            return 1;
        }
        else{
            return 0;
        }
    }

}
