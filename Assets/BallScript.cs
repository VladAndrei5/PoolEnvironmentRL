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
    void Awake()
    {
        isActive = true;
        isMoving = false;
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
                    env.currentReward -= 100;
                    env.gameOver = true;
                }
                else if(ballColour == 3){
                    env.currentReward -= 10;
                    env.resetWhiteBall = true;
                }
                else if(env.currentPlayer == ballColour){
                    env.currentReward += 5;
                    DisableBall();
                }
                else if(env.currentPlayer != ballColour){
                    env.currentReward -= 10;
                    env.changePlayer = true;
                    DisableBall();
                }
                
            }

        }
    }

}
