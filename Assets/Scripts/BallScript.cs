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
        //transform.position = new Vector2(Random.Range(-9.8f, 9.8f), Random.Range(-4.8f, 4.8f));
        transform.position = new Vector3(originalPosition.x, originalPosition.y, transform.position.z);
        isActive = true;
        isMoving = false;
        Rigidbody2D rb = GetComponent<Rigidbody2D>();
        rb.velocity = Vector2.zero;
        rb.angularVelocity = 0f;
        GetComponent<SpriteRenderer>().enabled = true;
        GetComponent<CircleCollider2D>().enabled = true;
        GetComponent<Rigidbody2D>().simulated = true;
    }
    

    /*
    public void ResetBall(){
        transform.position = new Vector3(originalPosition.x, originalPosition.y, transform.position.z);
        isActive = true;
        isMoving = false;
        Rigidbody2D rb = GetComponent<Rigidbody2D>();
        rb.velocity = Vector2.zero;
        rb.angularVelocity = 0f;
        GetComponent<SpriteRenderer>().enabled = true;
        GetComponent<CircleCollider2D>().enabled = true;
        GetComponent<Rigidbody2D>().simulated = true;
    }
    */

    private void DisableBall(){
        isActive = false;
        Rigidbody2D rb = GetComponent<Rigidbody2D>();
        rb.velocity = Vector2.zero;
        rb.angularVelocity = 0f;
        GetComponent<SpriteRenderer>().enabled = false;
        GetComponent<CircleCollider2D>().enabled = false;
        GetComponent<Rigidbody2D>().simulated = false;

        env.gameOver = true;

        //DONT FORGET MAYBE TO REMOVE
        //transform.position = new Vector3(0f, 100f, 0f);
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        // Check if the collider that the rigid body entered is a trigger collider
        if (other.isTrigger)
        {   

            //Give rewards if ball falls in pocket based on its colour 
            if(isActive){
                env.UpdateReward(env.rewardPerWhiteBall);
                DisableBall();
                env.gameOver = true;
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
