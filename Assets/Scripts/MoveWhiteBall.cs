using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveWhiteBall : MonoBehaviour
{
    public Vector2 direction;
    public float initialVelocity;
    private Vector2 whiteBallSpawnPosition = new Vector2(-5.98f, 0f);

    private Rigidbody2D rb;

    public Environment env;

    private bool firstHit;

    private void Awake()
    {
        firstHit = true;
        rb = GetComponent<Rigidbody2D>();
    }

    public void Reset(){
        //Debug.Log("Reset White Ball");
        Rigidbody2D rb = GetComponent<Rigidbody2D>();
        rb.velocity = Vector2.zero;
        rb.angularVelocity = 0f;
        transform.position = whiteBallSpawnPosition;
    }
    public static Vector2 AngleToDirection(float angle)
    {
        // Convert angle from degrees to radians
        float radians = angle * Mathf.Deg2Rad;

        // Calculate the direction vector components
        float x = Mathf.Cos(radians);
        float y = Mathf.Sin(radians);

        // Create and return the direction vector
        return new Vector2(x, y);
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        // Check if the collided object has the "ball" tag.
        if (collision.gameObject.CompareTag("Ball"))
        {
            if(firstHit){
            
                BallScript ball = collision.gameObject.GetComponent<BallScript>();
                if(ball.GetBallColour() == env.currentPlayerColour){
                    env.UpdateReward(env.rewardPerCorrectBall);
                }
                else{
                    env.UpdateReward(env.rewardPerWrongBall);
                }
                firstHit = false;
            }
        }
    }

    public void CheckIfItHitReward(){
        //if it did not hit anything bad
        if(firstHit){
            env.UpdateReward(env.rewardPerNotHittingBall);
        }
    }

    public void MoveBall(float angle, float velocity){
        firstHit = true;
        Vector2 direction = AngleToDirection(angle);
        //Debug.Log(direction);
        Vector2 force = direction.normalized * velocity;
        rb.AddForce(force, ForceMode2D.Impulse);
    }
}
