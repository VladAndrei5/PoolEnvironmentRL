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
        //transform.position = new Vector2(Random.Range(-9.8f, 9.8f), Random.Range(-4.8f, 4.8f));
        transform.position = whiteBallSpawnPosition;
    }

    /*
    public void Reset(){
        //Debug.Log("Reset White Ball");
        Rigidbody2D rb = GetComponent<Rigidbody2D>();
        rb.velocity = Vector2.zero;
        rb.angularVelocity = 0f;
        transform.position = whiteBallSpawnPosition;
    }
    */


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

    public float GetPositionX(){
        return transform.position.x;
    }

    public float GetPositionY(){
        return transform.position.y;
    }
    
    public void MoveBall(float xCord, float yCord, float velocity){
        //float angleRadians = Mathf.Atan2(yCord, xCord);
        float wbPositionX = transform.position.x;
        float wbPositionY = transform.position.y;
        firstHit = true;
        //Vector2 direction = AngleToDirection(angle);
        Vector2 direction = new Vector2(xCord - wbPositionX, yCord - wbPositionY);
        //Debug.Log(direction);
        Vector2 force = direction.normalized * velocity;
        rb.AddForce(force, ForceMode2D.Impulse);
    }
}
