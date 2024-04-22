using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveWhiteBall : MonoBehaviour
{
    public Vector2 direction;
    public float initialVelocity;
    private Vector2 whiteBallSpawnPosition = new Vector2(-5.98f, 0f);

    private Rigidbody2D rb;

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    public void Reset(){
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

    public void MoveBall(float angle, float velocity){
        Vector2 direction = AngleToDirection(angle);
        //Debug.Log(direction);
        Vector2 force = direction.normalized * velocity;
        rb.AddForce(force, ForceMode2D.Impulse);
    }
}
