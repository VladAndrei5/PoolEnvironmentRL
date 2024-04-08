using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameLogic : MonoBehaviour
{
    public float gameSpeed = 0.1f;

    private void Start()
    {
        // Update the time scale based on the game speed
        SetGameSpeed(gameSpeed);
    }

    private void SetGameSpeed(float speed){
        Time.timeScale = speed;
    }

}
