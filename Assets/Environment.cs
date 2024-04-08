using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Net;
using System.Net.Sockets;

public class Environment : MonoBehaviour
{
    public MoveWhiteBall whiteBallControls;
    private (float, float) action;
    public float angle = 0.2f;
    //public float speed;
    public float maxSpeed = 20f;
    // value between 0 - 1
    private float power = 1f;

    void Start(){
        Step((angle, power));
    }

    private void Step((float, float) action){
        whiteBallControls.MoveBall(action.Item1, action.Item2 * maxSpeed);
    }

    private void ResetEnv(){
        SceneManager.LoadScene(0);
    }

}
