using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NetworkSettings : MonoBehaviour
{
    public string remoteIP = "0.0.0.0";
    public string remotePort = "12344";
    public string listenIP = "0.0.0.0";
    public string listenPort = "12344";

    //dictates what messages to send based on the method
    public SceneController.METHOD method = SceneController.METHOD.RL;

    private void Start()
    {
        DontDestroyOnLoad(this.gameObject);
    }
}
