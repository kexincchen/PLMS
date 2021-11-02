using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if !UNITY_EDITOR
using System;
using System.Threading;
using System.Threading.Tasks;
#endif

public class UDPControllerThread : MonoBehaviour
{
    public NetworkSettings networkSettings;

#if !UNITY_EDITOR
    
    UDPSocket sendSocket;
    
    UDPSocket listenSocket;
    Task listenTask;
    CancellationTokenSource tokenSource;
    CancellationToken token;
    
#endif

    // Start is called before the first frame update
    void Start()
    {
        #if !UNITY_EDITOR
        this.sendSocket = new UDPSocket(this.networkSettings);
        this.listenSocket = new UDPSocket(this.networkSettings);    
        this.StartListening();
        #endif
    }

    // Update is called once per frame
    void Update()
    {
     
    }

#if !UNITY_EDITOR
    public void StartListening()
    {
        this.tokenSource = new CancellationTokenSource();
        this.token = tokenSource.Token;

        this.listenTask = new Task(()=>
            {
                try
                {
                    this.InitiateListening(token);
                }catch(OperationCanceledException)
                {
                    Debug.Log("Cancelled");
                }
                
            });
        this.listenTask.Start();
    }

    /// <summary>
    /// helper function to be run on a separate thread to continuously listen
    /// </summary>
    public void InitiateListening(CancellationToken cancellationToken)
    {
        Debug.Log("Initiated Listening");

        this.listenSocket.Listen();
        while(true)
        {
            if(cancellationToken.IsCancellationRequested)
            {
                return;
            }

            if(this.listenSocket.MsgAvailable())
            {
                this.MessageReceivedCallback(this.listenSocket.ReceiveMsg());
            }
        }
}

    /// <summary>
    /// Aborts listening thread
    /// </summary>
    public void StopListening()
    {
        this.tokenSource.Cancel();
    }
#endif

    void MessageReceivedCallback(byte[] serializedMsg)
    {
        
    }
}
