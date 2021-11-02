using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;

public class UDPController : MonoBehaviour
{
    private NetworkSettings networkSettings;
    public SceneController sceneController;

    List<PostItContent> postItContents;

#if !UNITY_EDITOR
    
    UDPSocket sendSocket;
    
    UDPSocket listenSocket;
    
#endif

    private void Awake()
    {
        this.networkSettings = GameObject.Find("NetworkSettings").GetComponent<NetworkSettings>();
    }

    // Start is called before the first frame update
    void Start()
    {
        this.postItContents = new List<PostItContent>();

#if !UNITY_EDITOR
        this.sendSocket = new UDPSocket(this.networkSettings);
        this.listenSocket = new UDPSocket(this.networkSettings);    
        this.listenSocket.Listen();

        //test
        /*
        this.SendStartMsg();
        
        JsonMessage jmTest = new JsonMessage();
        jmTest.messageType = "TestEntity";
        TestEntity te = new TestEntity();
        te.key = "k";
        te.value = 1;
        jmTest.messageObject = te;
        byte[] msg = Serializer.Serialize<JsonMessage>(jmTest);
        this.listenSocket.SendMessage(msg);      

        JsonMessage jmPC = new JsonMessage();
        jmPC.messageType = "PostItContent";
        PostItContent pc = new PostItContent();
        pc.id = 1;
        pc.clue = "what is this?";
        pc.header = "question";
        pc.topics = new List<string>();
        pc.topics.Add("one");
        pc.topics.Add("two");
        jmPC.messageObject = pc;
        byte[] msgPC = Serializer.Serialize<JsonMessage>(jmPC);
        this.listenSocket.SendMessage(msgPC); 
        

        JsonMessage jmTest = new JsonMessage();
        jmTest.messageType = "PostItValList";

        PostItVal postItVal = new PostItVal();
        postItVal.currentPostItState = (int)PostItController.POST_IT_STATES.MAX;
        postItVal.isSelected = false;
        postItVal.id = 1;
        postItVal.dwellTimeRatio = 0.56f;

        PostItVal postItVal1 = new PostItVal();
        postItVal1.currentPostItState = (int)PostItController.POST_IT_STATES.MIN;
        postItVal1.isSelected = false;
        postItVal1.id = 2;
        postItVal1.dwellTimeRatio = 0.86f;

        PostItValList pList = new PostItValList();
        pList.values = new List<PostItVal>();

        pList.values.Add(postItVal);
        pList.values.Add(postItVal1);

        jmTest.messageObject = pList;
        byte[] msg = Serializer.Serialize<JsonMessage>(jmTest);
        this.listenSocket.SendMessage(msg);   
        */
#endif
    }

    // Update is called once per frame
    void Update()
    {
        #if !UNITY_EDITOR

        this.CheckAndReadMessage();

        #endif
    }

#if !UNITY_EDITOR
    void CheckAndReadMessage()
    {
        if (this.listenSocket.MsgAvailable())
        {
            MessageReceivedHandler(this.listenSocket.ReceiveMsg());
        }
    }
#endif

    void MessageReceivedHandler(byte[] serializedMsg)
    {
        JsonMessage jm = Serializer.Deserialize<JsonMessage>(serializedMsg);
        if (jm.messageObject is TestEntity)
        {
            Debug.Log("key = " + ((TestEntity)jm.messageObject).key);
            Debug.Log("value = " + ((TestEntity)jm.messageObject).value.ToString());
        }
        if (jm.messageObject is PostItContent)
        {
            /*
            Debug.Log("id = " + ((PostItContent)jm.messageObject).id);
            Debug.Log("clue = " + ((PostItContent)jm.messageObject).clue);
            Debug.Log("header = " + ((PostItContent)jm.messageObject).header);
            */
            
            this.postItContents.Add((PostItContent)jm.messageObject);
        }
        if(jm.messageObject is PostItNumber)
        {
            if (this.postItContents.Count != ((PostItNumber)jm.messageObject).number)
            {
                StartCoroutine(this.sceneController.Quit("Received "+ this.postItContents.Count+"/"+ ((PostItNumber)jm.messageObject).number));
            }
        }
        if (jm.messageObject is PostItVal)
        {
            /*
            Debug.Log("id = " + ((PostItVal)jm.messageObject).id.ToString());
            Debug.Log("state = " + ((PostItVal)jm.messageObject).currentPostItState.ToString());
            */
        }
        if (jm.messageObject is PostItValList)
        {
            /*
            foreach(PostItVal val in ((PostItValList)jm.messageObject).values)
            {
                Debug.Log("id = " + val.id.ToString());
                Debug.Log("state = " + val.currentPostItState.ToString());
            }
            */
        }
        if (jm.messageObject is ActionMap)
        {
            this.sceneController.ExecuteAction((ActionMap)jm.messageObject);
        }
        if (jm.messageObject is ActionMapList)
        {
            this.sceneController.ExecuteActions((ActionMapList)jm.messageObject);
        }
        if (jm.messageObject is WhiteBoardPostItMap)
        {
            this.sceneController.ClusterColorPostIt((WhiteBoardPostItMap)jm.messageObject);
        }
        if (jm.messageObject is WhiteBoardPostItMapList)
        {
            this.sceneController.ClusterColorPostIts((WhiteBoardPostItMapList)jm.messageObject);
        }

    }

    /// <summary>
    /// Used to send the initial message to start the experience
    /// </summary>
    public void SendStartMsg()
    {
#if !UNITY_EDITOR
        JsonMessage jmStart = new JsonMessage();
        jmStart.messageType = "Start";
        jmStart.messageObject = null;
        byte[] msg = Serializer.Serialize<JsonMessage>(jmStart);
        this.listenSocket.SendMessage(msg);
#endif
    }

    /// <summary>
    /// Used to send the last message to end the experience and record the completion time
    /// </summary>
    public void SendEndMsg(float completionTime)
    {
#if !UNITY_EDITOR
        JsonMessage jm = new JsonMessage();
        jm.messageType = "Completed";

        CompletionTime ct = new CompletionTime();
        ct.completionTime = completionTime;
        
        jm.messageObject = ct;

        byte[] msg = Serializer.Serialize<JsonMessage>(jm);
        this.listenSocket.SendMessage(msg);
#endif
    }

    /// <summary>
    /// Sends a snapshot of the observation space.
    /// </summary>
    /// <param name="pList">List of all the variables related to the post it gameobjects</param>
    public void SendObsRL(PostItValList pList)
    {
#if !UNITY_EDITOR
        //sending the whole list does not seem to work as only a part of it is received on the script    
        
        /*
        JsonMessage jm = new JsonMessage();
        jm.messageType = "PostItValList";
        jm.messageObject = pList;
        byte[] msg = Serializer.Serialize<JsonMessage>(jm);
        this.listenSocket.SendMessage(msg);
        Debug.Log(Encoding.UTF8.GetString(msg));
        */

        foreach(PostItVal pVal in pList.values)
        {
            JsonMessage jm = new JsonMessage();
            jm.messageType = "PostItVal";
            jm.messageObject = pVal;
            byte[] msg = Serializer.Serialize<JsonMessage>(jm);
            this.listenSocket.SendMessage(msg);
        }


#endif
    }

    /// <summary>
    /// Sends a list of whiteboards with respectively attached post it notes
    /// </summary>
    /// <param name="wbMapList">list of (whiteboard id, postIt Id) representing the attached postIt to the whiteBoad</param>
    public void SendObsCluster(WhiteBoardPostItMapList wbMapList)
    {
#if !UNITY_EDITOR
        foreach(WhiteBoardPostItMap wbPostItMap in wbMapList.values)
        {
            JsonMessage jm = new JsonMessage();
            jm.messageType = "WhiteBoardPostItMap";
            jm.messageObject = wbPostItMap;
            byte[] msg = Serializer.Serialize<JsonMessage>(jm);
            this.listenSocket.SendMessage(msg);
        }

#endif
    }

    public List<PostItContent> GetPostItContents()
    {
        return this.postItContents;
    }

    public SceneController.METHOD getMethod()
    {
        return this.networkSettings.method;
    }
}
