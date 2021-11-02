using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class SceneController : MonoBehaviour
{
    object _whiteBoardIDLock;

    public Dictionary<Color, bool> whiteBoardColors;

    public enum METHOD { AR, RL, CLUSTER, RL_CLUSTER };

    private METHOD method;

    public UDPController udpController;
    public GazeHandler gazeHandler;

    //counts the number of whiteboards
    //always starts with 1 as there will always be a primary whiteboard
    private int whiteBoardLastID;
    
    // this doesn't seem to work in Release but works in Debug
    public GameObject TimerPrefab;
    GameObject timerObj;
    // waits for waitTime to get all the postIt content from the python server.
    private float timer = 0.0f;
    private float waitTime = 10.0f;

    private bool isStarted = false;
    private bool isDone = false;

    Dictionary<int,GameObject> postItObjs;
    public Dictionary<int, GameObject> whiteBoardObjs;

    //the initial wait time to allow the user to read and organize notes
    public float initWaitTime = 60.0f;

    // the time after which an observation is sent to the agent
    public float ObsTimeWindow = 10.0f;

    //keeps track of the total saccade ins on post it notes for this time window
    private int totalSaccadeInCount = 0;



    //completion time recording
    private float completionTime = 0.0f;

    public void IncreaseWhiteBoardLastID()
    {
        this.whiteBoardLastID++;
    }

    public int GetWhiteBoardLastID()
    {
        return this.whiteBoardLastID;
    }

    private void Awake()
    {
        this.SetWBColors();

        this.whiteBoardLastID = 1;
        this.postItObjs = new Dictionary<int, GameObject>();
        this.whiteBoardObjs = new Dictionary<int, GameObject>();
    }

    private void Start()
    {
        if(this.udpController != null) //only in the tutorial
        {
            this.method = this.udpController.getMethod();
        }
        
    }

    // Update is called once per frame
    void Update()
    {
        if (!this.isDone)
        {
            if (this.isStarted && (this.timer < this.waitTime))
            {
                this.timer += Time.deltaTime;

                //show timer to user
                this.timerObj.GetComponent<PostItController>().SetBody(Mathf.Round(this.waitTime - this.timer).ToString());
            }
            else
            {
                if (this.timer >= this.waitTime)
                {
                    if (this.timerObj)
                    {
                        GameObject.Destroy(this.timerObj);
                    }

                    //reset timer to be used for ObsTimeWindow
                    this.ResetTimer();

                    //TODO 
                    // send number of post its from python scripts 
                    // check if number received  == this.udpController.GetPostItContents().Count

                    //gets called over and over again
                    this.postItObjs = this.gazeHandler.InstantiatePostIts(this.udpController.GetPostItContents());

                    this.isDone = true;
                }
            }
        }
        else
        {
            //keep updating completion time until speech command indicates "Completed"
            this.completionTime += Time.deltaTime;

            // TOTAL Initial wait time is actually  = initWaitTime + ObsTimeWindow
            if(this.initWaitTime > 0.0f)
            {
                this.initWaitTime -= Time.deltaTime;
            }
            else 
            {
                if (this.timer < this.ObsTimeWindow)
                {
                    this.timer += Time.deltaTime;
                }
                else
                {
                    /*
                     * This is now handled in the python scripts receiving the messages
                    //IF simple clustering, send for each whiteboard a map{whiteboardID, postitID}
                    if(this.method == METHOD.CLUSTER)
                    {
                        this.udpController.SendObsCluster(this.GetWhiteBoardPostItList(this.postItObjs));
                    }

                    // IF RL, send postit's selected and states and dwell times
                    if(this.method == METHOD.RL)
                    {
                        this.udpController.SendObsRL(this.GetPostItVariableList(this.postItObjs));
                    }

                    if(this.method == METHOD.RL_CLUSTER)
                    {
                        this.udpController.SendObsCluster(this.GetWhiteBoardPostItList(this.postItObjs));
                        this.udpController.SendObsRL(this.GetPostItVariableList(this.postItObjs));
                    }
                    */

                    this.udpController.SendObsCluster(this.GetWhiteBoardPostItList(this.postItObjs));
                    this.udpController.SendObsRL(this.GetPostItVariableList(this.postItObjs));

                    this.ResetTimer();
                    this.ResetTotalSaccadeInCount();
                }
            }
        }
    }

    private void SetWBColors()
    {
        this.whiteBoardColors = new Dictionary<Color, bool>();

        this.whiteBoardColors.Add(new Color32(0, 175, 255, 255), false);
        this.whiteBoardColors.Add(new Color32(179, 160, 160, 255), false);
        this.whiteBoardColors.Add(new Color32(197, 96, 255, 255), false);
        this.whiteBoardColors.Add(new Color32(215, 155, 91, 255), false);
        this.whiteBoardColors.Add(new Color32(255, 75, 181, 255), false);
        this.whiteBoardColors.Add(new Color32(66, 177, 66, 255), false);
        this.whiteBoardColors.Add(new Color32(255, 30, 0, 255), false);
        this.whiteBoardColors.Add(new Color32(207, 255, 220, 255), false);
    }

    public Color GetWBColor()
    {
        Color color = Random.ColorHSV(0f, 1f, 1f, 1f, 0.75f, 1f);
           
        foreach(KeyValuePair<Color,bool> entry in this.whiteBoardColors)
        {
            if (!entry.Value)
            {
                color = entry.Key;
                this.whiteBoardColors[entry.Key] = true;
                break;
            }
        }

        return color;

    }

    public void FreeWBColor(Color key)
    {
        this.whiteBoardColors[key] = false;
    }

    private PostItValList GetPostItVariableList(Dictionary<int, GameObject> postItObjs)
    {
        PostItValList pList = new PostItValList();
        foreach(KeyValuePair<int, GameObject> entry in postItObjs)
        {
            pList.values.Add(this.GetPostItVariable(entry.Value));
        }

        return pList;
    }

    private PostItVal GetPostItVariable(GameObject postIt)
    {
        PostItController postItController = postIt.GetComponent<PostItController>();
        PostItVal pVal = postItController.postItVal;

        //calculate the dwell time ratio based on the postIt dwell time and the Observation Time Window
        pVal.dwellTimeRatio = postItController.dwellTimeCurrentWindow / this.ObsTimeWindow;

        // the number of saccadeIns for a post it note divided by the total saccadeIn for all post it notes
        if (this.totalSaccadeInCount != 0)
        {
            pVal.saccadeInRatio = (float)postItController.saccadeInCounterCurrentWindow / (float)this.totalSaccadeInCount;
        }
        else
        {
            pVal.saccadeInRatio = 0;
        }
        

        //reset dwell time  of post it for next window
        postItController.dwellTimeCurrentWindow = 0.0f;

        //reset saccade in of post it for next window
        postItController.saccadeInCounterCurrentWindow = 0;

        return pVal;
    }

    //in case of clustering technique
    private WhiteBoardPostItMapList GetWhiteBoardPostItList(Dictionary<int, GameObject> postItObjs)
    {
        WhiteBoardPostItMapList wbPostItList = new WhiteBoardPostItMapList();
        wbPostItList.values = new List<WhiteBoardPostItMap>();
        foreach (KeyValuePair<int, GameObject> entry in postItObjs)
        {
            PostItController postItController = entry.Value.GetComponent<PostItController>();
            WhiteBoardPostItMap wbPI = new WhiteBoardPostItMap();
            wbPI.whiteBoardID = postItController.GetWhiteBoardID();
            wbPI.postItID = entry.Key;
            wbPostItList.values.Add(wbPI);
        }

        return wbPostItList;
    }

    public void StartExperience()
    {
        if (!this.isStarted)
        {
            this.udpController.SendStartMsg();
            this.isStarted = true;
            this.timerObj = Instantiate(TimerPrefab, Camera.main.transform.position + Camera.main.transform.forward * (0.6f), Quaternion.identity);
        }
    }

    public void EndExperience()
    {
            this.udpController.SendEndMsg(this.completionTime);
            //maybe change scene
    }

    private void ResetTimer()
    {
        this.timer = 0.0f;
    }

    public void incrementTotalSaccadeIns()
    {
        this.totalSaccadeInCount++;
    }

    private void ResetTotalSaccadeInCount()
    {
        this.totalSaccadeInCount = 0;
    }

    public void ExecuteAction(ActionMap aMap)
    {
        this.postItObjs[aMap.id].GetComponent<PostItController>().SetPendingAction(aMap.action);
    }
    public void ExecuteActions(ActionMapList aMapList)
    {
        foreach(ActionMap aMap in aMapList.values)
        {
            this.ExecuteAction(aMap);
        }
    }

    public void ClusterColorPostIt(WhiteBoardPostItMap wbPostItMap)
    {
        PostItController postItController = this.postItObjs[wbPostItMap.postItID].GetComponent<PostItController>();
        WhiteBoardController whiteBoardController = this.whiteBoardObjs[wbPostItMap.whiteBoardID].GetComponent<WhiteBoardController>();
        postItController.ChangeColor(whiteBoardController.GetAttachedColor());
    }

    public void ClusterColorPostIts(WhiteBoardPostItMapList wbPostItMapList)
    {
        foreach(WhiteBoardPostItMap wbPostItMap in wbPostItMapList.values)
        {
            this.ClusterColorPostIt(wbPostItMap);
        }
    }

    public SceneController.METHOD checkMethod()
    {
        return this.method;
    }

    public IEnumerator Quit(string error)
    {
        Destroy(this.timerObj);

        GameObject errorObject = Instantiate(TimerPrefab, Camera.main.transform.position + Camera.main.transform.forward * (0.6f), Quaternion.identity);
        errorObject.GetComponent<PostItController>().SetBody(error);
        yield return new WaitForSeconds(5f);

        Application.Quit();

    }
}
