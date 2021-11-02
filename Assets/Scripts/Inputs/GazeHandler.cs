using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class GazeHandler : MonoBehaviour
{
    GameObject eyeTarget;
    GameObject previousEyeTarget;
    GameObject headTarget;

    public GameObject secondaryWhiteBoardPrefab;
    public GameObject garbageWhiteBoardPrefab;
    public GameObject postItPrefab;

    //distance to instantiate white board
    public float distInstantiateWB = 1.2f;

    //distance to instantiate post its
    public float distInstantiatePostIt = 1f;

    public SceneController sceneController;

    //used to revert color back to previous color is Highlighting Eye target
    private PostItController previousEyePostIt;
    private Color previousEyeTargetColor;

    // Start is called before the first frame update
    void Start()
    {
        this.eyeTarget = null;
        this.previousEyeTarget = null;
    }

    // Update is called once per frame
    void Update()
    {
        if (!CoreServices.InputSystem.EyeGazeProvider.IsEyeTrackingEnabled)
        {
            Debug.Log("Eye tracking disabled");
            return;
        }

        if (!CoreServices.InputSystem.EyeGazeProvider.IsEyeTrackingDataValid)
        {
            Debug.Log("Eye tracking data invalid");
            return;
        }

        if (CoreServices.InputSystem.EyeGazeProvider.IsEyeCalibrationValid == null)
        {
            Debug.Log("Eye tracker calibration invalid");
            return;
        }

        //the previous gameobject that was gazed on
        //can be null
        this.previousEyeTarget = this.eyeTarget;

        //the gameobject currently gazed on.
        this.eyeTarget = CoreServices.InputSystem.EyeGazeProvider.GazeTarget;


        SetSaccadeInCounter();
        SetGazeDwellRatio();


        HighlightEyeGaze();
    }
    /// <summary>
    /// uses postItContents to instantiate Post It gameobjects and stores them in postItObjs
    /// </summary>
    /// <param name="postItContents"></param>
    /// <param name="postItObjs"></param>
    public Dictionary<int, GameObject> InstantiatePostIts(List<PostItContent> postItContents)
    {
        Dictionary<int, GameObject> postItObjs = new Dictionary<int, GameObject>();

        RaycastHit hitInfo;
        Vector3 position;

        foreach(PostItContent postItContent in postItContents)
        {
            //Debug.Log(postItContent.id + ":" + postItContent.clue);

            GameObject postIt;

            Ray ray = Camera.main.ViewportPointToRay(new Vector2(Random.Range(0.3f, 0.7f), Random.Range(0.3f, 0.7f)));
            if (Physics.Raycast(ray, out hitInfo, 100.0f))
            {
                position = Camera.main.transform.position + ray.direction * (hitInfo.distance - 0.1f);
                postIt = Instantiate(this.postItPrefab, position, Quaternion.identity);
            }
            else
            {
                position = new Vector3(Random.Range(0.3f, 0.7f), Random.Range(0.3f, 0.7f), this.distInstantiatePostIt);
                postIt = Instantiate(this.postItPrefab, Camera.main.ViewportToWorldPoint(position), Quaternion.identity);
                //postIt = Instantiate(this.postItPrefab, Camera.main.transform.position + Camera.main.transform.forward * this.distInstantiatePostIt, Quaternion.identity);
            }

            // Math works out but I know I'll forget so here's the link: https://answers.unity.com/questions/132592/lookat-in-opposite-direction.html
            // Note To Self: 2 * sWhiteBoard.transform.position - Camera.main.transform.position give a directions from origin
            postIt.transform.LookAt(2 * postIt.transform.position - Camera.main.transform.position);


            PostItController postItController = postIt.GetComponent<PostItController>();

            //set properties of post it
            postItController.SetIdHeader(postItContent.id, postItContent.header);
            postItController.SetBody(postItContent.clue);

            postItObjs.Add(postItContent.id, postIt);
        }

        return postItObjs;
    }

    public void CreateBoard()
    {
        GameObject sWhiteBoard;

        RaycastHit hitInfo;
        if (Physics.Raycast(Camera.main.transform.position, Camera.main.transform.forward, out hitInfo, 100.0f))
        {
             sWhiteBoard = Instantiate(this.secondaryWhiteBoardPrefab, Camera.main.transform.position + Camera.main.transform.forward * (hitInfo.distance - 0.2f), Quaternion.identity);
        }
        else
        {
            sWhiteBoard = Instantiate(this.secondaryWhiteBoardPrefab, Camera.main.transform.position + Camera.main.transform.forward * this.distInstantiateWB, Quaternion.identity);
        }

        // Math works out but I know I'll forget so here's the link: https://answers.unity.com/questions/132592/lookat-in-opposite-direction.html
        // Note To Self: 2 * sWhiteBoard.transform.position - Camera.main.transform.position give a directions from origin
        sWhiteBoard.transform.LookAt(2 * sWhiteBoard.transform.position - Camera.main.transform.position);
    }

    public void CreateGarbageBoard()
    {
        GameObject gWhiteBoard;

        RaycastHit hitInfo;
        if (Physics.Raycast(Camera.main.transform.position, Camera.main.transform.forward, out hitInfo, 100.0f))
        {
            gWhiteBoard = Instantiate(this.garbageWhiteBoardPrefab, Camera.main.transform.position + Camera.main.transform.forward * (hitInfo.distance - 0.2f), Quaternion.identity);
        }
        else
        {
            gWhiteBoard = Instantiate(this.garbageWhiteBoardPrefab, Camera.main.transform.position + Camera.main.transform.forward * this.distInstantiateWB, Quaternion.identity);
        }

        // Math works out but I know I'll forget so here's the link: https://answers.unity.com/questions/132592/lookat-in-opposite-direction.html
        // Note To Self: 2 * sWhiteBoard.transform.position - Camera.main.transform.position give a directions from origin
        gWhiteBoard.transform.LookAt(2 * gWhiteBoard.transform.position - Camera.main.transform.position);
    }

    public void DestroyBoard()
    {
        //Can only destroy white boards
        //Post-it notes cannot be destroyed
        if (this.eyeTarget!=null && (this.eyeTarget.tag.Equals("white_board") || this.eyeTarget.tag.Equals("garbage_whiteboard")) && !this.eyeTarget.GetComponent<WhiteBoardController>().isPrimary)
        {
            this.eyeTarget.GetComponent<WhiteBoardController>().RemovePostIt();
            Destroy(this.eyeTarget);
        }

        if(this.eyeTarget != null && this.eyeTarget.tag.Equals("post_it"))
        {
            if(this.eyeTarget.transform.parent != null &&
                (this.eyeTarget.tag.Equals("white_board") || this.eyeTarget.tag.Equals("garbage_whiteboard")) &&
                !this.eyeTarget.transform.parent.GetComponent<WhiteBoardController>().isPrimary)
            {

                //since whiteboard will be removed as the parent, we need to store it to destroy it.
                GameObject whiteBoard = this.eyeTarget.transform.parent.gameObject;

                this.eyeTarget.transform.parent.GetComponent<WhiteBoardController>().RemovePostIt();
                
                Destroy(whiteBoard);
            }
        }
    }

    void SetSaccadeInCounter()
    {
        if (this.eyeTarget != null && this.eyeTarget.tag.Equals("post_it") && this.eyeTarget != this.previousEyeTarget)
        {
            PostItController postIt = this.eyeTarget.GetComponentInParent<PostItController>();
            if (postIt != null)
            {
                postIt.saccadeInCounterCurrentWindow += 1;
                postIt.saccadeInCounter += 1;

                //increments the total saccdeIn counter
                //this is used to calculate the ratio of saccades into a post it note based on the total number of saccade ins on all post it notes
                this.sceneController.incrementTotalSaccadeIns();
            }
        }
    }
    void SetGazeDwellRatio()
    {
        if (this.eyeTarget != null && this.eyeTarget.tag.Equals("post_it"))
        {
            PostItController postIt = this.eyeTarget.GetComponentInParent<PostItController>();
            if (postIt != null)
            {
                postIt.dwellTimeCurrentWindow += Time.deltaTime;
                postIt.dwellTime += Time.deltaTime;
                //set action wait time so that the agent does not update this post its state when the user just looked at it
                postIt.ResetActionWaitTime();
            }
        }
    }

    void HighlightEyeGaze()
    {
        if (this.eyeTarget != null)
        {
            if (this.previousEyePostIt != null)
            {
                if (this.eyeTarget != this.previousEyePostIt.gameObject)
                {
                    this.previousEyePostIt.GazeIncognito();
                    this.previousEyePostIt = null;
                }

            }

            PostItController postIt = this.eyeTarget.GetComponentInParent<PostItController>();
            if (postIt != null)
            {
                this.previousEyePostIt = postIt;

                postIt.GazeHighlight();
            }
        }      
    }
}
