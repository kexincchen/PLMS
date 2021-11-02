using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class PostItController : MonoBehaviour, IMixedRealityTouchHandler
{
    public GameObject header;
    public GameObject body;

    Image outlineImage;
    Outline outline;
    Outline headerOutline;
    Outline bodyOutline;

    //highlight outline color and size
    Color32 highlightColor = new Color32(0, 255, 0, 255);
    Vector2 highlightSize = new Vector2(4, 4);

    //highlight outline color and size
    Color32 gazeColor = new Color32(0, 0, 0, 255);
    Vector2 gazeSize = new Vector2(1, 1);

    //deprecated
    //Color highlightColor;
    //Color incogColor;

    PostItMetaData postItMetaData;

    public enum POST_IT_STATES {MIN = 0, MAX = 1, HIGHLIGHT = 2};


    /// <summary>
    /// pending action to be performed, this is the last action that was sent from the agent.
    /// if there was a previous action, that can be discarded
    /// if there is no action to be performed, this will have  value of -1
    /// else, it will have the value in the range of POST_IT_STATES
    /// </summary>
    private int pendingAction = -1;

    //wait to update state incase user is still viewing this post it
    private float actionWaitTime;

    //used to measure the dwellTimeRation
    // this is reset to 0f by the scene  controller after every obs is sent to the python script
    public float dwellTimeCurrentWindow = 0.0f;

    //used to store dwell time
    //NOT useful to send to NN unless total time elapsed is considered
    public float dwellTime = 0.0f;

    //used to store the amount of saccades into this post it per window
    // this is reset to 0f by the scene  controller after every obs is sent to the python script
    public int saccadeInCounterCurrentWindow = 0;

    //used to store the amount of saccades into this post it
    //NOT useful to send this to the NN 
    public int saccadeInCounter = 0;

    //variable states
    public PostItVal postItVal;

    private void Awake()
    {
        this.UpdateCollider();

        this.outlineImage = this.GetComponent<Image>();
        this.outline = this.GetComponent<Outline>();

        this.headerOutline = this.header.GetComponentInChildren<Outline>();
        this.bodyOutline = this.body.GetComponentInChildren<Outline>();

        //this.incogColor = new Color32(255, 189, 117, 255);
        //this.highlightColor = new Color32(243, 250, 61, 255);

        this.postItMetaData = this.GetComponent<PostItMetaData>();

        this.postItVal = new PostItVal();
        this.postItVal.currentPostItState = (int) POST_IT_STATES.MAX;
        this.postItVal.dwellTimeRatio = 0.0f;

        this.actionWaitTime = 0.0f;
    }

    // Start is called before the first frame update
    void Start()
    {
        this.postItMetaData.SetHeader(this.header.GetComponentInChildren<TextMeshProUGUI>().text);
        this.postItMetaData.SetBody(this.body.GetComponentInChildren<TextMeshProUGUI>().text);
    }

    // Update is called once per frame
    void Update()
    {
        // only minimize or maximize if the post it is not currently selected
        if(!this.postItVal.isSelected)
        {
            if (this.pendingAction != -1)
            {
                if (this.actionWaitTime > 0.0f)
                {
                    this.actionWaitTime -= Time.deltaTime;
                    return;
                }

                //might want to add an extra if to check if the current postit note is within the camera view
                switch (this.pendingAction)
                {
                    case (int)POST_IT_STATES.MIN:
                        this.Minimize();
                        this.pendingAction = -1;
                        break;
                    case (int)POST_IT_STATES.MAX:
                        this.Maximize();
                        this.pendingAction = -1;
                        break;
                    case (int)POST_IT_STATES.HIGHLIGHT:
                        this.Highlight();
                        this.pendingAction = -1;
                        break;
                }
            }
        }
    }

    public void SetPendingAction(int action)
    {
        this.pendingAction = action;
    }

    public void ChangeColor(Color color)
    {
        this.header.GetComponentInChildren<Image>().color = color;
        this.body.GetComponentInChildren<Image>().color = color;
    }
    public void HighlightOnly()
    {
        this.outline.enabled = true;
        this.outline.effectColor = this.highlightColor;
        this.outline.effectDistance = this.highlightSize;  
    }

    public void GoIncognitoOnly()
    {
        if (this.body.activeSelf)
        {
            this.outline.enabled = false;
        }
        else
        {
            this.headerOutline.enabled = false;
        }
        
    }

    public void MinimizeOnly()
    {
       
        this.body.SetActive(false);
        UpdateCollider();

        this.outlineImage.enabled = false;
        this.outline.enabled = false;
        this.headerOutline.enabled = false;
        

        /*
        this.body.SetActive(false);
        UpdateCollider();
        Bounds bound = this.gameObject.GetComponent<BoxCollider>().bounds;
        this.outlineImage.rectTransform.position = bound.center;
        this.outlineImage.rectTransform.sizeDelta = bound.size;
        */
    }

    public void MaximizeOnly()
    {
        this.body.SetActive(true);
        UpdateCollider();
        
        this.outlineImage.enabled = true;
        this.outline.enabled = false;
        this.headerOutline.enabled = false;

        /*
        this.body.SetActive(true);
        UpdateCollider();
        Bounds bound = this.gameObject.GetComponent<BoxCollider>().bounds;
        this.outlineImage.rectTransform.position = bound.center;
        this.outlineImage.rectTransform.sizeDelta = bound.size;
        */
    }

    public void Minimize()
    {
        this.MinimizeOnly();
        this.GoIncognitoOnly();
        this.postItVal.currentPostItState = (int) POST_IT_STATES.MIN;
    }

    public void Maximize()
    {
        this.MaximizeOnly();
        this.GoIncognitoOnly();
        this.postItVal.currentPostItState = (int)POST_IT_STATES.MAX;
    }

    public void Highlight()
    {
        this.MaximizeOnly();
        this.HighlightOnly();
        this.postItVal.currentPostItState = (int)POST_IT_STATES.HIGHLIGHT;
    }

    public void GazeHighlight()
    {
        if(this.postItVal.currentPostItState == (int)POST_IT_STATES.HIGHLIGHT || this.postItVal.isSelected)
        {
            return;
        }

        if (this.body.activeSelf)
        {
            this.outline.enabled = true;
            this.outline.effectColor = this.gazeColor;
            this.outline.effectDistance = this.gazeSize;
        }
        else
        {
            this.headerOutline.enabled = true;
            this.headerOutline.effectColor = this.gazeColor;
            this.headerOutline.effectDistance = this.gazeSize;
        }
           
    }

    public void GazeIncognito()
    {
        if (this.postItVal.currentPostItState == (int)POST_IT_STATES.HIGHLIGHT || this.postItVal.isSelected)
        {
            return;
        }
        this.GoIncognitoOnly();
    }




    //functions -  set body and header text which also updates postit meta data
    //function - set topics and id of meta data

    /// <summary>
    /// Sets both the post it id and header
    /// </summary>
    /// <param name="id">sets the id</param>
    /// <param name="header">sets the header text</param>
    public void SetIdHeader(int id, string header)
    {
        //this.header.GetComponentInChildren<TextMeshProUGUI>().text = id.ToString() + ". " + header;
        this.header.GetComponentInChildren<TextMeshProUGUI>().text = header;
        this.postItMetaData.SetId(id);
        this.postItMetaData.SetHeader(header);

        this.postItVal.id = id;
    }

    public void SetBody(string body)
    {
        this.body.GetComponentInChildren<TextMeshProUGUI>().text = body;
        this.postItMetaData.SetBody(body);
    }

    //Updates the box collider when the post-it note is minimized or maximized
    void UpdateCollider()
    {
        BoxCollider bx = this.gameObject.GetComponent<BoxCollider>();

        int childCount = 0;
        Bounds updatedBounds = new Bounds();

        foreach (RectTransform child in this.transform)
        {

            if(childCount == 0)
            {
                if (child.gameObject.activeSelf)
                {
                    updatedBounds = new Bounds(
                        this.transform.InverseTransformPoint(child.position),  //world position of child 
                        this.transform.InverseTransformSize(child.transform.TransformSize(child.rect.size)) // transform local size of child rect to world size
                        );
                }
            }
            else
            {
                if (child.gameObject.activeSelf)
                {
                    updatedBounds.Encapsulate(
                        new Bounds(
                            this.transform.InverseTransformPoint(child.position),  //world position of child 
                            this.transform.InverseTransformSize(child.transform.TransformSize(child.rect.size)) // transform local size of child rect to world size
                            )
                        );
                }
            }

            childCount++;
        }

        bx.center = updatedBounds.center;
        bx.size = updatedBounds.size;

       
        //increase the size.z of the Bounding box (to 0.01) so that the post-it note does not go through surfaces
        bx.size = new Vector3(bx.size.x, bx.size.y, 0.01f);
    }

    public int GetWhiteBoardID()
    {
        WhiteBoardController wb = this.gameObject.GetComponentInParent<WhiteBoardController>();
        if(wb != null)
        {
            return wb.GetId();
        }

        return -1;
    }

    public void OnTouchStarted(HandTrackingInputEventData eventData) { }

    public void OnTouchCompleted(HandTrackingInputEventData eventData)
    {
        if (this.postItVal.currentPostItState == (int) POST_IT_STATES.MIN)
        {
            //do not update current post it state as this calculate reward for MAX state instead of MIN state
            this.MaximizeOnly();
            this.ResetActionWaitTime();
        }
    }

    public void OnTouchUpdated(HandTrackingInputEventData eventData) { }

    public void ResetActionWaitTime()
    {
        this.actionWaitTime = 30.0f;
    }

    /*
     * Not as generalizable.
     * 
    void UpdateColliderAlternate()
    {
        BoxCollider bx = this.gameObject.GetComponent<BoxCollider>();

        Transform headerTransform = this.transform.Find("Header").transform;
        Transform bodyTransform = this.transform.Find("Body").transform;

        if (bodyTransform.gameObject.activeSelf)
        {
            bx.center = this.transform.InverseTransformPoint(this.transform.position);
            bx.size = this.transform.gameObject.GetComponent<RectTransform>().rect.size;
        }
        else
        {
            bx.center = this.transform.InverseTransformPoint(headerTransform.position);
            bx.size = this.transform.InverseTransformSize( // transform world size to local size (wrt to this.gameobject)
                    headerTransform.transform.TransformSize( //transforms local size to world size (wrt to header)  
                            headerTransform.gameObject.GetComponent<RectTransform>().rect.size //gets local size wrt to header
                        )
                );
        }
    }
    */
}
