using Microsoft.MixedReality.Toolkit.Teleport;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WhiteBoardController : MonoBehaviour
{
    public bool isPrimary = false;

    private int id;

    public HashSet<GameObject> postIts;

    private SceneController sceneController;

    private Color attachedPostItColor;

    private void Awake()
    {
        this.sceneController = GameObject.Find("SceneController").GetComponent<SceneController>();
    }

    private void Start()
    {
        this.attachedPostItColor = this.sceneController.GetWBColor();
        this.postIts = new HashSet<GameObject>();
        this.id = this.sceneController.GetWhiteBoardLastID();
        this.sceneController.IncreaseWhiteBoardLastID();
        this.sceneController.whiteBoardObjs.Add(this.id, this.gameObject);
    }

    public void SetId(int id)
    {
        this.id = id;
    }

    public int GetId()
    {
        return this.id;
    }

    public void RemovePostIt()
    {
        HashSet<GameObject> temp = new HashSet<GameObject>(postIts);
        foreach(GameObject postIt in temp)
        {
            DetachPostIt(postIt);
        }
    }

    public void AttachPostIt(GameObject postIt)
    {
        postIt.transform.SetParent(this.transform);

        this.postIts.Add(postIt);

        PostItController postItController = postIt.GetComponent<PostItController>();

        postItController.MaximizeOnly();
        postItController.HighlightOnly();

        // Only for NON GARBAGE WHITEBOARD 
        // organize irrelevant items on garbage whiteboard
        if (!this.gameObject.tag.Equals("garbage_whiteboard"))
        {
            postItController.postItVal.isSelected = true;
        }
    }

    public void DetachPostIt(GameObject postIt)
    {
        Vector3 worldPos = postIt.transform.position;
        postIt.transform.SetParent(null);
        postIt.transform.position = worldPos;

        postIt.GetComponent<PostItController>().postItVal.isSelected = false;

        this.postIts.Remove(postIt);
    }

    public Color GetAttachedColor()
    {
        return this.attachedPostItColor;
    }

    private void OnDestroy()
    {
        this.sceneController.whiteBoardObjs.Remove(this.id);
        this.sceneController.FreeWBColor(this.attachedPostItColor);
    }
}
