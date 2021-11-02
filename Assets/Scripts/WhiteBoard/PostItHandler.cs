using Microsoft.MixedReality.Toolkit.UI;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PostItHandler : MonoBehaviour
{

    public WhiteBoardController whiteBoardController;

    // Start is called before the first frame update
    void Start()
    {
        //this.gameObject.tag = "white_board";
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    
    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.tag.Equals("post_it"))
        {
            this.whiteBoardController.AttachPostIt(other.gameObject);
        }
    }

    /*
     *does not work as once the rigidbody is removed, this event will not trigger anymore.
     *
    private void OnTriggerStay(Collider other)
    {
        Rigidbody rb = this.transform.parent.GetComponent<Rigidbody>();
        Rigidbody otherRB = other.GetComponent<Rigidbody>();

        //if this object is being manipulated
        //kinematic turns off
        if (!rb.isKinematic)
        {
            if (otherRB != null)
            {
                Destroy(otherRB);
                Destroy(other.GetComponent<ObjectManipulator>());
            }
        }
        //if object is stationary
        else
        {
            if(otherRB == null)
            {
                otherRB = this.gameObject.AddComponent<Rigidbody>();
                otherRB.isKinematic = true;
                otherRB.useGravity = false;

                other.gameObject.AddComponent<ObjectManipulator>();
            }
        }
    }
    */

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.tag.Equals("post_it"))
        {
            this.whiteBoardController.DetachPostIt(other.gameObject);
        }
    }
}
