using Microsoft.MixedReality.Toolkit.UI;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using TMPro;
using UnityEngine;

public class KinematicAdapter : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        UpdatePostItRB();
    }

    void UpdatePostItRB()
    {
        Transform parent = this.transform.parent;

        if (parent != null && (parent.tag.Equals("white_board") || parent.tag.Equals("garbage_whiteboard")) && !parent.GetComponent<Rigidbody>().isKinematic)
        {
            Rigidbody rb = this.GetComponent<Rigidbody>();
            if (rb != null)
            {
                Destroy(rb);
                Destroy(this.GetComponent<ObjectManipulator>());
            }
        }
        else
        {
            Rigidbody rb = this.GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = this.gameObject.AddComponent<Rigidbody>();
                rb.isKinematic = true;
                rb.useGravity = false;

                this.gameObject.AddComponent<ObjectManipulator>();
            }
        }
    }

    /*
    void UpdatePostItRB()
    {
        Transform parent = this.transform.parent;

        if (parent != null && parent.tag.Equals("white_board"))
        {
            Rigidbody parentRB = parent.GetComponent<Rigidbody>();
            if (parentRB.isKinematic)
            {
                Rigidbody rb = this.GetComponent<Rigidbody>();
                if (rb == null)
                {
                    rb = this.gameObject.AddComponent<Rigidbody>();
                    rb.isKinematic = true;
                    rb.useGravity = false;

                    this.gameObject.AddComponent<ObjectManipulator>();
                }
            }
            else
            {
                Rigidbody rb = this.GetComponent<Rigidbody>();
                if (rb != null)
                {
                    Destroy(rb);
                    Destroy(this.GetComponent<ObjectManipulator>());
                }
            }
        }
    }
    */
}
