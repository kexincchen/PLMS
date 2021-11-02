using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class PointerEventLogger : MonoBehaviour
{
    GraphicRaycaster m_Raycaster;
    PointerEventData m_PointerEventData;
    EventSystem m_EventSystem;

    //the input type we are using
    InputScheme inputScheme;

    // Start is called before the first frame update
    void Start()
    {
        this.m_Raycaster = this.GetComponent<GraphicRaycaster>();
        this.m_EventSystem = EventSystem.current;
        this.inputScheme = GameObject.Find("InputScheme").GetComponent<InputScheme>();
    }

    // Update is called once per frame
    void Update()
    {
        //Set up the new Pointer Event
        m_PointerEventData = new PointerEventData(m_EventSystem);
        //Set the Pointer Event Position to that of the mouse position
        m_PointerEventData.position = inputScheme.getInputPLMS().GetSecondaryPointerPosition();

        //Create a list of Raycast Results
        List<RaycastResult> results = new List<RaycastResult>();

        //Raycast using the Graphics Raycaster and mouse position
        m_Raycaster.Raycast(m_PointerEventData, results);

        //For every result returned, output the name of the GameObject on the Canvas hit by the Ray
        foreach (RaycastResult result in results)
        {
            Debug.Log("Hit " + result.gameObject.GetComponentInParent<PostItMetaData>().GetHeader());
        }
    }
}
