using Microsoft.MixedReality.Toolkit.Input;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class Keyboard : MonoBehaviour, IMixedRealityTouchHandler, IMixedRealityPointerHandler
{
    TouchScreenKeyboard keyboard;

    public void OnPointerClicked(MixedRealityPointerEventData eventData)
    {
        this.OpenSystemKeyboard();
    }

    public void OnPointerDown(MixedRealityPointerEventData eventData)
    {
 
    }

    public void OnPointerDragged(MixedRealityPointerEventData eventData)
    {
 
    }

    public void OnPointerUp(MixedRealityPointerEventData eventData)
    {
       
    }

    public void OnTouchCompleted(HandTrackingInputEventData eventData)
    {
        this.GetComponent<TMP_InputField>().Select();
        //this.GetComponent<TMP_InputField>().ActivateInputField();
        this.OpenSystemKeyboard();
    }

    public void OnTouchStarted(HandTrackingInputEventData eventData)
    {
        
    }

    public void OnTouchUpdated(HandTrackingInputEventData eventData)
    {
        
    }

    public void OpenSystemKeyboard()
    {
        keyboard = TouchScreenKeyboard.Open("", TouchScreenKeyboardType.Default, false, false, false, false);
    }

    void Update()
    {
        /*
        if (this.keyboard != null)
        {
            if (this.keyboard.active)
            {
                this.GetComponent<TMP_InputField>().text = this.keyboard.text;
            }
        }
        */
    }
}
