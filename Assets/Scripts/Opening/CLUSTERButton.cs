using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.SceneSystem;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class CLUSTERButton : MonoBehaviour, IMixedRealityTouchHandler
{
    Button button;
    public NetworkSettings networkSettings;
    public TMP_InputField inputField;

    void Awake()
    {
        this.button = this.gameObject.GetComponent<Button>();
    }

    public void HandleButtonClick()
    {
        Regex ip = new Regex(@"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b");
        if (ip.IsMatch(this.inputField.text.Trim()))
        {
            this.networkSettings.remoteIP = this.inputField.text.Trim();
        }
        else
        {
            if (!this.inputField.text.Trim().Equals(""))
            {
                this.inputField.text = "";
                return;
            }
        }

        this.networkSettings.method = SceneController.METHOD.CLUSTER;

        this.LoadScene("main");
    }

    private async void LoadScene(string sceneName)
    {
        IMixedRealitySceneSystem sceneSystem = MixedRealityToolkit.Instance.GetService<IMixedRealitySceneSystem>();

        //this should unload the current scene but does not seem to work
        await sceneSystem.LoadContent(sceneName, LoadSceneMode.Single);
        await sceneSystem.UnloadContent("Opening");
    }

    public void OnTouchCompleted(HandTrackingInputEventData eventData)
    {
        ExecuteEvents.Execute<IPointerClickHandler>(this.button.gameObject, new PointerEventData(EventSystem.current), ExecuteEvents.pointerClickHandler);
    }

    public void OnTouchStarted(HandTrackingInputEventData eventData)
    {
        
    }

    public void OnTouchUpdated(HandTrackingInputEventData eventData)
    {
        
    }
}
