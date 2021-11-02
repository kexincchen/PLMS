using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SceneSystem;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TutorialSceneAdder : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        IMixedRealitySceneSystem sceneSystem = MixedRealityToolkit.Instance.GetService<IMixedRealitySceneSystem>();
        sceneSystem.LoadContent("Tutorial");
        Destroy(this.gameObject);
    }

    
}
