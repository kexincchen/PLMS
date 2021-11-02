using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SceneSystem;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class TutorialSceneController : MonoBehaviour
{
    async public void EndTutorial()
    {
        IMixedRealitySceneSystem sceneSystem = MixedRealityToolkit.Instance.GetService<IMixedRealitySceneSystem>();
        await sceneSystem.UnloadContent("Tutorial");
        await sceneSystem.LoadContent("Opening");
        
    }
}
