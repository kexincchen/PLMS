using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class ActiveSceneManager : MonoBehaviour
{

    bool hasTutorialSceneActive = false; 

    // Update is called once per frame
    void Update()
    {
        if (SceneManager.GetSceneByName("Tutorial").isLoaded && !this.hasTutorialSceneActive)
        {
            SceneManager.SetActiveScene(SceneManager.GetSceneByName("Tutorial"));
            this.hasTutorialSceneActive = true;
        }

        if (!SceneManager.GetSceneByName("Tutorial").isLoaded && this.hasTutorialSceneActive)
        {
            SceneManager.SetActiveScene(SceneManager.GetSceneByName("Base"));
        }

    }
}
