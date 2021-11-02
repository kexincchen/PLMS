using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PostItMinimizer : MonoBehaviour
{
    private PostItController postItController;

    private float timerToMinimize = 0f;

    // Start is called before the first frame update
    void Start()
    {
        this.postItController = this.gameObject.GetComponent<PostItController>();
        this.postItController.Minimize();
    }

    // Update is called once per frame
    void Update()
    {
        if(this.timerToMinimize > 5.0f)
        {
            this.postItController.Minimize();
            this.timerToMinimize = 0.0f;
        }

        if(this.postItController.body.activeSelf)
        {
            this.timerToMinimize += Time.deltaTime;
        }
        

    }
}
