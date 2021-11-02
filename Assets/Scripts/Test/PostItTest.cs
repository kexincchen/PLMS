using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PostItTest : MonoBehaviour
{
    float timer = 0.0f;
    bool done = false;

    void Awake()
    {
        
    }

    // Start is called before the first frame update
    void Start()
    {
        this.gameObject.GetComponent<PostItController>().Minimize();
    }

    // Update is called once per frame
    void Update()
    {
        if (!done)
        {
            if (timer > 5.0f)
            {
                this.gameObject.GetComponent<PostItController>().Maximize();
            }

            if (timer > 10.0f)
            {
                this.gameObject.GetComponent<PostItController>().Highlight();
            }

            if (timer > 15.0f)
            {
                this.gameObject.GetComponent<PostItController>().Minimize();
                done = true;
            }

            timer += Time.deltaTime;
        }

        
    }
}
