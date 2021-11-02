using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestBehaviours : MonoBehaviour
{
    float timer = 0.0f;
    int count = 0;
    PostItController postItController;
    // Start is called before the first frame update
    void Start()
    {
        this.postItController = this.gameObject.GetComponent<PostItController>();
    }

    // Update is called once per frame
    void Update()
    {
        this.timer += Time.deltaTime;
        if(this.timer > 5.0f & this.postItController.postItVal.currentPostItState != (int)PostItController.POST_IT_STATES.MIN & count == 0)
        {
            Debug.Log("Min");
            this.postItController.Minimize();
            count++; 
        }
        if(this.timer > 10.0f & this.postItController.postItVal.currentPostItState != (int)PostItController.POST_IT_STATES.MAX & count == 1)
        {
            Debug.Log("Max");
            this.postItController.Maximize();
            count++;
        }
        if(this.timer > 15.0f & this.postItController.postItVal.currentPostItState != (int)PostItController.POST_IT_STATES.HIGHLIGHT & count ==2)
        {
            Debug.Log("Highlight");
            this.postItController.Highlight();
            count++;
        }
        if (this.timer > 20.0f & this.postItController.postItVal.currentPostItState != (int)PostItController.POST_IT_STATES.MIN & count ==3)
        {
            Debug.Log("Min");
            this.postItController.Minimize();
            count++;
        }
        if (this.timer > 25.0f & this.postItController.postItVal.currentPostItState != (int)PostItController.POST_IT_STATES.HIGHLIGHT & count == 4)
        {
            Debug.Log("Highlight");
            this.postItController.Highlight();
            count++;
        }
    }
}
