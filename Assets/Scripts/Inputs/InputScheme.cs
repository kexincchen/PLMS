using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputScheme : MonoBehaviour
{

    public PLMSSettings plmsSettings;
    private IInputParameters inputPLMS;      

    // Start is called before the first frame update
    void Start()
    {
        switch (this.plmsSettings.build)
        {
            case PLMSSettings.BuildType.PC:
                this.inputPLMS = new MouseInputParameters();
                break;
            case PLMSSettings.BuildType.AR:
                this.inputPLMS = new ARInputParameters();
                break;
            case PLMSSettings.BuildType.VR:
                this.inputPLMS = new VRInputParameters();
                break;
        }

        DontDestroyOnLoad(this.gameObject);
    }

    public IInputParameters getInputPLMS()
    {
        return this.inputPLMS;
    }
}
