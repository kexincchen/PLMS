using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PLMSSettings : MonoBehaviour
{
  public enum BuildType
    {
        PC, 
        AR, 
        VR
    };

    public BuildType build = BuildType.PC;
}
