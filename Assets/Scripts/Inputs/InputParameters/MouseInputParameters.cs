using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MouseInputParameters : IInputParameters
{
    /// <summary>
    /// Returns the mouse position in pixels
    /// </summary>
    /// <returns></returns>
    public Vector3 GetPrimaryPointerPosition()
    {
        return Input.mousePosition;
    }

    /// <summary>
    /// Since there is no secondary input such as gaze, this returns the mouse position in pixes
    /// </summary>
    /// <returns></returns>
    public Vector3 GetSecondaryPointerPosition()
    {
        return GetPrimaryPointerPosition();
    }
}
