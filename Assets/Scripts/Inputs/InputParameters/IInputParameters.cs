using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IInputParameters
{
    ///<summary> method <c>getPrimaryPointerPosition</c> returns a vector 3 position for the primary pointer input. Head pointer for HMDs and mouse pointer position for a mouse input</summary>
    Vector3 GetPrimaryPointerPosition();

    ///<summary> method <c>getPrimaryPointerPosition</c> returns a vector 3 position for the secondary pointer input.
    /// Only applicable when there is a secondary pointing input mechanism such as eye tracking gaze point</summary>
    Vector3 GetSecondaryPointerPosition();

    //wiil have to include select gestures such as pinch for hand tracking or button/mouse click for controller/mouse.
}
