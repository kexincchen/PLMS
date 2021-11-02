using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PostItMetaData : MonoBehaviour
{
    private int id;
    private string headerText;
    private string bodyText;

    private void Start()
    {
    }

    public int GetId()
    {
        return this.id;
    }

    public void SetId(int id)
    {
        this.id = id;
    }

    public string GetHeader()
    {
        return this.headerText;
    }

    public void SetHeader(string header)
    {
        this.headerText = header;
    }

    public string GetBody()
    {
        return this.bodyText;
    }

    public void SetBody(string body)
    {
        this.bodyText = body;
    }
}
