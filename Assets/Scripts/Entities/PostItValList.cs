using System.Collections;
using System.Collections.Generic;

public class PostItValList : AbsMessage
{
    public List<PostItVal> values { get; set; }

    public PostItValList()
    {
        this.values = new List<PostItVal>();
    }
}
