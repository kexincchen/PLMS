using System.Collections.Generic;

public class PostItContent : AbsMessage
{
    public int id { get; set; }
    public string clue { get; set; }
    public string header { get; set; }
    public List<string> topics { get; set; }
}
