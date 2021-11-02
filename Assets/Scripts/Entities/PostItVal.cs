public class PostItVal : AbsMessage
{
    public int id { get; set; }
    public bool isSelected { get; set; }
    public int currentPostItState { get; set; }
    public float dwellTimeRatio { get; set; }
    public float saccadeInRatio { get; set; }
}
