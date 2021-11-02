using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Text;

public static class Serializer
{
    public static byte[] Serialize<T>(T data)
    {
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(data));
    }

    public static T Deserialize<T>(byte[] data)
    {
        return JsonConvert.DeserializeObject<T>(Encoding.UTF8.GetString(data), new JsonMessageConverter());
    }
}
