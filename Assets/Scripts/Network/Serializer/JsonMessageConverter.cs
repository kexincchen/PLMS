using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;

public class JsonMessageConverter : JsonConverter
{
    public override bool CanConvert(Type objectType)
    {
        return (objectType == typeof(JsonMessage));
    }

    public override bool CanRead
    {
        get { return true; }
    }

    public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
    {
        JsonMessage jMessage = new JsonMessage();

        JObject jo = JObject.Load(reader);
        jMessage.messageType = (string)jo["messageType"];

        switch (jMessage.messageType)
        {
            case "TestEntity":
                jMessage.messageObject = jo["messageObject"].ToObject<TestEntity>();
                break;
            case "PostItContent":
                jMessage.messageObject = jo["messageObject"].ToObject<PostItContent>();
                break;
            case "PostItNumber":
                jMessage.messageObject = jo["messageObject"].ToObject<PostItNumber>();
                break;
            case "PostItVal":
                jMessage.messageObject = jo["messageObject"].ToObject<PostItVal>();
                break;
            case "PostItValList":
                jMessage.messageObject = jo["messageObject"].ToObject<PostItValList>();
                break;
            case "ActionMap":
                jMessage.messageObject = jo["messageObject"].ToObject<ActionMap>();
                break;
            case "ActionMapList":
                jMessage.messageObject = jo["messageObject"].ToObject<ActionMapList>();
                break;
            case "WhiteBoardPostItMap":
                jMessage.messageObject = jo["messageObject"].ToObject<WhiteBoardPostItMap>();
                break;
            case "WhiteBoardPostItMapList":
                jMessage.messageObject = jo["messageObject"].ToObject<WhiteBoardPostItMapList>();
                break;

        }

        return jMessage;

        /*
        JsonMessage jMessage = new JsonMessage();

        if(reader.TokenType == JsonToken.StartObject)
        {
            JObject jo = JObject.Load(reader);
            jMessage.messageType = (string)jo["messageType"];

            switch (jMessage.messageType)
            {
                case "TestEntity":
                    jMessage.messageObject = jo["messageObject"].ToObject<TestEntity>();
                    break;
                case "PostItContent":
                    jMessage.messageObject = jo["messageObject"].ToObject<PostItContent>();
                    break;
            }
        }
        if (reader.TokenType == JsonToken.StartArray)
        {
            JArray ja = JArray.Load(reader);
            switch (jMessage.messageType)
            {
                case "PostItContent":
                    ((PostItContent)jMessage.messageObject).topics = ja.ToObject<List<string>>();
                    break;
            }
        }
        return jMessage;
        */
    }

    public override bool CanWrite
    {
        get { return true; }
    }

    public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
    {
        throw new NotImplementedException();
        //serializer.Serialize(writer, value);
    }
}
