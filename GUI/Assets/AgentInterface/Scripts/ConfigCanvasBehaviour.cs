using UnityEngine;
using System;
using System.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Samples.Helpers;
using Boomlagoon.JSON;


public class ConfigCanvasBehaviour : MonoBehaviour
{
  public Config config;
  string pathwayFile = "Assets/temp/pathway.txt";

  void Start()
  {
    string appPath = System.IO.File.ReadAllText(@pathwayFile);
    string configYML = System.IO.File.ReadAllText(@appPath+"\\config.yml");

    var readConfigYML = new StringReader(@configYML);

    var deserializer = new DeserializerBuilder().Build();
    var yamlObject = deserializer.Deserialize(readConfigYML);
    var serializerYML = new SerializerBuilder().JsonCompatible().Build();

    var configJSON = serializerYML.Serialize(yamlObject);
    Debug.Log(configJSON);

    JSONObject configJSONObject = JSONObject.Parse(configJSON);

    Debug.Log(configJSONObject);

    config.name = configJSONObject.GetString("name");
    config.version = configJSONObject.GetNumber("version");
    config.controllerIP = configJSONObject.GetString("controller_ip");
    config.SonosPort = configJSONObject.GetNumber("SONOS_port");
    config.trainHotword = configJSONObject.GetBoolean("hotword_train_bypass");
    config.trainHotwordAmount = configJSONObject.GetNumber("hotword_train_amount");
    config.userFirstName = configJSONObject.GetString("user_firstname");
    config.userLastName = configJSONObject.GetString("user_lastname");


    Debug.Log(config.version);
    Debug.Log(config.controllerIP);
  }
}
