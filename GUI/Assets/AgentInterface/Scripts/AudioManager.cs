using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;


public class AudioManager : MonoBehaviour
{
  public Audio[] audios;
  // string audioFile = "Assets/temp/audio.txt";

  void Awake()
  {
    foreach (Audio a in audios)
    {
      a.source = gameObject.AddComponent<AudioSource>();
      a.source.clip = a.clip;
      a.source.volume = a.volume;
    }
  }

  public void Play (string name)
  {
    Audio a = Array.Find(audios, audio => audio.name == name);
    a.source.Play();
  }
}
