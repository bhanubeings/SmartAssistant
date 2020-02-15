using System.Collections;
using System.Collections.Generic;
using System;
using System.Threading;
using UnityEngine;
using UnityEditor.VFX;


public class AvatarBehaviour : MonoBehaviour
{
  public float spawnRateMax;
  public float spawnRateNormal;
  public float fadeDelayTime;
  public int fadeIncrementValue;

  float spawnRateInterval;
  float spawnRateIncrement;
  string fromPy;

  UnityEngine.Experimental.VFX.VisualEffect visualEffect;
  string receiveData;

  void Start()
  {
    spawnRateInterval = spawnRateMax - spawnRateNormal;
    Debug.Log(spawnRateInterval);
    visualEffect = this.GetComponent<UnityEngine.Experimental.VFX.VisualEffect>();

    visualEffect.SetFloat("Spawn Rate", 0);
    StartCoroutine(FadeIn());
  }

  void Update()
  {
    fromPy = FindObjectOfType<Socket>().pubRecv;
    if (fromPy.StartsWith("$spawnRate#"))
    {
      spawnRateIncrement = float.Parse(fromPy.Split('#')[1]);
      if (spawnRateIncrement >= 0.5)
      {
        visualEffect.SetFloat("Spawn Rate", spawnRateNormal + spawnRateInterval * spawnRateIncrement);
      }
      if (spawnRateIncrement < 0.5)
      {
        visualEffect.SetFloat("Spawn Rate", spawnRateNormal);
      }
    }
    /* next to do list:
    1. receive amplitude/loudness of the audio ..done
    2. set a max & min threshold value,
       brighten the avatar base on % ..on going (need to add negative values)
    3. Sqlite on C#, url: http://zetcode.com/csharp/sqlite
    4. Able to finally edit the sql database from a clean
       UI in Unity.
    5. Can input commands and names of devices as well as
       the tagname of the devices!
    6. more to come...
    */
  }

  private IEnumerator FadeIn()
  {
    // delay before fading in, then play boot sound
    yield return new WaitForSeconds(fadeDelayTime);
    FindObjectOfType<AudioManager>().Play("Boot");

    for (int i=0; i <= spawnRateNormal; i+=fadeIncrementValue)
    {
      visualEffect.SetFloat("Spawn Rate", i);
      yield return new WaitForSeconds(0.001f);
    }
  }
}

