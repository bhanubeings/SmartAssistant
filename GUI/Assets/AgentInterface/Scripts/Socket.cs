using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

using System.Text;
using System.IO;
using System.Net;
using System.Net.Sockets;

using System.Threading;


public class Socket : MonoBehaviour
{
  public int port;
  public string hostname;
  public string pubRecv;
  TcpClient client;
  NetworkStream stream;
  Thread socketThread;

  // send var
  byte[] sendData;

  // receive var
  byte[] receiveData;
  string responseData;
  int receiveBytes;

  void Awake()
  {
    Debug.Log("Starting thread...");
    socketThread = new Thread(Run);
    socketThread.Start();
  }

  void Run()
  {
    Debug.Log("Running thread...");
    try
    {
      client = new TcpClient(hostname, port);
      stream = client.GetStream();
      Debug.Log("Connection made!");
    }
    catch(System.Net.Sockets.SocketException)
    {
      Debug.Log("Connection failed...");
    }

    while(true)
    {
      pubRecv = Receive();
      Debug.Log("Received message: " + pubRecv);
    }
  }

  public void Send(string message)
  {
    try
    {
      sendData = Encoding.ASCII.GetBytes(message);
      stream.Write(sendData, 0, sendData.Length);

      Debug.Log("Sending message: " + message);
    }
    catch(System.NullReferenceException)
    {
      Debug.Log("Failed to send data...");
    }
  }

  public string Receive()
  {
    try
    {
      receiveData = new Byte[1024];
      responseData = String.Empty;

      receiveBytes = stream.Read(receiveData, 0, receiveData.Length);
      responseData = Encoding.ASCII.GetString(receiveData, 0, receiveBytes);

      return responseData;
    }
    catch(System.NullReferenceException)
    {
      Debug.Log("Failed to receive data...");
      return "";
    }
  }

  public void Close(string message)
  {
    if (socketThread != null)
    {
      socketThread.Abort();
    }

    stream.Close();
    client.Close();
    Debug.Log("Connection Terminated!");
  }

  public void ConnectClient()
  {
    try
    {
      client = new TcpClient(hostname, port);
      Debug.Log("Connection made!");
    }
    catch(System.Net.Sockets.SocketException)
    {
      Debug.Log("Connection failed...");
    }
  }
}