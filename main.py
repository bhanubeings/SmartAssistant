import os, sys, pathways
os.chdir(os.path.dirname(os.path.realpath(__file__)))


all_pathways = [item for item in dir(pathways) if item.endswith("_PATH")]

for path in all_pathways:
  sys.path.insert(0, getattr(pathways, path))

from config import config_dict as cf
import pyaudio, audioop
from speech import Speech
import requests
import time
import random
from utils import Util
from mic import Mic
from sonos_util import HttpServer, SonosUtil
from client.conversation import Conversation
from socket_server import Socket




class Agent(Util):

  def __init__(self, config, verbose=1):
    # retrieve settings for the agent
    self.name = config["name"]
    self.version = config["version"]
    self.controller_ip = config["controller_ip"]
    self.port = config["SONOS_port"]
    self.user_firstname = config["user_firstname"]
    self.user_lastname = config["user_lastname"]
    self.hotword_train_bypass = config["hotword_train_bypass"]
    self.hotword_train_amount = config["hotword_train_amount"]
    self.socket_port = config["socket_port"]
    self.socket_max_conn = config["socket_max_conn"]

    # simple responses
    self.AFFIRM = config["AFFIRM"]
    self.REJECT = config["REJECT"]
    self.GREET = config["GREET"]

    # all pathways
    self.APP_PATH = pathways.APP_PATH
    self.MODULES_PATH = pathways.MODULES_PATH
    self.CLIENT_PATH = pathways.CLIENT_PATH
    self.TEMP_PATH = pathways.TEMP_PATH

    if not os.path.isdir(self.TEMP_PATH): # create temp folder to store temp items
      os.mkdir(self.TEMP_PATH)

    self.verbose = verbose

    if not self.checkInternet():
      raise Exception("Agent needs to be connected to the internet in order to function.")
      self.internet_condition = False
    else:
      self.internet_condition = True

    super().__init__(self.verbose)
    self.Mic = Mic(self)
    self.Speech = Speech(self)
    self.Socket = Socket(self) # Python - Unity comm
    self.Socket.send_msg("msg from Python.")
    self.server = HttpServer(self.port)
    self.server.start()
    self.Conversation = Conversation(self)

  def __str__(self):
    sep = "="*len("Agent")
    return f"Agent\n{sep}\n\nName: {self.name}\nVersion: {self.version}\nController IP: {self.controller_ip}\n"

  def __repr__(self):
    return f'Agent("{self.name}", "{self.version}", "{self.controller_ip}")'

  def run(self):
    print("\n")
    print(self)
    print("Please follow the instructions if you are running this for the first time.")

    # Check if hotword data is present
    if not os.path.isfile(f"{self.APP_PATH}/hotwords.txt"):
      self.Mic.train_hotword()
    elif not self.hotword_train_bypass:
      ans = ""
      while ans != "y" and ans != "n":
        ans = input("Do you want to retrain Hotword? [y/n]: ")
      if ans == "y":
        self.Mic.train_hotword()

    # get hotwords
    with open(f"{self.APP_PATH}/hotwords.txt", "r") as f:
      self.hotwords = f.read().split("\n")

    print("Agent initiated!")
    # try:
    #   self.Conversation.handleForever()
    # except Exception as e:
    #   self._print(f"\n==========================================\nError msg:\n{e}")
    #   print("\nStopping program...")
    #   self.Mic.stream.stop_stream()
    #   self.Mic.stream.close()
    #   self.Mic.audio.terminate()
    self.Conversation.handleForever()


if __name__ == '__main__':
  nyx_agent = Agent(cf, verbose=1)
  nyx_agent.run()


  # test local mic
  # print(nyx_agent.Mic.activeListen())

  # test local speaker
  # nyx_agent.Mic.say("It's fine!")
  # time.sleep(2)

