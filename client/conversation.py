from notifier import Notifier
from brain import Brain
import random




class Conversation(object):

  def __init__(self, Agent):
    self.Agent = Agent
    self.Brain = Brain(self.Agent)
    self.Notifier = Notifier(self.Agent)

  def handleForever(self):
    # main loop of the Agent
    stop_condition = False
    text = None

    while not stop_condition:
      data = self.Agent.Mic.stream.read(self.Agent.Mic.CHUNK)
      # check the surrounding volume
      if self.Agent.Mic.fetch_threshold(data):
        # check if hotword was spoken
        detection = self.Agent.Mic.passive_listen(self.Agent.hotwords)

        if detection:
          # greet when hotword is detected
          self.Agent._print(f"Hotword '{self.Agent.name}' detected!")
          self.Agent.Mic.say(random.choice(self.Agent.GREET))
          question = self.Agent.Mic.active_listen()

          if question:
            self.Brain.query(question)
          else:
            self.Agent.Mic.say("Pardon?")
            question = self.Agent.Mic.active_listen()
            if question:
              self.Brain.query(question)
            else:
              pass

