import re


PRIORITY = 7

def handle(text, Mic, Agent):
  """
  Abilities:
  - report the status of the Smart Assistant (will soon be deprecated
    for AI to reply)
  """
  Mic.say("Everything is copacetic!")


def isValid(text):
  # check if text is valid
  return True




if __name__ == '__main__':
  print(isValid("How is eveything going?"))