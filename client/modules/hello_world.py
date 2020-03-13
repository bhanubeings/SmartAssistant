import re


PRIORITY = 0

def handle(text, Mic, Agent):
  Mic.say("Hi!")


def isValid(text):
  # check if text is valid
  return bool(re.search(r'\bhello world\b', text, re.IGNORECASE))




if __name__ == '__main__':
  print(isValid("Hello world"))