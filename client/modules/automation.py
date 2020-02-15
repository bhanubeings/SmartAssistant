import os, sys
import re
from logix_control import controller




def handle(text, mic, profile):
  """
  Reports the current time based on the user's timezone.

  Arguments:
  text -- user-input, typically transcribed speech
  mic -- used to interact with the user (for both input and output)
  profile -- contains information related to the user (e.g., phone
             number)
  """

  tz = getTimezone(profile)
  now = datetime.datetime.now(tz=tz)
  service = DateService()
  response = service.convertTime(now)
  mic.say("It is %s right now." % response)


def isValid(text):
  # check if text is valid
  return bool(re.search(r'\bswitch (off|on)?\b', text, re.IGNORECASE))


# class Automation(IntentionClassifier):

#   def __init__(self):
#     pass



if __name__ == '__main__':
  print(isValid("switch on the lights"))