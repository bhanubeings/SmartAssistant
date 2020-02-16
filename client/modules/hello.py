import re




def handle(text, Mic, Agent):
  Mic.say("whats up!")
  print("please,anymore instructions?\n")


def isValid(text):
  # check if text is valid
  return (bool(re.search(r"\bhey |hello |hi\b", text, re.IGNORECASE)))




if __name__ == '__main__':
  print(isValid("How is eveything going?"))