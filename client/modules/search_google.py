import re
from nltk.tokenize import word_tokenize, sent_tokenize
import webbrowser




def handle(text, Mic, Agent):
  remove_words = ["google ", "Google ", "search ", "for "]
  for rw in remove_words:
    text = text.replace("google ", "")

  Mic.say(f"Here's what I found on google on '{text}'")
  query = "https://www.google.com/search?q="
  words = word_tokenize(text)
  phrase = "+".join(words)
  query += phrase
  webbrowser.open_new_tab(query)


def isValid(text):
  # check if text is valid
  return (bool(re.search(r"\bgoogle |search \b", text, re.IGNORECASE)))




if __name__ == '__main__':
  print(isValid("what is earth?"))

