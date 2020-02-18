import re
from nltk.tokenize import word_tokenize, sent_tokenize
import webbrowser




def handle(text, Mic, Agent):
  Mic.say(f"Searching for google on {text}!")
  query = "https://www.google.com/search?q="
  words = word_tokenize(text)
  s = "+".join(words)
  query = query+s
  webbrowser.open_new_tab(query)




def isValid(text):
  # check if text is valid
  return (bool(re.search(r"\bgoogle |search |what |how |why\b", text, re.IGNORECASE)))




if __name__ == '__main__':
  print(isValid("what is earth?"))

'''
>>> from nltk.tokenize import word_tokenize, sent_tokenize
>>> sentence = "Ali goes to the mall, he meets Mr. Abu"
>>> print(sentence)
Ali goes to the mall, he meets Mr. Abu
>>> ["Ali", "goes", "to", "the", "mall", ",", "he".....]

>>> words = sentence.split(" ")
>>> words
['Ali', 'goes', 'to', 'the', 'mall,', 'he', 'meets', 'Mr.', 'Abu']
>>> words =  word_tokenize(sentence)
>>> words
['Ali', 'goes', 'to', 'the', 'mall', ',', 'he', 'meets', 'Mr.', 'Abu']
>>> sentence = "Ali goes to the mall, he meets Mr.Abu"
>>> words =  word_tokenize(sentence)
>>> words
['Ali', 'goes', 'to', 'the', 'mall', ',', 'he', 'meets', 'Mr.Abu']
>>> paragraph = "A paragraph is a group of words put together to form a group that is usually longer than a sentence. Paragraphs are often made up of several sentences. There are usually between three and eight sentences. Paragraphs can begin with an indentation (about five spaces), or by missing a line out, and then starting again. This makes it easier to see when one paragraph ends and another begins. Mr.Abu is a polite guy."
>>> sentences = sent_tokenize(paragraph)
>>> sentences
['A paragraph is a group of words put together to form a group that is usually longer than a sentence.', 'Paragraphs are often made up of several sentences.', 'There are usually between three and eight sentences.', 'Paragraphs can begin with an indentation (about five spaces), or by missing a line out, and then starting again.', 'This makes it easier to see when one paragraph ends and another begins.', 'Mr.Abu is a polite guy.']
>>> for i in sentences:
...     print(i)
...
A paragraph is a group of words put together to form a group that is usually longer than a sentence.
Paragraphs are often made up of several sentences.
There are usually between three and eight sentences.
Paragraphs can begin with an indentation (about five spaces), or by missing a line out, and then starting again.
This makes it easier to see when one paragraph ends and another begins.
Mr.Abu is a polite guy.

>>> text = "open youtube.com link"
>>> words = word_tokenize(text)
>>> words
['open', 'youtube.com', 'link']
>>> link = words[1]
>>> link
'youtube.com'
>>>
>>> link = ""
>>> link
''
>>> for w in words:
...   if ".com" in w or "www" in w:
...     link = w
...
>>> link
'youtube.com'
'''