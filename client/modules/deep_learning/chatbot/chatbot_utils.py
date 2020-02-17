from tqdm import tqdm
import os
import numpy as np
import random
from nltk.tokenize import word_tokenize, sent_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import one_hot
from transformers import BertTokenizer




"""
Reads movie_conversations.txt to get the right pairs
Sorts movie_lines.txt using the data from movie_conversations.txt
"""

CONVERSE_FILEPATH = "./data/movie_conversations.txt"
LINES_FILEPATH = "./data/movie_lines.txt"

def sort_data(converse_filepath, lines_filepath):
  seperator = " +++$+++ "
  """
  - movie_conversations.txt
  - the structure of the conversations
  - fields
    - characterID of the first character involved in the conversation
    - characterID of the second character involved in the conversation
    - movieID of the movie in which the conversation occurred
    - list of the utterances that make the conversation, in chronological 
      order: ['lineID1','lineID2',É,'lineIDN']

  - movie_lines.txt
  - contains the actual text of each utterance
  - fields:
    - lineID
    - characterID (who uttered this phrase)
    - movieID
    - character name
    - text of the utterance

  output data: data[mov1[line1[name, converse], line2[...]], mov2[...], ...]
  """
  with open(converse_filepath, "r") as cf:
    cf_lines = [l for l in cf.read().split("\n") if l != ""]
    cf_fields = [f.split(seperator) for f in cf_lines]

  with open(lines_filepath, "r") as lf:
    lf_lines = [l for l in lf.read().split("\n") if l != ""]
    lf_fields = [f.split(seperator) for f in lf_lines]
    lf_dict = dict()
    for f in lf_fields:
      lf_dict[f[0]] = f[3:5]

  data = list()
  movie_batch = list()
  converse_batch = list()
  line_id1 = cf_fields[0][0]
  line_id2 = cf_fields[0][1]
  movie_id = cf_fields[0][2]

  for f in tqdm(cf_fields):
    # print(f)
    if movie_id == f[2]:

      if line_id1 == f[0] and line_id2 == f[1]:
        for idx in eval(f[3]):
          converse_batch.append(lf_dict[idx])

      else:
        movie_batch.append(converse_batch)
        converse_batch = list()
        for idx in eval(f[3]):
          converse_batch.append(lf_dict[idx])

      line_id1 = f[0]
      line_id2 = f[1]

    else:
      data.append(movie_batch)
      movie_batch = list()
      movie_id = f[2]

  return data

def generator(data, vocab_size=30522, max_input=512, max_output=50, shuffle=True):
  """
  generates inputs with 30 [MASK] at the end
  generates outputs with by completing the [MASK] with appropriate words from the dataset
  """

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  curr_mov = 0
  if shuffle:
    random.shuffle(data)

  while True:
    for mov in data:
      curr_mov += 1
      print(f"\nCurrently at movie number: {curr_mov}\n")
      chat_history = [101]
      for converse in mov:
        for i, line in enumerate(converse):
          phrase = tokenizer.encode(line[1], add_special_tokens=True)

          enc_inputs = pad_sequences(sequences=[chat_history], maxlen=max_input,
            padding="post", truncating='pre', value=tokenizer.pad_token_id)
          chat_history = [c for c in list(enc_inputs[0]) if c != 0]

          for p in phrase[1:]:
            chat_history.append(p)

          dec_inputs = pad_sequences(sequences=[phrase], maxlen=max_output,
            padding="post", truncating="post", value=tokenizer.pad_token_id)
          dec_outputs = pad_sequences(sequences=[phrase[1:]], maxlen=max_output,
            padding="post", truncating="post", value=tokenizer.pad_token_id)
          dec_inputs = one_hot(dec_inputs, vocab_size)
          dec_outputs = one_hot(dec_outputs, vocab_size)

          yield [enc_inputs, dec_inputs], dec_outputs

def sample_generator(data, vocab_size=30522, max_input=512, max_output=50, shuffle=True):
  """
  generates inputs with 30 [MASK] at the end
  generates outputs with by completing the [MASK] with appropriate words from the dataset
  """
  curr_mov = 0
  if shuffle:
    random.shuffle(data)

  while True:
    for mov in data:
      curr_mov += 1
      print(f"\nCurrently at movie number: {curr_mov}\n")
      chat_history = ""
      for converse in mov:
        for i, line in enumerate(converse):
          inputs = chat_history
          outputs = line[1]

          yield inputs, outputs
          chat_history += " " + outputs
          if len(word_tokenize(chat_history)) > max_input:
            chat_history = " ".join(word_tokenize(chat_history)[-max_input:])


if __name__ == '__main__':
  cornell_data = sort_data(CONVERSE_FILEPATH, LINES_FILEPATH)
  # print(cornell_data[0][0])

  if False:
    print(len(cornell_data))
    generator = generator(cornell_data)
    for i in range(7):
      print(next(generator)[0][0])

  if True:
    sample_generator = sample_generator(cornell_data)
    for i in range(50):
      next(sample_generator)
    for i in range(10):
      inputs, outputs = next(sample_generator)
      print(f"In: {inputs}")
      print(f"Out: {outputs}\n")

