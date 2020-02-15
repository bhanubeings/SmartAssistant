from tqdm import tqdm
import os
import numpy as np
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

# def onehot(phrase, vocab_size, max_output):
#   onehot_output = np.zeros((max_output, vocab_size))
#   for i, p in enumerate(phrase):
#     onehot_output[i][p] = 1

#   return onehot_output

def generator(data, vocab_size=30522, max_input=512, max_output=30):
  """
  return model inputs:
    - name of bot (not using this now)
    - name of user (not using this now)
    - chat history (a few lines back) [words]
    - current input [words]
  and outputs:
    - reply [words]
    - chat history (from start) [state]
  """
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  while True:
    for mov in data:
      chat_history = [101]
      for converse in mov:
        for i, line in enumerate(converse):
          phrase = tokenizer.encode(line[1], add_special_tokens=True)

          for p in phrase[1:]:
            chat_history.append(p)

          enc_inputs = pad_sequences(sequences=[chat_history], maxlen=max_input,
                                     padding="post", truncating='pre')
          dec_inputs = pad_sequences(sequences=[phrase], maxlen=max_output,
                                     padding="post", truncating='post')
          dec_outputs = pad_sequences(sequences=[phrase[1:]], maxlen=max_output,
                                      padding="post", truncating='post')
          dec_outputs = one_hot(dec_outputs, vocab_size)
          yield [enc_inputs, dec_inputs], dec_outputs


if __name__ == '__main__':
  cornell_data = sort_data(CONVERSE_FILEPATH, LINES_FILEPATH)
  # print(cornell_data[0][0])
  print(len(cornell_data))
  generator = generator(cornell_data)
  # next(generator)
  print(next(generator))
  # print(next(generator))

