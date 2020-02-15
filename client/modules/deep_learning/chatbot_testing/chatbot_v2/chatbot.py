import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import one_hot
import numpy as np
import chatbot_utils
import os

print(f"using tensorflow v{tf.__version__}")
print(f"using tensorflow.keras v{tf.keras.__version__}")

# from deep_learning.dl_utils import AbstractLayers
# from dl_utils import AbstractLayers




# chatbot model

class BertChatbot(object):

  """
  Trains a chatbot model using tensorflow
  basic architecture: convolutional neural network

  This tweaked CNN will remember the history chat between the user and the bot.
  inputs:
    - name of bot
    - name of user
    - chat history (from the start) [state]
    - chat history (a few lines back) [words]
    - current input [words]

    # positivity of input (maybe? not sure yet)

  outputs:
    - reply [words]
    - chat history (from start) [state]
  """

  def __init__(self, vocab_size=30522, max_input=512, name_len=10, max_output=50, latent_dim=128):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    self.max_input = max_input
    self.name_len = name_len
    self.max_output = max_output
    self.vocab_size = vocab_size
    self.latent_dim = latent_dim
    self.cls_id = self.tokenizer.cls_token_id
    self.sep_id = self.tokenizer.sep_token_id
    self.pad_id = self.tokenizer.pad_token_id

  def training_model(self):
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_model.trainable = False
    filters = 250
    kernel_size = 3

    # defining all layers
    enc_inputs = tf.keras.Input(shape=(self.max_input,), dtype=tf.int32)
    dec_inputs = tf.keras.Input(shape=(None, self.vocab_size))
    conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
      padding="valid", activation="relu", strides=1)

    enc_lstm = tf.keras.layers.LSTM(self.latent_dim, return_state=True)
    dec_lstm = tf.keras.layers.LSTM(self.latent_dim, return_state=True, return_sequences=True)
    dense_out = tf.keras.layers.Dense(self.vocab_size, activation="softmax")


    bert_states = bert_model(enc_inputs)[0]
    conv_layer1 = conv(bert_states)

    enc_lstm_layer, enc_state_h, enc_state_c = enc_lstm(conv_layer1)
    enc_lstm_states = [enc_state_h, enc_state_c]

    dec_lstm_layer, _, _ = dec_lstm(dec_inputs, initial_state=enc_lstm_states)
    dec_outputs = dense_out(dec_lstm_layer)


    model = tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=dec_outputs)

    return model

  def predict_model(self, weights_filepath="./models/weights.h5"):
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_model.trainable = False
    filters = 250
    kernel_size = 3

    # defining all layers
    enc_inputs = tf.keras.Input(shape=(self.max_input,), dtype=tf.int32)
    dec_inputs = tf.keras.Input(shape=(None, self.vocab_size))
    conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
      padding="valid", activation="relu", strides=1)

    enc_lstm = tf.keras.layers.LSTM(self.latent_dim, return_state=True)
    dec_lstm = tf.keras.layers.LSTM(self.latent_dim, return_state=True, return_sequences=True)
    dense_out = tf.keras.layers.Dense(self.vocab_size, activation="softmax")


    bert_states = bert_model(enc_inputs)[0]
    conv_layer1 = conv(bert_states)

    enc_lstm_layer, enc_state_h, enc_state_c = enc_lstm(conv_layer1)
    enc_lstm_states = [enc_state_h, enc_state_c]

    dec_lstm_layer, _, _ = dec_lstm(dec_inputs, initial_state=enc_lstm_states)
    dec_outputs = dense_out(dec_lstm_layer)


    model = tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=dec_outputs)

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # load the trained weights into the model
    model.load_weights(weights_filepath)

    # create encoder model
    enc_model = tf.keras.Model(inputs=enc_inputs, outputs=enc_lstm_states)

    # create decoder model
    dec_state_input_h = tf.keras.Input(shape=(self.latent_dim,))
    dec_state_input_c = tf.keras.Input(shape=(self.latent_dim,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_lstm_outputs, dec_state_h, dec_state_c = dec_lstm(dec_inputs, initial_state=dec_states_inputs)
    dec_lstm_states = [dec_state_h, dec_state_c]
    dec_outputs = dense_out(dec_lstm_outputs)

    dec_model = tf.keras.Model(inputs=[dec_inputs] + dec_states_inputs,
                               outputs=[dec_outputs] + dec_lstm_states)

    return enc_model, dec_model, model

  def decode_sequence(self, input_seq, enc_model, dec_model):
    # Encode the input as state vectors.
    states_value = enc_model.predict(input_seq)

    # Populate the first character of target sequence with the start character.
    target_seq = np.zeros((1, 1, self.vocab_size))
    target_seq[0, 0, self.cls_id] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_tokens = list()
    while not stop_condition:
      output_tokens, h, c = dec_model.predict([target_seq] + states_value)

      # Sample a token
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      decoded_tokens.append(sampled_token_index)

      # Exit condition: either hit max length or find stop character.
      if (sampled_token_index == self.sep_id or len(decoded_tokens) > self.max_output):
        stop_condition = True

      # Update the target sequence (of length 1).
      target_seq = np.zeros((1, 1, self.vocab_size))
      target_seq[0, 0, sampled_token_index] = 1
      # Update states
      states_value = [h, c]

    decoded_sentence = self.tokenizer.decode(decoded_tokens)
    return decoded_sentence

  def train(self, old_weights=None, save_as="./models/weights.h5", model_save_as="./models/model.h5",
            epochs=1000, steps_per_epoch=32):
    if not old_weights:
      model = self.training_model()
    elif old_weights:
      model = self.training_model()
      model.load_weights(old_weights)

    model.summary()

    converse_filepath = "./data/movie_conversations.txt"
    lines_filepath = "./data/movie_lines.txt"
    checkpoint_path = "./checkpoints"
    log_dir = "./logs"
    model_filepath = "./models"

    for path in [checkpoint_path, log_dir, model_filepath]:
      if not os.path.exists(path):
        os.mkdir(path)

    data = chatbot_utils.sort_data(converse_filepath, lines_filepath)
    generator = chatbot_utils.generator(data=data)
    callbacks = list()
    # callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'], callbacks=callbacks)
    model.fit_generator(generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=callbacks,
                        shuffle=False)
    model.save_weights(save_as)
    model.save(model_save_as)

  def test_chatbot(self): #need alot of changes here, especially the input_seq for the decode_sequence method!
    # load encoder model and decoder model
    enc_model, dec_model, _ = self.predict_model()
    exit_keyword = ".exit"
    chat_history = [self.cls_id]

    while True:
      usr_input = input("[USER]: ")
      if usr_input == exit_keyword:
        print("Exiting BertChatbot...")
        break
      else:
        usr_input = self.tokenizer.encode(usr_input, add_special_tokens=True)
        usr_input = [u for u in usr_input[1:] if u != 0]
        for u in usr_input:
          chat_history.append(u)

        chat_history = pad_sequences(sequences=[chat_history], maxlen=self.max_input,
                                     padding="post", truncating="pre")

        decoded_sentence = self.decode_sequence(chat_history, enc_model, dec_model)
        print(f"[BertChatbot]: {decoded_sentence}")

        chat_history = [ch for ch in list(chat_history)[0] if ch != 0]
        decoded_sentence = self.tokenizer.encode(decoded_sentence, add_special_tokens=True)
        decoded_sentence = [ds for ds in decoded_sentence[1:] if ds != 0]
        for dt in decoded_sentence:
          chat_history.append(dt)

  def train_test(self, save_as="./models/weights.h5", epochs=10, steps_per_epoch=10):
    enc_model, dec_model, model = self.predict_model()
    model.summary()

    converse_filepath = "./data/movie_conversations.txt"
    lines_filepath = "./data/movie_lines.txt"
    checkpoint_path = "./checkpoints"
    log_dir = "./logs"
    model_filepath = "./models"

    for path in [checkpoint_path, log_dir, model_filepath]:
      if not os.path.exists(path):
        os.mkdir(path)

    data = chatbot_utils.sort_data(converse_filepath, lines_filepath)
    generator = chatbot_utils.generator(data=data)
    callbacks = list()
    # callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    model.fit_generator(generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=callbacks,
                        shuffle=False)

    for i in range(10):
      model_inputs, model_outputs = next(generator)
      prompt = self.tokenizer.decode(enc_inputs[0])
      reply = self.decode_sequence(model_inputs, model_outputs)
      print(f"[PROMPT]: {prompt}")
      print(f"[REPLY]: {reply}")



# BertChatbot().train(epochs=100)
BertChatbot().train(old_weights="./models/weights.h5", epochs=100)
# BertChatbot().test_chatbot()
# BertChatbot().train_test()


