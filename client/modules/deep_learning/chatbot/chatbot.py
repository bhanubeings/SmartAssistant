import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import one_hot
import numpy as np
import chatbot_utils
import os
import time

print(f"using tensorflow v{tf.__version__}")
print(f"using tensorflow.keras v{tf.keras.__version__}")




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

  def __init__(self, vocab_size=30522, max_input=512, name_len=10, max_output=50, latent_dim=128, learning_rate=1e-3):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    self.max_input = max_input
    self.name_len = name_len
    self.max_output = max_output
    self.vocab_size = vocab_size
    self.latent_dim = latent_dim
    self.learning_rate = learning_rate
    self.cls_id = self.tokenizer.cls_token_id
    self.sep_id = self.tokenizer.sep_token_id
    self.pad_id = self.tokenizer.pad_token_id

  def models(self):
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_model.trainable = False
    filters = 250
    kernel_size = 3

    # defining all layers
    enc_inputs = tf.keras.Input(shape=(self.max_input,), dtype=tf.int32)
    dec_inputs = tf.keras.Input(shape=(None, self.vocab_size,))
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

    # create training model
    model = tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=dec_outputs)

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

    return model, enc_model, dec_model

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

  def train(self, weights_filepath, enc_weights_filepath, dec_weights_filepath,
            old_weights=None, epochs=1000, steps_per_epoch=100, test_after_train=False):
    if not old_weights:
      model, enc_model, dec_model = self.models()
    elif old_weights:
      model, enc_model, dec_model = self.models()
      print("Loading last trained weights...")
      model.load_weights(old_weights)
      print("Loaded!\n")
      time.sleep(1)

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
    optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])
    model.fit_generator(generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=callbacks,
                        shuffle=False)
    print("Saving training_model weights...")
    model.save_weights(weights_filepath)
    print("Saving enc_model weights...")
    enc_model.save_weights(enc_weights_filepath)
    print("Saving dec_model weights...")
    dec_model.save_weights(dec_weights_filepath)
    print("Done!")

    print(f"\nTest after training: {test_after_train}")    
    if test_after_train:
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


  def test(self, enc_weights_filepath, dec_weights_filepath):
    # load encoder model and decoder model
    _, enc_model, dec_model= self.models()
    print("Loading enc_model weights...")
    enc_model.load_weights(enc_weights_filepath)
    print("Loading dec_model weights...")
    dec_model.load_weights(dec_weights_filepath)
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


if __name__ == "__main__":
  save_path = r"D:\Nyx\Codes\SAModels\chatbot"
  WEIGHTS_FILEPATH = rf"{save_path}\weights.h5"
  ENC_WEIGHTS_FILEPATH = rf"{save_path}\enc_weights.h5"
  DEC_WEIGHTS_FILEPATH = rf"{save_path}\dec_weights.h5"
  BertChatbot().train(old_weights=WEIGHTS_FILEPATH, epochs=5000,
                      weights_filepath=WEIGHTS_FILEPATH,
                      enc_weights_filepath=ENC_WEIGHTS_FILEPATH,
                      dec_weights_filepath=DEC_WEIGHTS_FILEPATH,
                      test_after_train=True)
  # BertChatbot().test(enc_weights_filepath=rf"{save_path}\enc_weights.h5",
  #                    dec_weights_filepath=rf"{save_path}\dec_weights.h5",)


