import tensorflow as tf
import tensorlayer as tl
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

  def __init__(self,
               vocab_size=30522,
               max_input=30,
               max_output=30,
               latent_dim=256,
               learning_rate=1e-3,
               n_layer=3):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    self.max_input = max_input
    self.max_output = max_output
    self.vocab_size = vocab_size
    self.latent_dim = latent_dim
    self.learning_rate = learning_rate
    self.n_layer = n_layer
    self.cls_id = self.tokenizer.cls_token_id
    self.sep_id = self.tokenizer.sep_token_id
    self.pad_id = self.tokenizer.pad_token_id

    self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    self.bert_model.trainable = False

    # defining all layers
    self.enc_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="enc_inputs")
    self.dec_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="dec_inputs")

    self.gru = tf.keras.layers.GRU
    self.dense_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size,
                                                                           activation="softmax",
                                                                           name="dense_out"))

    self.enc_layers = [self.gru(self.latent_dim, return_state=True, return_sequences=True) for i in range(self.n_layer)]
    self.dec_layers = [self.gru(self.latent_dim, return_state=True, return_sequences=True) for i in range(self.n_layer)]
    self.enc_states = [None for i in range(self.n_layer)] # [None, None, None]
    self.dec_states = [None for i in range(self.n_layer)] # [None, None, None]

  def models(self):

    enc_output = self.bert_model(self.enc_inputs, training=False)[0]

    for i in range(self.n_layer):
      enc_output, self.enc_states[i] = self.enc_layers[i](enc_output)

    dec_output = self.bert_model(self.dec_inputs, training=False)[0]

    for i in range(self.n_layer):
      dec_output, self.dec_states[i] = self.dec_layers[i](dec_output, initial_state=self.enc_states[i])

    dense_output = self.dense_out(dec_output)

    # create training model
    model = tf.keras.Model(inputs=[self.enc_inputs, self.dec_inputs], outputs=dense_output)

    # create encoder model
    enc_model = tf.keras.Model(inputs=self.enc_inputs, outputs=self.enc_states)

    # create decoder model
    dec_state_input1 = tf.keras.Input(shape=(self.latent_dim,))
    dec_state_input2 = tf.keras.Input(shape=(self.latent_dim,))
    dec_state_input3 = tf.keras.Input(shape=(self.latent_dim,))
    dec_state_inputs = [dec_state_input1, dec_state_input2, dec_state_input3]

    dec_states_inf = [None for i in range(self.n_layer)]
    dec_output_inf = self.bert_model(self.dec_inputs, training=False)[0]

    for i in range(self.n_layer):
      dec_output_inf, dec_states_inf[i] = self.dec_layers[i](dec_output_inf, initial_state=dec_state_inputs[i])

    dense_output = self.dense_out(dec_output_inf)

    dec_model = tf.keras.Model(inputs=[self.dec_inputs] + dec_state_inputs,
                               outputs=[dense_output] + dec_states_inf)

    return model, enc_model, dec_model

  def decode_sequence(self, input_seq, enc_model, dec_model):
    # Encode the input as state vectors.
    states_value = enc_model.predict(input_seq)

    # Populate the first character of target sequence with the start character.
    target_seq = np.zeros((1, 1))
    target_seq[0] = self.cls_id
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_tokens = list()
    while not stop_condition:
      output_tokens, state1, state2, state3 = dec_model.predict([target_seq] + states_value)

      # Sample a token
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      if sampled_token_index != self.sep_id:
        decoded_tokens.append(sampled_token_index)

      # Exit condition: either hit max length or find stop character.
      if (sampled_token_index == self.sep_id or len(decoded_tokens) > self.max_output):
        stop_condition = True

      # Update the target sequence (of length 1).
      target_seq = np.zeros((1, 1))
      target_seq[0] = sampled_token_index
      # Update states
      states_value = [state1, state2, state3]

    decoded_sentence = self.tokenizer.decode(decoded_tokens)
    return decoded_sentence

  def train(self, weights_filepath, enc_weights_filepath, dec_weights_filepath, old_weights=None,
            epochs=1000, steps_per_epoch=100, test_after_train=False):
    start_time = time.time()
    print("\n\nMODE: Train")
    print(f"Test after training: {test_after_train}\n") 
    if not old_weights:
      model, enc_model, dec_model = self.models()
    elif old_weights:
      model, enc_model, dec_model = self.models()
      print("Loading last trained weights...")
      model.load_weights(old_weights)
      print("Loaded!\n")
      time.sleep(0.5)

    model.summary()

    converse_filepath = "./data/movie_conversations.txt"
    twitter_filepath = "./data/chat.txt"
    lines_filepath = "./data/movie_lines.txt"
    checkpoint_path = "./checkpoints"
    log_dir = "./logs"
    model_filepath = "./models"

    for path in [checkpoint_path, log_dir, model_filepath]:
      if not os.path.exists(path):
        os.mkdir(path)

    twitter_data = chatbot_utils.pull_twitter(twitter_filepath)
    twitter_generator = chatbot_utils.twitter_generator(twitter_data)
    callbacks = list()
    # callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])
    model.fit_generator(generator=twitter_generator,
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

    end_time = time.time()
    print(f"Time taken: {(end_time-start_time)/60} min(s)")
    if test_after_train:
      self.test(enc_weights_filepath, dec_weights_filepath)


  def test(self, enc_weights_filepath, dec_weights_filepath):
    print("\n\nMODE: Test")
    # load encoder model and decoder model
    _, enc_model, dec_model= self.models()
    print("Loading enc_model weights...")
    enc_model.load_weights(enc_weights_filepath)
    print("Loading dec_model weights...")
    dec_model.load_weights(dec_weights_filepath)

    exit_keyword = ".exit"

    while True:
      usr_input = input("[USER]: ")
      if usr_input == exit_keyword:
        print("Exiting BertChatbot...")
        break
      else:
        usr_input = self.tokenizer.encode(usr_input, add_special_tokens=True)
        usr_input = pad_sequences(sequences=[usr_input], maxlen=self.max_input,
                                     padding="post", truncating="post")

        decoded_sentence = self.decode_sequence(usr_input, enc_model, dec_model)
        print(f"[BertChatbot]: {decoded_sentence}")


if __name__ == "__main__":
  save_path = r"D:\Nyx\Codes\SAModels\chatbot"
  WEIGHTS_FILEPATH = rf"{save_path}\weights.h5"
  ENC_WEIGHTS_FILEPATH = rf"{save_path}\enc_weights.h5"
  DEC_WEIGHTS_FILEPATH = rf"{save_path}\dec_weights.h5"

  bert_chatbot = BertChatbot(learning_rate=0.001)
  bert_chatbot.train(old_weights=WEIGHTS_FILEPATH, epochs=2000,
                     weights_filepath=WEIGHTS_FILEPATH,
                     enc_weights_filepath=ENC_WEIGHTS_FILEPATH,
                     dec_weights_filepath=DEC_WEIGHTS_FILEPATH,
                     test_after_train=True)
  # BertChatbot().test(enc_weights_filepath=rf"{save_path}\enc_weights.h5",
  #                    dec_weights_filepath=rf"{save_path}\dec_weights.h5",)


