import tensorflow as tf
from transformers import BertTokenizer, TFBertForPreTraining

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForPreTraining.from_pretrained('bert-base-uncased')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
outputs = model(input_ids)
prediction_scores, seq_relationship_scores = outputs[:2]
print(prediction_scores)