from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import pickle
import spacy
from utils import Transformer, translate_sentence

# run the app
app = Flask(__name__)
CORS(app)

# Load Vocabulary
# spacy_de = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "my_checkpoint.pth"

# def tokenize_de(text):
#   return [tok.text for tok in spacy_de.tokenizer(text)]
#
# def tokenize_eng(text):
#   return [tok.text for tok in spacy_eng.tokenizer(text)]
#
# german = Field(tokenize=tokenize_de, lower=True, init_token="<sos>", eos_token="<eos>")
#
#
# english = Field(
#   tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
# )


with open('german.pkl', 'rb') as f:
    german_stoi = pickle.load(f)

with open('english.pkl', 'rb') as f:
    english_stoi = pickle.load(f)

german_vocab = {"itos" : [k for k, v in german_stoi.items()], "stoi" : german_stoi}
english_vocab = {"itos" : [k for k, v in english_stoi.items()], "stoi" : english_stoi}

# train_data, valid_data, test_data = Multi30k.splits(
#   exts=(".en", ".de"), fields=(english, german)
# )
#
# german.build_vocab(train_data, max_size=10000, min_freq=2)
# english.build_vocab(train_data, max_size=10000, min_freq=2)

# with open('german.pkl', 'wb') as f:
#     pickle.dump(german.vocab.stoi, f)
#
# with open('english.pkl', 'wb') as f:
#     pickle.dump(english.vocab.stoi, f)

# del train_data, valid_data, test_data, spacy_de

# Load Model
src_vocab_size = len(english_vocab["itos"])
trg_vocab_size = len(german_vocab["itos"])
embedding_size = 128
num_heads = 2
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 50
feed_forward_layer = 512
src_pad_idx = english_vocab["stoi"]["<pad>"]
model = Transformer(
  embedding_size,
  src_vocab_size,
  trg_vocab_size,
  src_pad_idx,
  num_heads,
  num_encoder_layers,
  num_decoder_layers,
  feed_forward_layer,
  dropout,
  max_len,
  device,
).to(device)

# Load Weights
model.load_state_dict(torch.load(model_name))


@app.route('/one-predict', methods=['POST'])
def predict():
  # Get the sentence to be translated
  data = request.get_json()
  sentence = data.get('sentence')

  # Make prediction using model loaded from disk as per the data.
  prediction = translate_sentence(model, sentence, german_vocab, english_vocab, device, spacy_eng)

  return jsonify({'translation': prediction})

if __name__ == '__main__':
  app.run(host="localhost", port=5000, debug=True)
