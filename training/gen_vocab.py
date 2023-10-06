import collections

import torch
import pickle
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field

# Load Vocabulary
spacy_de = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "my_checkpoint.pth"


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_de, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".en", ".de"), fields=(english, german)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

with open('../app/german.pkl', 'wb') as f:
    german_stoi = {}
    for k, v in german.vocab.stoi.items():
        german_stoi[k] = v
    pickle.dump(german_stoi, f)

with open('../app/english.pkl', 'wb') as f:
    english_stoi = {}
    for k, v in english.vocab.stoi.items():
        english_stoi[k] = v

    pickle.dump(english_stoi, f)

del train_data, valid_data, test_data, spacy_de
