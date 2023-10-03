"""
The base code is sourced from https://github.com/aladdinpersson/Machine-Learning-Collection.git
"""

import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, german, english, device, max_length=50):

    # Load English tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_eng(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, english.init_token)
    tokens.append(english.eos_token)

    # Go through each English token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [german.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        target_mask = torch.nn.Transformer.generate_square_subsequent_mask(trg_tensor.shape[0]).to(device)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor, target_mask)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == german.vocab.stoi["<eos>"]:
            break

    translated_sentence = [german.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
