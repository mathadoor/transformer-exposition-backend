"""
The base code is sourced from https://github.com/aladdinpersson/Machine-Learning-Collection.git
"""

import torch
from torch import nn
import spacy


class Transformer(nn.Module):
  def __init__(
    self,
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    feedforward_dim,
    dropout,
    max_len,
    device,
  ):
    super(Transformer, self).__init__()
    self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
    self.src_position_embedding = nn.Embedding(max_len, embedding_size)
    self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
    self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

    self.device = device
    self.transformer = nn.Transformer(
      embedding_size,
      num_heads,
      num_encoder_layers,
      num_decoder_layers,
      feedforward_dim,
      dropout,
    )
    self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.src_pad_idx = src_pad_idx

  def make_src_mask(self, src):
    src_mask = src.transpose(0, 1) == self.src_pad_idx

    # (N, src_len)
    return src_mask.to(self.device)

  def forward(self, src, trg, trg_mask):
    src_seq_length, N = src.shape
    trg_seq_length, N = trg.shape

    src_positions = (
      torch.arange(0, src_seq_length)
      .unsqueeze(1)
      .expand(src_seq_length, N)
      .to(self.device)
    )

    trg_positions = (
      torch.arange(0, trg_seq_length)
      .unsqueeze(1)
      .expand(trg_seq_length, N)
      .to(self.device)
    )

    # embed_src
    embed_src = self.src_word_embedding(src)
    embed_src += self.src_position_embedding(src_positions)
    embed_src = self.dropout(embed_src)

    embed_trg = self.trg_word_embedding(trg)
    embed_trg += self.trg_position_embedding(trg_positions)
    embed_trg = self.dropout(embed_trg)

    src_padding_mask = self.make_src_mask(src)

    out = self.transformer(
      embed_src,
      embed_trg,
      src_key_padding_mask=src_padding_mask,
      tgt_mask=trg_mask,
    )
    out = self.fc_out(out)
    return out


def translate_sentence(model, sentence, german, english, device, spacy_eng, max_length=50):
  # Load English tokenizer

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
