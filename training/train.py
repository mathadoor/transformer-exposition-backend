"""
The base code is sourced from https://github.com/aladdinpersson/Machine-Learning-Collection.git

Seq2Seq using Transformers on the Multi30k
dataset. In this video I utilize Pytorch
inbuilt Transformer modules, and have a
separate implementation for Transformers
from scratch. Training this model for a
while (not too long) gives a BLEU score
of ~35, and I think training for longer
would give even better results.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import torch.ao.quantization.quantize_fx as quantize_fx
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from tqdm import tqdm
import math
import io
from torch.nn.utils.rnn import pad_sequence

"""
To install spacy languages do:
python -m spacy download de
python -m spacy download en
"""
spacy_de = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_de(text):
  return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_eng(text):
  return [tok.text for tok in spacy_eng.tokenizer(text)]


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
    # embed_src = self.src_position_embedding(embed_src)
    embed_src = self.dropout(embed_src)

    embed_trg = self.trg_word_embedding(trg)
    embed_trg += self.trg_position_embedding(trg_positions)
    # embed_trg = self.trg_position_embedding(embed_trg)
    embed_trg = self.dropout(embed_trg)

    src_padding_mask = self.make_src_mask(src)
    # tgt_padding_mask = self.make_src_mask(trg)
    # trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
    #   self.device
    # )

    out = self.transformer(
      embed_src,
      embed_trg,
      src_key_padding_mask=src_padding_mask,
      # tgt_key_padding_mask=tgt_padding_mask,
      tgt_mask=trg_mask,
    )
    out = self.fc_out(out)
    return out


# If main function is called
german = Field(tokenize=tokenize_de, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
  tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
  exts=(".en", ".de"), fields=(english, german)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# We're ready to define everything we need for training our Seq2Seq model
mode = "train"
device = "cuda" if torch.cuda.is_available() and mode == "train" else "cpu"
# device = torch.device("cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 20
learning_rate = 5e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(english.vocab)
trg_vocab_size = len(german.vocab)
embedding_size = 128
num_heads = 2
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 50
feed_forward_layer = 512
src_pad_idx = english.vocab.stoi["<pad>"]
if __name__ == "__main__":

  # Tensorboard to get nice loss plot
  step = 0

  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
  )

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

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
  )

  pad_idx = english.vocab.stoi["<pad>"]
  criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

  if load_model:
    if mode == "train":
      load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    else:
      # model_int8 = torch.ao.quantization.quantize_dynamic(
      #   model,  # the original model
      #   {
      #     torch.nn.Embedding: torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig,
      #     torch.nn.Linear: torch.ao.quantization.qconfig.default_dynamic_qconfig
      #   },  # a set of layers to dynamically quantize
      #   dtype=torch.qint8)
      # model_int8.load_state_dict(torch.load("model_int8.pth"))
      model.load_state_dict(torch.load("my_checkpoint.pth"))
      # model = model_int8

  example_sentences = ["A man is walking in the street.", "The dog is running in the park.",
                       "The cat is sitting on the table.", "The computer is sitting on the table.",
                       "The woman is brushing her teeth.", "A man is smiling at a stuffed lion.",
                       "A man in a blue shirt is standing on a ladder cleaning a window.",
                       "A little girl climbing into a wooden playhouse.",
                       "Boys dancing on poles in the middle of the night.",
                       "A man in a neon green and orange uniform is driving on a green tractor.",
                       "We had a delicious dinner at the new restaurant.", "He enjoys playing soccer in his free time.",
                       "My daughter is sleeping in her room."]
  if mode == "train":
    for epoch in range(num_epochs):
      print(f"[Epoch {epoch} / {num_epochs}]")

      if save_model:
        checkpoint = {
          "state_dict": model.state_dict(),
          "optimizer": optimizer.state_dict(),
        }
        model.to("cpu")
        model.eval()
        model_int8 = torch.quantization.quantize_dynamic(
          model,  # the original model
          {
            torch.nn.Embedding: torch.quantization.qconfig.float_qparams_weight_only_qconfig,
            torch.nn.Linear: torch.quantization.qconfig.default_dynamic_qconfig
          },  # a set of layers to dynamically quantize
          dtype=torch.qint8)
        s = torch.jit.script(model)
        # torch.save(model_int8.state_dict(), "model_int8.pth")
        torch.jit.save(s, "model_int8.pt")
        torch.save(model.state_dict(), "my_checkpoint.pth")
        model.train()
        model.to(device)

      model.eval()
      for sentence in example_sentences:
        translated_sentence = translate_sentence(
          model, sentence, german, english, device, max_length=max_len
        )

        print(f"Translated example sentence: {sentence} \n {' '.join(translated_sentence)}")
      model.train()
      losses = []

      for batch_idx, batch in tqdm(enumerate(train_iterator)):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        target_mask = nn.Transformer.generate_square_subsequent_mask(batch.trg.shape[0] - 1).to(device)
        # Pad inp_data and target with pad token and increase their dimension to (max_len, batch)
        # src_pad_dims = (0, 0, 0, max_len - inp_data.shape[0])
        # target_pad_dims = (0, 0, 0, max_len - target.shape[0])
        # inp_data = torch.nn.functional.pad(inp_data, pad=src_pad_dims, value=english.vocab.stoi["<pad>"])
        # target = torch.nn.functional.pad(target, pad=target_pad_dims, value=german.vocab.stoi["<pad>"])


        # Forward prop

        output = model(inp_data, target[:-1, :], target_mask)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        # writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

      mean_loss = sum(losses) / len(losses)
      scheduler.step(mean_loss)

  # running on entire test data takes a while
  if mode != "train":
    model.eval()
    for sentence in example_sentences:
      translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=max_len
      )

      print(f"Translated example sentence: {sentence} \n {' '.join(translated_sentence)}")

  # Define Dummy Source and target translation
  # print(german.vocab.stoi['<pad>'], english.vocab.stoi['<pad>'])
  # model.eval()

  # dummy_src = torch.zeros((max_len, 1)).int()
  # dummy_trg = torch.zeros((max_len, 1)).int()
  # model.to("cpu")
  # torch.onnx.export(model, (dummy_src, dummy_trg), "model.onnx", input_names=["src", "trg"], output_names=["out"], export_params=True, opset_version=18, verbose=True,
  #                   dynamic_axes={
  #                     'src': [0, 1],
  #                     'trg': [0, 1],
  #                     'out': [0, 1]
  #                     })
  # model.to(device)
  # input = [[2], [4], [9], [6], [4], [1199], [52], [11], [86], [231], [10], [506], [8], [4], [52], [1103], [5], [3]]
  # output = [[2], [13], [7], [218], [7], [2843], [5], [56], [58], [12], [56], [5], [56], [398], [4]]
  #
  # input2 = pad_sequence(torch.tensor([input]), padding_value=english.vocab.stoi['<pad>'])
  # output2 = pad_sequence(torch.tensor([[2]]), padding_value=german.vocab.stoi['<pad>'])
  # print(torch.argmax(model(input2, output2)))
  # sentence = "A man in a neon green and orange uniform is driving on a green tractor."
  # tokens = [token.text.lower() for token in spacy_eng(sentence)]
  # print(" ".join([english.vocab.itos[token] for token in input]))
  # print(" ".join([german.vocab.itos[token[0]] for token in output]))

  score = bleu(test_data[1:100], model, german, english, device)
  print(f"Bleu score {score * 100:.2f}")
