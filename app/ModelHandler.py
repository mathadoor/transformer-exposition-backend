"""
ModelHandler defines a custom model handler.
"""
import torch
import pickle
import os
from ts.torch_handler.base_handler import BaseHandler
from utils import Transformer


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.tokenizer = None
        self.english_vocab = None
        self.german_vocab = None
        self._context = None
        self.initialized = False
        self.model = None
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.manifest = context.manifest
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        # extra_files = context

        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        serializable_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serializable_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # print("extra_files", context)
        #
        # if type(extra_files) != list or len(extra_files) != 3:
        #     raise RuntimeError("Missing the vocab files")

        with open('german.pkl', 'rb') as f:
            german_stoi = pickle.load(f)

        with open('english.pkl', 'rb') as f:
            english_stoi = pickle.load(f)

        with open('english_tokenizer.pkl', 'rb') as f:
            english_tokenizer = pickle.load(f)

        self.german_vocab = {"itos": [k for k, v in german_stoi.items()], "stoi": german_stoi}
        self.english_vocab = {"itos": [k for k, v in english_stoi.items()], "stoi": english_stoi}
        self.tokenizer = english_tokenizer

        self.model = Transformer(
            embedding_size=128,
            src_vocab_size=len(self.english_vocab["itos"]),
            trg_vocab_size=len(self.german_vocab["itos"]),
            src_pad_idx=self.english_vocab["stoi"]["<pad>"],
            num_heads=2,
            num_encoder_layers=3,
            num_decoder_layers=3,
            feedforward_dim=512,
            dropout=0.1,
            max_len=50,
            device=self.device,
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get('sentence')
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        return preprocessed_data['sentence']

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        # Create tokens using spacy and everything in lower case (which is what our vocab is)
        print("model_input", model_input)
        if type(model_input) == str:
            tokens = [token.text.lower() for token in self.tokenizer(model_input)]
        else:
            tokens = [token.lower() for token in model_input]

        # Go through each English token and convert to an index
        text_to_indices = [int(self.english_vocab["stoi"][token]) for token in tokens]

        # Add <SOS> and <EOS> in beginning and end respectively
        tokens.insert(0, self.english_vocab["stoi"]['<sos>'])
        tokens.append(self.english_vocab["stoi"]['<eos>'])

        # Convert to Tensor
        sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(self.device)
        outputs = [int(self.german_vocab["stoi"]["<sos>"])]
        for i in range(50):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(self.device)
            target_mask = torch.nn.Transformer.generate_square_subsequent_mask(trg_tensor.shape[0]).to(self.device)
            with torch.no_grad():
                output = self.model(sentence_tensor, trg_tensor, target_mask)

            best_guess = output.argmax(2)[-1, :].item()
            outputs.append(best_guess)

            if best_guess == self.german_vocab["stoi"]["<eos>"]:
                break

        # translated_sentence = [self.german_vocab["itos"][idx] for idx in outputs]
        # # remove start token
        # return translated_sentence[1:]
        return outputs

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = [self.german_vocab["itos"][idx] for idx in inference_output]
        # return postprocess_output
        #
        # return inference_output
        return [{"translation": " ".join(postprocess_output[1:])}]
