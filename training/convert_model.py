import onnx, onnxruntime
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "./model.onnx"
model_int8 = "./model_int8.onnx"
max_len = 50
# print(ort_session.get_inputs())
# ort_inputs = {ort_session.get_inputs()[0].name: np.ones(100, 1).astype(np.int)}
# ort_outs = ort_session.run(None, ort_inputs)
# quantized_model = quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)

model = onnx.load(model_int8)

print(model.graph.value_info)

onnx.checker.check_model(model)
input_sentence = np.array([[2], [4], [9], [6], [4], [1199], [52], [11], [86], [231], [10], [506], [8], [4], [52], [1103], [5], [3]]).astype(np.int32)
trg_sentence = np.array([[2]]).astype(np.int32)
# Extend input_sentence and trg_sentence so that the first dimension is 100 with rest of the values being 0
# input_sentence = np.pad(input_sentence, ((0, max_len - input_sentence.shape[0]), (0, 0)), 'constant', constant_values=1).astype(np.int32)
# print(input_sentence)
ort_session = onnxruntime.InferenceSession(model_fp32, providers= ['CPUExecutionProvider'])
print(ort_session.get_inputs())
# Define input for inference
i = 0
while i < max_len - 10 and trg_sentence[i][0] != 3:
  # trg_sentence_ = np.array(trg_sentence).astype(np.int32)
  # trg_sentence_ = np.pad(trg_sentence_, ((0, max_len - trg_sentence_.shape[0]), (0, 0)), 'constant', constant_values=1)
  ort_inputs = {ort_session.get_inputs()[0].name: input_sentence, ort_session.get_inputs()[1].name: trg_sentence}
  ort_outs = ort_session.run(None, ort_inputs)
  # trg_sentence.append([np.argmax(ort_outs[0][i, 0, :])])
  i += 1

print(trg_sentence)
