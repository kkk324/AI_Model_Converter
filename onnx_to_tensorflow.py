import onnx
from onnx_tf.backend import prepare

# Load the ONNX file
model = onnx.load('yolov3.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)
