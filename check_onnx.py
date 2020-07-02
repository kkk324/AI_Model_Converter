import onnx

onnx_model = onnx.load("yolov3.onnx")
onnx.checker.check_model(onnx_model)

