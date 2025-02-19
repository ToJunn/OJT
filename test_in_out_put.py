import onnx

model = onnx.load(r"model_repository\resnet50\1\model.onnx")  # Using raw string


for input in model.graph.input:
    print(input.name)

for output in model.graph.output:
    print(output.name)

