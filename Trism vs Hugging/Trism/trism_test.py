from trism import TritonModel
import numpy as np

model = TritonModel(
  model="resnet50",     # Model name.
  version=0,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True             # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = model.run(data=[input_data])

print(outputs)