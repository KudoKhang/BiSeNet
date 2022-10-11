import torch.onnx
import io
import numpy as np
from torch import nn
from networks import *
import onnx

def Convert_ONNX():
    model.eval()
    # Let's create a dummy input tensor
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 480, 360, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "checkpoints/bisenet.onnx",       # where to save the model
         verbose=True, # Show progress
         export_params=False,  # store the trained parameter weights inside the model file
         opset_version=12,    # the ONNX version to export the model to
         do_constant_folding=False,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BiSeNet(num_classes=2)
    weight = "checkpoints/lastest_model_CeFiLa.pth"
    model.load_state_dict(torch.load(weight, map_location=torch.device(device))['state_dict'])
    model = model.to(device)
    Convert_ONNX()