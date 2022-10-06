import onnxruntime
import torch.onnx
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
from networks import *
import cv2
import time
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def reverse_one_hot(image):
    # Convert output of model to predicted class
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x

def process_output(label, img):
    label = reverse_one_hot(label.squeeze(0))
    label = np.array(label.cpu(), dtype='uint8')
    label = (1 - label) * 255
    label = cv2.merge([label, label, label])
    label = cv2.resize(label, img.shape[:2][::-1])
    return label

def process_input(image_path):
    normalize = transforms.Compose([
        transforms.Resize((480, 360)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.fromarray(image_path[:, :, ::-1])
    img = normalize(img).to(device)
    return img.unsqueeze(0)

# img = cv2.imread('dataset/Figaro_1k/test/images/226.jpg')

root = 'dataset/Figaro_1k/test/images/'
list_img = [root + name for name in os.listdir(root) if name.endswith('jpg')]

model = BiSeNet(num_classes=2)
weight = "checkpoints/lastest_model_CeFiLa.pth"
model.load_state_dict(torch.load(weight, map_location=torch.device(device))['state_dict'])
model = model.to(device)
model.eval()


session = onnxruntime.InferenceSession("checkpoints/bisenet.onnx")

time_inference_model = []
time_inference_onnx_model = []

for path in tqdm(list_img):
    img = cv2.imread(path)
    img_processed = process_input(img)

    start = time.time()
    torch_out = model(img_processed)
    end_model = time.time() - start
    time_inference_model.append(end_model)

    # compute ONNX Runtime output prediction
    start = time.time()
    ort_inputs = {session.get_inputs()[0].name: to_numpy(img_processed)}
    ort_outs = session.run(None, ort_inputs)
    end_onnx_model = time.time() - start
    time_inference_onnx_model.append(end_onnx_model)
    # out = torch.from_numpy(ort_outs[0])
    # label = process_output(out, img)

    # cv2.imshow('results', label)
    # cv2.waitKey(0)

t_model = sum(time_inference_model) / len(time_inference_model)
t_onnx_model = sum(time_inference_onnx_model) / len(time_inference_onnx_model)

print('Time inference model PyTorch:', t_model)
print('Time inference model ONNX:', t_onnx_model)
print(f'ONNX inference faster {round(t_model / t_onnx_model, 2)} times than PyTorch')