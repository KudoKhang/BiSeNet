import cv2
import onnxruntime

from networks import *

class BSNONNXPredict:
    def __init__(self, pretrained='checkpoints/bisenet.onnx', is_draw_bbox=False):
        self.is_draw_bbox = is_draw_bbox
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Inference with ---{pretrained}---')
        self.model = onnxruntime.InferenceSession(pretrained)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def shift(self, image, shift_x=22, shift_y=10):
        shift_to_right_or_down = 1
        shift_to_left_or_top = -1
        for i in range(image.shape[1] - 1, image.shape[1] - shift_x, -1):
            image = np.roll(image, shift_to_right_or_down, axis=1)
            image[:, -1] = 0

        for i in range(image.shape[0] - 1, image.shape[0] - shift_y, -1):
            image = np.roll(image, 1, axis=0)
            image[-1, :] = 0

        return image

    def reverse_one_hot(self, image):
        # Convert output of model to predicted class
        image = image.permute(1, 2, 0)
        x = torch.argmax(image, dim=-1)
        return x

    def process_output(self, label, img):
        label = torch.from_numpy(label[0])
        label = reverse_one_hot(label.squeeze(0))
        label = np.array(label.cpu(), dtype='uint8')
        label = (1 - label) * 255
        label = cv2.resize(label, img.shape[:2][::-1])
        label = self.shift(label)
        return label

    def process_input(self, image_path):
        img = self.check_type(image_path)
        normalize = transforms.Compose([
            transforms.Resize((480, 360)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = Image.fromarray(img[:, :, ::-1])
        img = normalize(img).to(self.device)
        return img.unsqueeze(0)

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image path")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def visualize(self, img, label, color = (0, 255, 0)):
        label = cv2.merge([label, label, label])
        if color:
            label[:,:,0][np.where(label[:,:,0]==255)] = color[0]
            label[:,:,1][np.where(label[:,:,1]==255)] = color[1]
            label[:,:,2][np.where(label[:,:,2]==255)] = color[2]
        return cv2.addWeighted(img, 0.6, label, 0.4, 0)

    def predict(self, image, is_visualize=True):
        image_processed = self.process_input(image)
        inputs = {self.model.get_inputs()[0].name: self.to_numpy(image_processed)}
        outputs = self.model.run(None, inputs)
        mask = self.process_output(outputs, image)

        if is_visualize:
            mask = self.visualize(image, mask)

        return mask
