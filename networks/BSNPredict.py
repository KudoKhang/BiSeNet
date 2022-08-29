import numpy as np

from networks import *

class BSNPredict:
    def __init__(self, NUM_CLASSES=2, pretrained='checkpoints/best_model_900.pth'):
        self.NUM_CLASSES = NUM_CLASSES
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BiSeNet(num_classes=self.NUM_CLASSES)
        self.model.load_state_dict(torch.load(self.pretrained, map_location=torch.device(self.device))['state_dict'])
        print(f'Inference with ---{pretrained}---')
        self.model = self.model.to(self.device)
        self.model.eval()

    def reverse_one_hot(self, image):
        # Convert output of model to predicted class
        image = image.permute(1, 2, 0)
        x = torch.argmax(image, dim=-1)
        return x

    def process_input(self, image_path):
        normalize = transforms.Compose([
            transforms.Resize((480, 360)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = Image.fromarray(image_path[:,:,::-1])
        img = normalize(img).to(self.device)
        return img.unsqueeze(0)

    def process_output(self, label, img):
        label = self.reverse_one_hot(label.squeeze(0))
        label = np.array(label.cpu(), dtype='uint8')
        label = (1 - label) * 255
        label = cv2.merge([label, label, label])
        label = cv2.resize(label, img.shape[:2][::-1])
        return label

    def predict(self, image: np.ndarray, bbox):
        labels = np.zeros_like(image)
        temp_label = labels.copy()
        for bb in bbox:
            x1, y1, x2, y2 = bb
            person = image[y1: y2, x1: x2]
            label = self.model(self.process_input(person.copy()))
            label = self.process_output(label, person)
            temp_label[y1:y2, x1:x2] = label
            labels += temp_label
        _, labels = cv2.threshold(labels, 20, 255, cv2.THRESH_BINARY)
        return labels

