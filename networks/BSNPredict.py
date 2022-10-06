import time

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
        bbox_empty = [np.array([0, 0, image.shape[1], image.shape[0]])]
        bbox = bbox if len(bbox) > 0 else bbox_empty # if no person detected --> return original image
        for bb in bbox:
            start_pre_process = time.time()
            x1, y1, x2, y2 = bb
            person = image[y1: y2, x1: x2]
            image_processed = self.process_input(person.copy())
            pre_process_time = str(round((time.time() - start_pre_process) * 1e3, 2)) + 'ms'

            start_model = time.time()
            label = self.model(image_processed)
            model_inference_time = str(round((time.time() - start_model) * 1e3, 2)) + 'ms'

            start_post_process = time.time()
            label = self.process_output(label, person)
            temp_label[y1:y2, x1:x2] = label
            labels += temp_label
            post_process_time = str(round((time.time() - start_post_process) * 1e3, 2)) + 'ms'

        _, labels = cv2.threshold(labels, 20, 255, cv2.THRESH_BINARY)

        return labels, pre_process_time, model_inference_time, post_process_time

