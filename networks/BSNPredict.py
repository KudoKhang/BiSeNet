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

        # Model detect person
        self.model_detect_person = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
        # img = Image.fromarray(image_path)
        img = normalize(img)
        img = img.to(self.device)
        return img.unsqueeze(0)

    def process_output(self, label, img):
        label = self.reverse_one_hot(label.squeeze(0))
        label = np.array(label.cpu(), dtype='uint8')
        label = 255 - (label * 255)
        label = cv2.merge([label, label, label])
        label = cv2.resize(label, img.shape[:2][::-1])
        return label

    def visualize(self,img, label, color = (0, 255, 0)):
        if color:
            label[:,:,0][np.where(label[:,:,0]==255)] = color[0]
            label[:,:,1][np.where(label[:,:,1]==255)] = color[1]
            label[:,:,2][np.where(label[:,:,2]==255)] = color[2]
        img_visualize = cv2.addWeighted(img, 0.6, label, 0.4, 0)
        return img_visualize

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def draw(self,img, bbox):
        for bb in bbox:
            x1, y1, x2, y2 = bb
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

    def detect_person(self, img):
        results = self.model_detect_person(img)
        t = results.pandas().xyxy[0]
        bbox = list(np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 6] == 'person')]))
        self.draw(img, bbox)
        return bbox

    def predict(self, image_path, visualize=True):
        img_ori = self.check_type(image_path)
        bbox = self.detect_person(img_ori)
        labels = np.zeros_like(img_ori)
        for bb in bbox:
            x1, y1, x2, y2 = bb
            person = img_ori[y1: y2, x1: x2]
            label = self.model(self.process_input(person.copy()))
            label = self.process_output(label, person)
            temp_label = labels.copy()
            temp_label[y1:y2, x1:x2] = label
            labels += temp_label
        if visualize:
            return self.visualize(img_ori, labels)
        return labels
