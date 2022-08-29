from networks import *

class BSNPredict:
    def __init__(self, NUM_CLASSES=2, pretrained='checkpoints/best_model_900.pth', is_draw_bbox=False):
        self.NUM_CLASSES = NUM_CLASSES
        self.is_draw_bbox = is_draw_bbox
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BiSeNet(num_classes=self.NUM_CLASSES)
        self.model.load_state_dict(torch.load(self.pretrained, map_location=torch.device(self.device))['state_dict'])
        print(f'Inference with ---{pretrained}---')
        self.model = self.model.to(self.device)
        self.model.eval()

        """
            IDEA: thay detect_person = detect_head
            https://github.com/yxlijun/S3FD.pytorch
            https://github.com/deepakcrk/yolov5-crowdhuman 
        """
        # Model detect person
        self.model_detect_person = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/yolov5s.pt', force_reload=True)
        # self.model_detect_person = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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

    def visualize(self,img, label, color = (0, 255, 0)):
        if color:
            label[:,:,0][np.where(label[:,:,0]==255)] = color[0]
            label[:,:,1][np.where(label[:,:,1]==255)] = color[1]
            label[:,:,2][np.where(label[:,:,2]==255)] = color[2]
        # TODO: add color only hair via mask
        return cv2.addWeighted(img, 0.6, label, 0.4, 0)

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image path")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def draw_bbox(self, img, t):
        bbox = list(t[:, :5][np.where(t[:, 6] == 'person')][np.where(t[:, 4] > 0.4)])
        for bb in bbox:
            x1, y1, x2, y2 = np.uint32(bb[:4])
            confident = bb[4]
            img = cv2.putText(img, 'Confident: ' + str(round(confident * 100, 2)) + '%', (x1 + 2, y1 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    def detect_person(self, img):
        # TODO: FIX CASE NO PERSON
        results = self.model_detect_person(img)
        t = np.array(results.pandas().xyxy[0])
        # bbox = list(np.int32(t[:, :4][np.where(t[:, 6] == 'person')][np.where(t[:,4] > 0.7)])) # Get person have condident score > 0.7
        bbox = list(np.int32(t[:, :4][np.where(t[:, 6] == 'person')]))
        if self.is_draw_bbox:
            self.draw_bbox(img, t)
        return bbox

    def predict(self, image_path, visualize=False):
        img_ori = self.check_type(image_path)
        bbox = self.detect_person(img_ori)
        labels = np.zeros_like(img_ori)
        temp_label = labels.copy()
        for bb in bbox:
            x1, y1, x2, y2 = bb
            person = img_ori[y1: y2, x1: x2]
            label = self.model(self.process_input(person.copy()))
            label = self.process_output(label, person)
            temp_label[y1:y2, x1:x2] = label
            labels += temp_label
        _, labels = cv2.threshold(labels, 20, 255, cv2.THRESH_BINARY)
        if visualize:
            return self.visualize(img_ori, labels)
        return labels

