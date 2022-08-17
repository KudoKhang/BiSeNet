from networks import *

class BSNPredict:
    def __init__(self, NUM_CLASSES=2, pretrained='checkpoints/best_model.pth'):
        self.NUM_CLASSES = NUM_CLASSES
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BiSeNet(num_classes=self.NUM_CLASSES)
        self.model.load_state_dict(torch.load(self.pretrained, map_location=torch.device(self.device))['state_dict'])
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
        img = Image.open(image_path)
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

    def visualize(self, img, label, color = (0, 255, 0)):
        if color:
            label[:,:,0][np.where(label[:,:,0]==255)] = color[0]
            label[:,:,1][np.where(label[:,:,1]==255)] = color[1]
            label[:,:,2][np.where(label[:,:,2]==255)] = color[2]
        img_visualize = cv2.addWeighted(img, 0.6, label, 0.4, 0)
        return img_visualize

    def run(self, image_path, visualize=True):
        img_ori = cv2.imread(image_path)
        label = self.model(self.process_input(image_path))
        label = self.process_output(label, img_ori)
        img_visualize = self.visualize(img_ori, label)
        if visualize:
            return img_visualize
        return label

if __name__ == '__main__':
    BSN = BSNPredict()
    # img = BSN.run('Figaro_1k/test/images/187.jpg')
    img = BSN.run('src/7.jpg')
    cv2.imshow('result BiSeNet', img)
    cv2.waitKey(0)