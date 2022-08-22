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

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def predict(self, image_path, visualize=True):
        img_ori = self.check_type(image_path)
        label = self.model(self.process_input(img_ori.copy()))
        label = self.process_output(label, img_ori)
        img_visualize = self.visualize(img_ori, label)
        if visualize:
            return img_visualize
        return label

#---------------------------------------------------------------------------------------------------------------------

def webcam():
    print("Using webcam, press [q] to exit, press [s] to save")
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        start = time.time()
        frame = BSN.predict(frame)
        # FPS
        fps = round(1 / (time.time() - start), 2)
        cv2.putText(frame, "FPS : " + str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow('Prediction', frame + 30)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            os.makedirs('results/', exist_ok=True)
            cv2.imwrite('results/' + str(time.time()) + '.jpg', frame)
        if k == ord('q'):
            break

def video(path_video='src/video1.mp4', name='result_'):
    print('Processing video... \n Please wait...')
    cap = cv2.VideoCapture(path_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = 30
    os.makedirs('results/', exist_ok=True)
    out = cv2.VideoWriter(f'results/{name}' + path_video.split('/')[-1], cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    while True:
        _, frame = cap.read()
        try:
            frame = BSN.predict(frame)
            out.write(frame)
        except:
            out.release()
            break
    out.release()
    print('Done!')

def time_inference(root):
    path_name = [os.path.join(root, name) for name in os.listdir(root) if name.endswith('jpg')]
    start = time.time()
    for path in tqdm(path_name):
        BSN.predict(path)
    end = (time.time() - start) / len(path_name)
    print(f'Avg time inference {len(path_name)} images is:', round(end * 1e3), 'ms')

def image(path='dataset/Figaro_1k/test/images/971.jpg'):
    start = time.time()
    img = BSN.predict(path)
    print('Time inference: ', round((time.time() - start) * 1e3, 2), 'ms')
    cv2.imshow('Result BiSeNet', img)
    cv2.waitKey(0)

def process_folder(path, output):
    pass

if __name__ == '__main__':
    BSN = BSNPredict(pretrained='checkpoints/lastest_model_CeFiLa.pth')
    # image('dataset/Figaro_1k/test/images/79.jpg')
    # webcam()
    video('src/hair1.mp4', 'CeFiLa_')
    # time_inference('dataset/Figaro_1k/test/images/')