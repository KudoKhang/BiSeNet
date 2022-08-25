from networks import *

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
    cv2.imshow('BiSeNet Predict', img)
    cv2.waitKey(0)

def process_folder(path, output):
    pass
#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    BSN = BSNPredict(pretrained='checkpoints/lastest_model_CeFiLa.pth')
    # img = cv2.imread('dataset/Figaro_1k/test/images/450.jpg')
    # image('src/test.jpg')
    webcam()
    # video(BSN, 'src/hair1.mp4', 'CeFiLa_')
    # time_inference('dataset/Figaro_1k/test/images/')