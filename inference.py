from networks import *

if __name__ == '__main__':
    image = cv2.imread('dataset/Figaro_1k/test/images/51.jpg')
    BSN = BSNPredict(pretrained='checkpoints/best_model_CeFiLaIb.pth')

    # Detect person by yolov5
    model_detection_person = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model_detection_person(image)
    t = np.array(results.pandas().xyxy[0])
    bbox = list(np.int32(t[:, :4][np.where(t[:, 6] == 'person')]))

    label = BSN.predict(image, bbox)
    cv2.imshow('mask', label)
    cv2.waitKey(0)
