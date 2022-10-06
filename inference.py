import time

from networks import *
import termplotlib as tpl

def show_log(show_figure=False):
    total_time = str(int(detection_person_time.split('.')[0]) + int(pre_time.split('.')[0]) + int(model_time.split('.')[0]) + int(post_time.split('.')[0])) + 'ms'
    detection_person_time_percent = str(round(int(detection_person_time.split('.')[0]) * 100 / int(total_time[:-2]), 2)) + '%'
    pre_time_percent              = str(round(int(pre_time.split('.')[0]) * 100 / int(total_time[:-2]), 2)) + '%'
    model_time_percent            = str(round(int(model_time.split('.')[0]) * 100 / int(total_time[:-2]), 2)) + '%'
    post_time_percent             = str(round(int(post_time.split('.')[0]) * 100 / int(total_time[:-2]), 2)) + '%'

    logs = f"{'+'*100}\n{'Detect person':<20} {'Pre-process':<20} {'Model Inference':20} {'Post-process':<20} {'Total time':<20}\n{'-'*100}\n{detection_person_time:<20} {pre_time:<20} {model_time:<20} {post_time:<20} {total_time:<20}\n{detection_person_time_percent:<20} {pre_time_percent:<20} {model_time_percent:<20} {post_time_percent:<20} {'100%':<20}\n"
    print(logs)
    # sys.stdout.write('\r'+logs) # Print and replace output

    if show_figure:
        print(f"{'-'*100} \n{'---Visualization---':>35}")
        fig = tpl.figure()
        fig.barh([int(detection_person_time.split('.')[0]), int(pre_time.split('.')[0]), int(model_time.split('.')[0]), int(post_time.split('.')[0])], ["Detect Person", "Pre-Process", "Mode Inference", "Post-Process"], force_ascii=True)
        fig.show()

if __name__ == '__main__':
    # Init model
    BSN = BSNPredict(pretrained='checkpoints/best_model_CeFiLaIb.pth')
    model_detection_person = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Process with folder
    root = 'dataset/Figaro_1k/test/images/'
    list_img = [root + name for name in os.listdir(root) if name.endswith('jpg')]
    # for path in list_img:
    #     image = cv2.imread(path)
    #
    #     # Detect person by Yolov5
    #     start_detection_person = time.time()
    #     results = model_detection_person(image)
    #     t = np.array(results.pandas().xyxy[0])
    #     bbox = list(np.int32(t[:, :4][np.where(t[:, 6] == 'person')]))
    #     detection_person_time = str(round((time.time() - start_detection_person) * 1e3, 2)) + 'ms'
    #
    #     label, pre_time, model_time, post_time = BSN.predict(image, bbox)
    #
    #     show_log()


    # Process with image
    image = cv2.imread(root + '3.jpg')
    start_detection_person = time.time()
    results = model_detection_person(image)
    t = np.array(results.pandas().xyxy[0])
    bbox = list(np.int32(t[:, :4][np.where(t[:, 6] == 'person')]))
    detection_person_time = str(round((time.time() - start_detection_person) * 1e3, 2)) + 'ms'

    label, pre_time, model_time, post_time = BSN.predict(image, bbox)
    show_log(show_figure=True)
    # cv2.imshow('mask', label)
    # cv2.waitKey(0)