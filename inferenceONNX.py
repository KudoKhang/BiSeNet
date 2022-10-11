from networks import *

# predictor = BSNONNXPredict(pretrained='checkpoints/bisenet_no_opt.onnx')
predictor = BSNONNXPredict(pretrained='checkpoints/bisenet.onnx')

root = 'dataset/Figaro_1k/test/images/'
list_path = [root + name for name in os.listdir(root) if name.endswith(('jpg'))]
# for path in list_path[10:16]:
#     img = cv2.imread(path)
#     result = predictor.predict(img)
#     cv2.imshow('result', result)
#     cv2.waitKey(0)

img = cv2.imread(root + '959.jpg')
result = predictor.predict(img)
cv2.imshow('result', result)
cv2.waitKey(0)