import os.path

from networks import *

NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BiSeNet(num_classes=NUM_CLASSES, training=True)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=torch.device(DEVICE))['state_dict'])
model = model.to(DEVICE)
model.eval()
dataset_test = FigaroDataset('Figaro_1k/', mode='test')

def test_single_image(model, dataset_test, index):
    test_image, test_label = dataset_test.__getitem__(index)
    test_image, test_label = test_image.to(DEVICE), test_label.to(DEVICE)
    test_image = test_image.unsqueeze(0)
    test_output = model(test_image)
    # Convert the (num_classes, W, H) => (W, H) with one hot decoder
    test_output = reverse_one_hot(test_output.squeeze(0))
    test_output = np.array(test_output.cpu())
    # Process label. Convert to (W, H) image
    test_label = test_label.squeeze()
    test_label = np.array(test_label.cpu())
    return test_label, test_output

def inference(model, image_path):
    normalize = transforms.Compose([
        transforms.Resize((480, 360)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_ori = cv2.imread(image_path)
    img = Image.open(image_path)
    img = normalize(img)
    img = img.to(DEVICE)
    img = img.unsqueeze(0)
    label = model(img)
    label = reverse_one_hot(label.squeeze(0))
    label = np.array(label.cpu(), dtype='uint8')
    label = label * 255
    label = 255 - label
    label = cv2.merge([label, label, label])
    label = cv2.resize(label,img_ori.shape[:2][::-1])
    final = cv2.addWeighted(img_ori, 0.8, label, 0.2, 0)

    return img_ori, label, final
# test_label, test_output = test_single_image(model, dataset_test, 2)
img, label, filter = inference(model, 'Figaro_1k/test/images/187.jpg')

print('pause')