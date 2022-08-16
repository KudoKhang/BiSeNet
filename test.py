import os.path

from networks import *

NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BiSeNet(num_classes=NUM_CLASSES, training=True)
model = model.to(DEVICE)
model.load_state_dict(torch.load('checkpoints/lastest_model.pth'))

dataset_test = FigaroDataset('Figaro_1k/', mode='test')
dataloader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=True
)

val(model, dataloader_test)


def img_show(img):
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.show()


def test_single_image(model, dataset_test, index):
    test_image, test_label = dataset_test.__getitem__(index)
    test_image, test_label = test_image.to(DEVICE), test_label.to(DEVICE)

    model.eval()

    test_output = model(test_image.unsqueeze(0))
    # Convert the (num_classes, W, H) => (W, H) with one hot decoder
    test_output = reverse_one_hot(test_output.squeeze(0))
    test_output = np.array(test_output.cpu())
    # Process label. Convert to (W, H) image
    test_label = test_label.squeeze()
    test_label = np.array(test_label.cpu())
    return test_label, test_output

test_label, test_output = test_single_image(model, dataset_test, 1)

print('pause')