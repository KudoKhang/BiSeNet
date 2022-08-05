from .libs import *

class CamVidDataset(Dataset):
    color_encoding = [
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ]

    def __init__(self, mode='train', num_classes=14, device='cpu'):
        self.mode = mode
        self.device = device
        self.num_classes = num_classes
        # Normailization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.DATA_PATH = os.path.join(os.getcwd(), 'CamVid/')

        self.train_path, self.val_path, self.test_path = [os.path.join(self.DATA_PATH, x) for x in
                                                          ['train', 'val', 'test']]

        if self.mode == 'train':
            self.data_files = self.get_files(self.train_path)
            self.label_files = [self.get_label_file(f, 'train', 'train_labels') for f in self.data_files]
        elif self.mode == 'val':
            self.data_files = self.get_files(self.val_path)
            self.label_files = [self.get_label_file(f, 'val', 'val_labels') for f in self.data_files]
        elif self.mode == 'test':
            self.data_files = self.get_files(self.test_path)
            self.label_files = [self.get_label_file(f, 'test', 'test_labels') for f in self.data_files]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def get_files(self, data_folder):
        """
            Return all files in folder with extension
        """
        return glob("{}/*.{}".format(data_folder, 'png'))

    def get_label_file(self, data_path, data_dir, label_dir):
        """
            Return label path for data_path file
        """
        data_path = data_path.replace(data_dir, label_dir)
        fname, ext = data_path.split('.')
        return "{}_L.{}".format(fname, ext)

    def image_loader(self, data_path, label_path):
        data = Image.open(data_path)
        label = Image.open(label_path)

        return data, label

    def label_for_cross_entropy(self, label):
        """
            Convert label image to matrix classes for apply cross entropy loss.
            Return semantic index, label in enumemap of H x W x class
        """
        semantic_map = np.zeros(label.shape[:-1])
        # Fill all value with class 13 - default for all pixels
        semantic_map.fill(self.num_classes - 1)
        # Fill the pixel with correct class
        for class_index, color_info in enumerate(self.color_encoding):
            color = color_info[1]
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_index
        return semantic_map

    def __getitem__(self, index):
        """
            Args:
            - index (``int``): index of the item in the dataset
            Returns:
            A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
            of the image.
        """

        data_path, label_path = self.data_files[index], self.label_files[index]
        img, label = self.image_loader(data_path, label_path)

        # Apply normalization in img
        img = self.normalize(img)
        # Convert label for cross entropy
        label = np.array(label)
        label = self.label_for_cross_entropy(label)
        label = torch.from_numpy(label).long()

        return img, label

    def __len__(self):
        # return len(os.path.join(self.DATA_PATH))
        return len(self.DATA_PATH)

    def reverse_one_hot(image):
        # Convert output of model to predicted class
        image = image.permute(1, 2, 0)
        x = torch.argmax(image, dim=-1)
        return x

    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(hist):
        epsilon = 1e-5
        return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

    def val(model, dataloader):
        accuracy_arr = []

        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

        with torch.no_grad():
            model.eval()
            print('Starting validate')

            for i, (val_data, val_label) in enumerate(dataloader):
                val_data = val_data.to(self.device)
                # The output of model is (1, num_classes, W, H) => (num_classes, W, H)
                val_output = model(val_data).squeeze()
                # Convert the (num_classes, W, H) => (W, H) with one hot decoder
                val_output = reverse_one_hot(val_output)
                val_output = np.array(val_output.to(self.device))
                # Process label. Convert to (W, H) image
                val_label = val_label.squeeze()
                val_label = np.array(val_label.to(self.device))
                # Compute accuracy and iou
                accuracy = compute_accuracy(val_output, val_label)
                hist += fast_hist(val_label.flatten(), val_output.flatten(), NUM_CLASSES)
                # Append for calculate
                accuracy_arr.append(accuracy)
            miou_list = per_class_iu(hist)[:-1]
            mean_accuracy, mean_iou = np.mean(accuracy_arr), np.mean(miou_list)
            print('Mean accuracy: {} Mean IoU: {}'.format(mean_accuracy, mean_iou))
            return mean_accuracy, mean_iou

