from .libs import *

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

def compute_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

def val(model, dataloader, NUM_CLASSES=2, device='cpu'):
    accuracy_arr = []

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        model.to(device)
        model.eval()
        print('Starting validate')

        for i, (val_data, val_label) in enumerate(dataloader):
            val_data = val_data.to(device)
            # The output of model is (1, num_classes, W, H) => (num_classes, W, H)
            val_output = model(val_data).squeeze()
            # Convert the (num_classes, W, H) => (W, H) with one hot decoder
            val_output = reverse_one_hot(val_output)
            val_output = np.array(val_output.to(device))
            # Process label. Convert to (W, H) image
            val_label = val_label.squeeze()
            val_label = np.array(val_label.to(device))
            # Compute accuracy and iou
            accuracy = compute_accuracy(val_output, val_label)
            hist += fast_hist(val_label.flatten(), val_output.flatten(), NUM_CLASSES)
            # Append for calculate
            accuracy_arr.append(accuracy)
        miou_list = per_class_iu(hist)[:-1]
        mean_accuracy, mean_iou = np.mean(accuracy_arr), np.mean(miou_list)
        print('Mean accuracy: {} Mean IoU: {}'.format(mean_accuracy, mean_iou))
        return mean_accuracy, mean_iou
