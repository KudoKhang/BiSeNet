import numpy as np

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

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=True) -> torch.Tensor:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def val(model, dataloader, NUM_CLASSES=2, device='cuda'):
    accuracy_arr = []
    f1_arr = []
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        model.eval()
        print('Starting validate...')

        for i, (val_data, val_label) in tqdm(enumerate(dataloader)):
            val_data = val_data.to(device)
            val_label = val_label.to(device)
            val_output = model(val_data).squeeze()
            val_output = reverse_one_hot(val_output)
            val_output_f1 = torch.clone(val_output)
            val_output = np.array(val_output.cpu())
            val_label = val_label.squeeze()
            f1_score = f1_loss(val_label, val_output_f1)
            val_label = np.array(val_label.cpu())
            accuracy = compute_accuracy(val_output, val_label)
            hist += fast_hist(val_label.flatten(), val_output.flatten(), NUM_CLASSES)
            accuracy_arr.append(accuracy)
            f1_arr.append(np.array(f1_score.cpu()))
        miou_list = per_class_iu(hist)[:-1]
        mean_accuracy, mean_iou, f1_score = np.mean(accuracy_arr), np.mean(miou_list), np.mean(f1_arr)
        print('Mean accuracy: {} -- Mean IoU: {} -- F1 Score: {}'.format(mean_accuracy, mean_iou, f1_score))
        return mean_accuracy, mean_iou, f1_score

