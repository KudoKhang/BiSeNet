from alive_progress import alive_bar
from networks import *

BSN = BSNPredict(pretrained='checkpoints/lastest_model_CeFiLaIb.pth')

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, device, is_training=True) -> torch.Tensor:
    y_true = y_true.to(device).flatten()
    y_pred = y_pred.to(device).flatten()

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

def eval_model(folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f1score = []

    convert_tensor = transforms.ToTensor()

    # folder structure: ----images---*.jpg
    #                   |
    #                   ----masks---*.png

    list_ids = [name[:-4] for name in os.listdir(os.path.join(folder, 'images')) if name.endswith('jpg')]

    print(f'Start evaluation with {len(list_ids)} images from folder ---{folder}---...')

    time_inference = []
    for id in tqdm(list_ids):
        start = time.time()
        mask_predict = BSN.predict(os.path.join(folder, 'images', str(id) + '.jpg'), visualize=False)
        time_inference.append(time.time() - start)

        mask_predict = cv2.cvtColor(mask_predict, cv2.COLOR_BGR2GRAY)
        mask_predict = convert_tensor(mask_predict).squeeze()

        mask_gt = Image.open(os.path.join(folder, 'masks', str(id) + '.png')).convert('L')
        mask_gt = convert_tensor(mask_gt).squeeze()

        f1 = f1_loss(mask_gt, mask_predict, device)

        f1score.append(f1.detach().cpu())

    print('-'*40)
    print(f"Time inference 1 image: {np.mean(time_inference)}")
    return np.mean(f1score)

if __name__ == "__main__":
    f1score = eval_model('dataset/Figaro_1k/test')
    print("-"*40)
    print(f"F1 Score: {f1score}")