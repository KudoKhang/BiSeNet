import torch

from networks import *

args = get_args()

# Training
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = args.num_classes
ROOT = args.root
print(f"Using {DEVICE}")
model = BiSeNet(num_classes=NUM_CLASSES, training=False)
model = model.to(DEVICE)

ROOT = 'dataset/ibug/'
checkpoint = torch.load(os.path.join(args.pretrained, 'lastest_model_CeFiLa.pth'), map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['state_dict'])
dataset_val = FigaroDataset(ROOT, num_classes=NUM_CLASSES, mode='val', device=DEVICE)

dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False
)

_, mean_iou, f1_score = val(model, dataloader_val, 2, DEVICE)