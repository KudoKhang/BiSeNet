from networks import *

args = get_args()

# Training
EPOCHS = args.epoch
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch
CHECKPOINT_STEP = 1
VALIDATE_STEP = args.valid_step
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = args.num_classes
ROOT = args.root
start_epoch = 0
max_miou = 0

print(f"Using {DEVICE}")
model = BiSeNet(num_classes=NUM_CLASSES, training=False)
model = model.to(DEVICE)

checkpoint = torch.load(os.path.join(args.pretrained, 'best_model.pth'))
model.load_state_dict(checkpoint['state_dict'])
dataset_val = FigaroDataset('dataset/Figaro_1k/', num_classes=2, mode='val', device=DEVICE)

dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True
)

_, mean_iou, f1_score = val(model, dataloader_val, 2, DEVICE)
