from networks import *
from alive_progress import alive_bar
args = get_args()

# Config hyperparameter
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

print(f"Device using: {DEVICE}")

# Init model
model = BiSeNet(num_classes=NUM_CLASSES, training=True)
model = model.to(DEVICE)

# Load pretrained if exist
if os.path.exists(os.path.join(args.pretrained, 'lastest_model.pth')):
    checkpoint = torch.load(os.path.join(args.pretrained, 'lastest_model.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    miou = checkpoint['miou']
    print('Resume training from ---{}--- have mIoU = {}, start at epoch: {} \n'.format(args.pretrained, miou, start_epoch))

# Dataloader for train
dataset_train = FigaroDataset(ROOT, num_classes=NUM_CLASSES, mode='train', device=DEVICE)
dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )

# Dataloader for validate
dataset_val = FigaroDataset(ROOT, num_classes=NUM_CLASSES, mode='val', device=DEVICE)
dataloader_val = DataLoader(
        dataset_val,
        batch_size=1, # TODO: Code eval support each batch
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=True
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
loss_func = torch.nn.CrossEntropyLoss()

# Loop for training
torch.cuda.empty_cache()

# Init section in Wandb
wandb.init(project='Hair_segmentation', entity='khanghn')

for epoch in range(start_epoch, EPOCHS):
    model.train()
    loss_record = []
    with alive_bar(total=len(dataloader_train), theme='musical', length=100) as bar:
        for i, (data, label) in enumerate(dataloader_train):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            output, output_sup1, output_sup2 = model(data)
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)

            # Combine 3 loss
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
            bar()
    loss_train_mean = np.mean(loss_record)
    print('loss for train : %f' % (loss_train_mean))

    # Save checkpoint
    if not os.path.exists(args.pretrained):
        os.makedirs(args.pretrained, exist_ok=True)

    if epoch % CHECKPOINT_STEP == 0:
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'miou': max_miou
        }
        torch.save(states, f'{args.pretrained}/lastest_model.pth')

    # Save best checkpoint via mIoU
    if epoch % VALIDATE_STEP == 0:
        checkpoint = torch.load(os.path.join(args.pretrained, 'lastest_model.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        accuracy, mean_iou, f1_score = val(model, dataloader_val, NUM_CLASSES, DEVICE)
        if mean_iou > max_miou:
            max_miou = mean_iou
            print('---Save best model with mIoU = {}--- \n'.format(mean_iou))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'miou': max_miou
            }
            torch.save(states, f'{args.pretrained}/best_model.pth')
        else:
            print('---Save Failed---')

    # Log metrics to wandb
    wandb.log({"mIoU: ": max_miou,
               "Accuracy ": accuracy,
               "F1 Score ": f1_score,
               "Loss: ": loss_train_mean})