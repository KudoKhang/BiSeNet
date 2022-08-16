import os.path

from networks import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="Figaro_1k/", help="Path to input image")
    parser.add_argument("--num_classes", type=int, default=2, help="")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=200)
    args = parser.parse_args()
    print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    print("🎵 hhey, arguments are here if you need to check 🎵")
    for arg in vars(args):
        print("{:>15}: {:>30}".format(str(arg), str(getattr(args, arg))))
    print()
    return args

args = get_args()

# Training
EPOCHS = args.epoch
LEARNING_RATE = 0.0001
BATCH_SIZE = args.batch
CHECKPOINT_STEP = 2
VALIDATE_STEP = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = args.num_classes
ROOT = args.root

print(f"Using {DEVICE}")
model = BiSeNet(num_classes=NUM_CLASSES, training=True)
model = model.to(DEVICE)

# Dataloader for train
# dataset_train = CamVidDataset(mode='train', num_classes=NUM_CLASSES, device=DEVICE)
dataset_train = FigaroDataset(ROOT, num_classes=NUM_CLASSES, mode='train', device=DEVICE)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# Dataloader for validate
# dataset_val = CamVidDataset(mode='val')
dataset_val = FigaroDataset('Figaro_1k/', num_classes=NUM_CLASSES, mode='val', device=DEVICE)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=True
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

loss_func = torch.nn.CrossEntropyLoss()


# Loop for training
torch.cuda.empty_cache()

for epoch in range(EPOCHS):
    model.train()
    tq = tqdm(total=len(dataloader_train) * BATCH_SIZE)
    tq.set_description('Epoch {}/{}'.format(epoch, EPOCHS))

    loss_record = []
    max_miou = 0

    for i, (data, label) in enumerate(dataloader_train):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        output, output_sup1, output_sup2 = model(data)
        loss1 = loss_func(output, label)
        loss2 = loss_func(output_sup1, label)
        loss3 = loss_func(output_sup2, label)

        # Combine 3 loss
        loss = loss1 + loss2 + loss3
        tq.update(BATCH_SIZE)
        tq.set_postfix(loss='%.6f' % loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    tq.close()
    loss_train_mean = np.mean(loss_record)
    print('loss for train : %f' % (loss_train_mean))

    # Save checkpoint
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints/', exist_ok=True)

    if epoch % CHECKPOINT_STEP == 0:
        torch.save(model.state_dict(), 'checkpoints/lastest_model.pth')

    # Validate save best model
    # Save checkpoint
    if epoch % VALIDATE_STEP == 0:
        _, mean_iou = val(model, dataloader_val, NUM_CLASSES, DEVICE)
        if mean_iou > max_miou:
            max_miou = mean_iou
            print('Save best model with mIoU = {}'.format(mean_iou))
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
