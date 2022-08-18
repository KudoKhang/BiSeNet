from networks import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset/Figaro_1k_png/", help="Path to input image")
    parser.add_argument("--num_classes", type=int, default=2, help="")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--pretrained", type=str, default='checkpoints/')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--valid-step", type=int, default=1)
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
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch
CHECKPOINT_STEP = 1
VALIDATE_STEP = args.valid_step
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = args.num_classes
ROOT = args.root

print(f"Using {DEVICE}")
model = BiSeNet(num_classes=NUM_CLASSES, training=True)
model = model.to(DEVICE)

if os.path.exists(os.path.join(args.pretrained, 'lastest_model.pth')):
    checkpoint = torch.load(os.path.join(args.pretrained, 'lastest_model.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    miou = checkpoint['miou']
    print('Resume training from /{}/ have mIoU = {}, start at epoch: {} \n'.format(args.pretrained, miou, start_epoch))

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
dataset_val = FigaroDataset(ROOT, num_classes=NUM_CLASSES, mode='val', device=DEVICE)
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
    if not os.path.exists(args.pretrained):
        os.makedirs(args.pretrained, exist_ok=True)

    if epoch % CHECKPOINT_STEP == 0:
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'miou': max_miou
        }
        torch.save(states, f'{args.pretrained}/lastest_model.pth')

    # Save checkpoint
    if epoch % VALIDATE_STEP == 0:
        model.load_state_dict(torch.load(f'{args.pretrained}/lastest_model.pth')['state_dict'])
        _, mean_iou = val(model, dataloader_val, NUM_CLASSES)
        if mean_iou > max_miou:
            max_miou = mean_iou
            print('Save best model with mIoU = {} \n'.format(mean_iou))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'miou': max_miou
            }
            torch.save(states, f'{args.pretrained}/best_model.pth')