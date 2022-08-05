from networks import *

# Training
EPOCHS = 200
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
CHECKPOINT_STEP = 2
VALIDATE_STEP = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 14

print(f"Using {DEVICE}")
model = BiSeNet(num_classes=NUM_CLASSES, training=True)
model = model.to(DEVICE)

# Dataloader for train
dataset_train = CamVidDataset(mode='train', device=DEVICE)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# Dataloader for validate
dataset_val = CamVidDataset(mode='val')
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
        data = data.cpu()
        label = label.cpu()
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
    if epoch % CHECKPOINT_STEP == 0:
        torch.save(model.state_dict(), 'checkpoints/lastest_model.pth')

    # Validate save best model
    # Save checkpoint
    if epoch % VALIDATE_STEP == 0:
        _, mean_iou = val(model, dataloader_val)
        if mean_iou > max_miou:
            max_miou = mean_iou
            print('Save best model with mIoU = {}'.format(mean_iou))
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
