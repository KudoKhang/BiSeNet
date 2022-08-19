from networks.libs import *

def log2wandb(miou, loss, project, epoch, data_dir, log_dir):
    wandb.init(project=project, entity='khanghn')
    wandb.log({"mIoU: ": miou, "Loss: ": loss.data.cpu()})

    def log(weight=os.path.join(log_dir, "model_parsing_best.pth.tar"), mode='Best', data_dir=None):
        mIoU = eval(weight, data_dir)
        for key, value in mIoU.items():
            wandb.log({f"{key} [{mode}]": value})
        # save to csv
        mIoU.update({"Lr: ": lr, "Loss: ": loss.data.cpu(), "Epoch": epoch})
        values = []
        for _, value in mIoU.items():
            values.append(value)

        with open(os.path.join(log_dir, f"log_{mode}.csv"), "a") as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(values)

        Human_parsing_predictor = HumanParsing(dataset='mhp', weight=weight)

        log_image = cv2.imread(path)
        log_mask = Human_parsing_predictor.run(log_image)
        mask_img = wandb.Image(log_image[:, :, ::-1],
                               caption=f"Prediction {mode}",
                               masks={"predictions":
                                          {"mask_data": log_mask,
                                           "class_labels": class_labels}})
        wandb.log({f'mask-{mode}': mask_img})
    log(weight=os.path.join(log_dir, "model_parsing_best.pth.tar"), mode='Best', data_dir=data_dir)
    log(weight=os.path.join(log_dir, "checkpoint_last.pth.tar"), mode='Last', data_dir=data_dir)
