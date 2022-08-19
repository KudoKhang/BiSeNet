from .libs import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset/Figaro_1k/", help="Path to input image")
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
start_epoch = 0
max_miou = 0

print(f"Using {DEVICE}")
