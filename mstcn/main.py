import argparse
import os
import random
from pathlib import Path

import torch

from mstcn.batch_gen import BatchGenerator
from mstcn.model import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--action", default="train")
parser.add_argument("--dataset", default="gtea")
parser.add_argument("--split", default="1")

args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 50

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2
elif args.dataset == "bike_students_blip2":
    features_dim = 256


DATA_DIR = Path("./data")
assert DATA_DIR.exists()

vid_train_list_file = DATA_DIR / args.dataset / "splits" / f"train.split{args.split}.bundle"
vid_test_list_file = DATA_DIR / args.dataset / "splits" / f"test.split{args.split}.bundle"

features_dir = DATA_DIR / args.dataset / "features"
gt_dir = DATA_DIR / args.dataset / "groundTruth"
mapping_file = DATA_DIR / args.dataset / "mapping.txt"

model_dir = Path("./models") / args.dataset / f"split_{args.split}"
results_dir = Path("./results") / args.dataset / f"split_{args.split}"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

action_name_to_id: dict[str, int] = {}

with open(mapping_file, "r") as f:
    for line in f.readlines():
        act_id, act_name = line.split()
        action_name_to_id[act_name] = int(act_id)

num_classes = len(action_name_to_id)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(
        num_classes, action_name_to_id, gt_dir, features_dir, sample_rate
    )
    batch_gen.read_data(vid_train_list_file)
    trainer.train(
        model_dir,
        batch_gen,
        num_epochs=num_epochs,
        batch_size=bz,
        learning_rate=lr,
        device=device,
    )

if args.action == "predict":
    trainer.predict(
        model_dir,
        results_dir,
        features_dir,
        vid_test_list_file,
        num_epochs,
        action_name_to_id,
        device,
        sample_rate,
    )
