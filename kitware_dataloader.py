import os
import yaml
import glob
import warnings
import kwcoco

import numpy as np
import ubelt as ub

from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    objs_as_dataframe,
    time_from_name,
    sanitize_str
)
from angel_system.activity_hmm.train_activity_classifier import (
    data_loader,
    compute_feats
)

#####################
# Inputs
#####################
ptg_root = "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system/"
activity_config_path = f"{ptg_root}/config/activity_labels"
recipe = "coffee"
activity_config_fn = f"{activity_config_path}/recipe_{recipe}.yaml"

data_dir = "/media/hannah.defazio/Padlock_DT/Data/notpublic/PTG/data/Coffee"
extracted_data_dir = f"{data_dir}/coffee_recordings/extracted"
activity_gt_dir = f"{data_dir}/coffee_labels/Labels"

exp_name = "coffee_base"
obj_dets_dir = f"/media/hannah.defazio/Padlock_DT/Data/notpublic/PTG/data/Coffee/object_detection_results/{exp_name}"

training_split = {
    "train_activity": [f"all_activities_{x}" for x in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 47, 48, 49]],
    "val": [f"all_activities_{x}" for x in [23, 24, 42, 46]],
    "test": [f"all_activities_{x}" for x in [20, 33, 39, 50, 51, 52, 53, 54]],
}  # Coffee specific

#####################
# Output
#####################
output_data_dir = f"{data_dir}/TCN_data/{exp_name}"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

gt_dir = f"{output_data_dir}/groundTruth"
if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)

bundle_dir = f"{output_data_dir}/splits"
if not os.path.exists(bundle_dir):
    os.makedirs(bundle_dir)

features_dir = f"{output_data_dir}/features"
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

#####################
# Create mapping.txt
#####################
print("Creating mapping...")
with open(activity_config_fn, "r") as stream:
    activity_config = yaml.safe_load(stream)
activity_labels = activity_config["labels"]

with open(f"{output_data_dir}/mapping.txt", "w") as mapping:
    for label in activity_labels:
        i = label["id"]
        label_str = label["label"]
        mapping.write(f"{i} {label_str}\n")

#####################
# Create groundtruth
#####################
print("Creating groundtruth...")
videos = training_split["train_activity"] + training_split["val"] + training_split["test"]
for video in ub.ProgIter(videos, desc="Creating groundtruth and bundles"):
    extracted_frames_dir = f"{extracted_data_dir}/{video}/_extracted/images"
    frames = glob.glob(f"{extracted_frames_dir}/*.png")

    activity_gt_fn = f"{activity_gt_dir}/{video}.csv"
    gt = activities_from_dive_csv(activity_gt_fn)
    gt = objs_as_dataframe(gt)

    with open(f"{gt_dir}/{video}.txt", "w") as gt_f:
        for frame in frames:
            frame_idx, time = time_from_name(frame) 
            matching_gt = gt.loc[(gt["start"] <= time) & (gt["end"] >= time)]
            #print('matching gt', matching_gt)
            if matching_gt.empty:
                label = "background"
                activity_label = label
            else:
                label = matching_gt.iloc[0]["class_label"]
                activity = [x for x in activity_labels[1:-1] if sanitize_str(x["full_str"]) == label]
                if not activity:
                    warnings.warn(f"Label: {label} is not in the activity labels config, ignoring")
                    continue
                activity = activity[0]
                activity_label = activity["label"]

            gt_f.write(f"{activity_label}\n")

    # Create bundles
    for split, split_videos in training_split.items():
        if video in split_videos:
            break
    with open(f"{bundle_dir}/{split}.split1.bundle", "a+") as bundle:
        bundle.write(f"{video}.txt\n")

#####################
# Create features
#####################
print("Creating features...")
pred_fnames = []
for split in training_split.keys():
    kwcoco_file = f"{obj_dets_dir}/{exp_name}_results_{split}.mscoco.json"
    dset = kwcoco.CocoDataset(kwcoco_file)

    num_classes = len(dset.cats)

    for video_id in ub.ProgIter(dset.index.videos.keys(), desc=f"Creating features for videos in {split}"):
        video = dset.index.videos[video_id]
        video_name = video["name"]
        
        image_ids = dset.index.vidid_to_gids[video_id]
        num_images = len(image_ids)
        video_dset = dset.subset(gids=image_ids, copy=True)

        (
            act_map,
            inv_act_map,
            image_activity_gt,
            image_id_to_dataset,
            label_to_ind,
            act_id_to_str,
            ann_by_image,
        ) = data_loader(video_dset, activity_config)
        X, y = compute_feats(
            act_map,
            image_activity_gt,
            image_id_to_dataset,
            label_to_ind,
            act_id_to_str,
            ann_by_image,
        )

        X.reshape(num_classes, -1)
        np.save(f"{features_dir}/{video_name}.npy", X)

print("Done!")
print(f"Saved training data to {output_data_dir}")
