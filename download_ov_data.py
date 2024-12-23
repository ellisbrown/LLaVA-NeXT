import os
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from tqdm import tqdm
import json


def clean_subset_name(name):
    """Clean subset name for use as directory name."""
    return ''.join(c if c.isalnum() else '_' for c in name).strip('_')


def process_subset(config, dataset_root):
    """Process a single subset of the dataset."""
    # Create paths
    cleaned_config = clean_subset_name(config)
    subset_image_folder = os.path.join(dataset_root, "images", cleaned_config)
    subset_json_file = os.path.join(subset_image_folder, "data.json")
    os.makedirs(subset_image_folder, exist_ok=True)

    # Load subset data
    print(f"Loading subset: {config}")
    data = load_dataset("lmms-lab/LLaVA-OneVision-Data", config, split="train")

    converted_data = []

    # Process each item in the subset
    for da in tqdm(data, desc=f"Processing {config}"):
        json_data = {
            "id": da["id"],
            "conversations": da["conversations"]
        }

        if da["image"] is not None:
            # For subset-specific json, use relative path within subset
            img_name = os.path.basename(da["id"])
            if not img_name.endswith(".jpg"):
                img_name = f"{img_name}.jpg"

            json_data["image"] = img_name

            img_path = os.path.join(subset_image_folder, json_data["image"])

            img = da["image"].convert("RGB")
            img.save(img_path)

        converted_data.append(json_data)

    # Save subset-specific json
    with open(subset_json_file, "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    return converted_data, cleaned_config


def main(dataset_root):
    # Create base directories
    os.makedirs(os.path.join(dataset_root, "images"), exist_ok=True)

    # Get all configurations
    configs = get_dataset_config_names("lmms-lab/LLaVA-OneVision-Data")
    print(f"Available configs: {configs}")

    # Process each subset and collect data for combined json
    all_converted_data = []

    # for config in configs:
    for config in tqdm(configs, desc="Processing all subsets"):
        subset_data, cleaned_config = process_subset(config, dataset_root)

        # For combined json, update image paths to include subset directory
        for item in subset_data:
            if "image" in item:
                item["image"] = os.path.join("images", cleaned_config, item["image"])
                item["subset"] = config

        all_converted_data.extend(subset_data)

    # Save combined json file
    combined_json_file = os.path.join(dataset_root, "data.json")
    with open(combined_json_file, "w") as f:
        json.dump(all_converted_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--dataset_root", type=str, default="data/llava_ov")

    # args = parser.parse_args()

    # dataset_root = args.dataset_root

    dataset_root = "/data/weka/ellisb/datasets/LLaVA-OneVision-Data/"

    print(f"dataset_root: {dataset_root}\n")

    main(dataset_root)
