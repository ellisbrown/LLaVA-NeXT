import os
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from tqdm import tqdm
import json


def main(dataset_root):
    image_folder = os.path.join(dataset_root, "images")
    combined_json_file = os.path.join(dataset_root, "data.json")
    os.makedirs(image_folder, exist_ok=True)

    # data = load_dataset("lmms-lab/LLaVA-OneVision-Data", split="train")

    # Step 1: Get All Configurations

    configs = get_dataset_config_names("lmms-lab/LLaVA-OneVision-Data")
    print(f"Available configs: {configs}")

    # Step 2: Load All Subsets

    all_data = {}

    # Loop through each configuration and load its data
    for config in tqdm(configs, desc="Downloading subsets"):
        print(f"Loading subset: {config}")
        data = load_dataset("lmms-lab/LLaVA-OneVision-Data", config, split="train")
        all_data[config] = data

    # Now `all_data` contains the loaded data for all subsets
    print(f"Loaded subsets: {list(all_data.keys())}")

    # Step 3: Combine All Subsets (Optional)
    # Combine all subsets into one dataset
    combined_data = concatenate_datasets([all_data[config] for config in all_data])

    print(f"Total number of rows in combined dataset: {len(combined_data)}")
    data = combined_data

    converted_data = []

    for da in tqdm(data, desc="Converting data"):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            json_data["image"] = f"{da['id']}.jpg"
            img = da["image"].convert("RGB")
            img.save(os.path.join(image_folder, json_data["image"]))
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)


    with open(combined_json_file, "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--dataset_root", type=str, default="data/llava_ov")

    # args = parser.parse_args()

    # dataset_root = args.dataset_root

    dataset_root = "/data/weka/ellisb/datasets/LLaVA-OneVision-Data/

    print(f"dataset_root: {dataset_root}\n")

    main(dataset_root)
