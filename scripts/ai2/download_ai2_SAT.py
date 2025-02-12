import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import concurrent.futures

# Base directory under which images and jsonl files will be saved.
BASE_DIR = "/data/weka/ellisb/datasets/SAT"
# BATCH_SIZE = 1000  # Number of examples to process per batch
BATCH_SIZE = 512  # Number of examples to process per batch

N_WORKERS = 64  # Number of concurrent workers to use for processing examples

def save_image(pil_image, split, sample_id, image_idx):
    """
    Saves the given PIL image as a JPEG into BASE_DIR/SAT/{split}/{sample_id}_{image_idx}.jpg.
    Returns the full path to the saved image.
    """
    sample_dir = os.path.join(BASE_DIR, split, str(sample_id))
    os.makedirs(sample_dir, exist_ok=True)
    file_path = os.path.join(sample_dir, f"{image_idx}.jpg")
    pil_image.save(file_path, format="JPEG")
    return file_path

def process_example(args):
    """
    Process a single example:
      - Save its images to disk.
      - Build the corresponding JSON dictionary.
    """
    idx, example, split, num_samples = args

    n_digits = len(str(num_samples))
    sample_id = str(idx).zfill(n_digits)

    images = example["image_bytes"]
    question = example["question"]
    correct_answer = example.get("correct_answer", "")
    question_type = example.get("question_type", "")

    # Save each image to disk.
    image_paths = []
    for image_idx, img in enumerate(images):
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        image_path = save_image(img, split, sample_id, image_idx)
        image_paths.append(image_path)

    # Build conversation text: one <image> token per image followed by a newline and the question.
    image_token_str = "<image>" * len(image_paths)
    human_text = f"{image_token_str}\n{question}"
    gpt_text = correct_answer

    conversations = [
        {"from": "human", "value": human_text},
        {"from": "gpt", "value": gpt_text}
    ]

    metadata = {
        "dataset": "SAT",
        "split": split,
        "num_sample": num_samples,
        "task_instruction": question,  # Change this to a more general instruction if desired.
        "question_type": question_type
    }

    sample_dict = {
        "sample_id": sample_id,
        "conversations": conversations,
        "image": image_paths,
        "choice_list": None,
        "metadata": metadata
    }
    return sample_dict

def main():
    # Load the SAT dataset (assuming it has splits like 'train' and 'validation')
    dataset = load_dataset("array/SAT", batch_size=BATCH_SIZE)
    os.makedirs(BASE_DIR, exist_ok=True)

    # Process each split separately.
    for split in dataset.keys():
        print(f"Processing split: {split}")
        split_dir = os.path.join(BASE_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        jsonl_file = os.path.join(BASE_DIR, f"{split}.jsonl")
        num_samples = len(dataset[split])

        with open(jsonl_file, "w") as out_f:
            # Process examples in batches.
            for batch_start in tqdm(range(0, num_samples, BATCH_SIZE), desc=f"Processing '{split}' in batches"):
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                # Prepare a list of tasks for this batch.
                batch_tasks = (
                    (idx, dataset[split][idx], split, num_samples)
                    for idx in range(batch_start, batch_end)
                )

                # Process the current batch concurrently.d
                with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                    batch_results = list(executor.map(process_example, batch_tasks))

                # Write each processed example as one JSON line.
                for sample_dict in batch_results:
                    out_f.write(json.dumps(sample_dict) + "\n")

        print(f"Finished processing split '{split}'. JSONL saved to: {jsonl_file}")

if __name__ == "__main__":
    main()
