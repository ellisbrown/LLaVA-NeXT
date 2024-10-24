import argparse
import copy
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai

from PIL import Image

import numpy as np
import re

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_dir", help="Base path to the video files.", required=True)
    parser.add_argument("--input_jsonl", help="Path to input JSONL file.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the output JSONL file.", required=True)
    parser.add_argument("--output_name", help="Name of the output JSONL file.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--torch_dtype",  type=str, default="half", choices=["half", "bfloat16"])
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=lambda x: (str(x).lower() == 'true'), default=False)
    return parser.parse_args()


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def clean_answer(answer: str, choices: list) -> str:
    """
    Extract the predicted answer from the model's output based on the provided choices.
    Prioritize matching choices at the beginning of the answer.
    """
    # Normalize the answer to lowercase for case-insensitive comparison
    answer = answer.lower().strip()
    
    # Normalize choices to lowercase and create a mapping from normalized to original
    normalized_choices = {choice.lower(): choice for choice in choices}
    
    # Try to match choices at the beginning of the answer
    for choice_text in normalized_choices.keys():
        pattern = r'^' + re.escape(choice_text) + r'\b'
        if re.match(pattern, answer):
            return normalized_choices[choice_text]  # Return the original choice text
    
    # If not found, look for exact matches of choices in the answer
    for choice_text in normalized_choices.keys():
        pattern = r'\b' + re.escape(choice_text) + r'\b'
        if re.search(pattern, answer):
            return normalized_choices[choice_text]  # Return the original choice text
    
    # If no match found, return empty string
    return ''


def is_correct_answer(pred, gt_answer, choices=["A", "B", "C", "D"]):
    """
    Compare predicted answer to ground truth answer, ignoring case, whitespace, and punctuation.

    Args:
        pred (str): The predicted answer.
        gt_answer (str): The ground truth answer.
        choices (list): List of answer choices.

    Returns:
        bool: True if the answers match, False otherwise.
    """
    cleaned_pred = clean_answer(pred, choices).lower()
    cleaned_gt_answer = clean_answer(gt_answer, choices).lower()
    return cleaned_pred == cleaned_gt_answer


def run_inference(args):
    """
    Run inference on the dataset using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    if args.model_path != "gpt4v":
        model_name = get_model_name_from_path(args.model_path)
        if args.overwrite:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(f"Scaling factor: {float(scaling_factor)}")
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
            else:
                model_name = "llava_qwen"

            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, args.model_base, model_name,
                load_8bit=args.load_8bit, overwrite_config=overwrite_config, torch_dtype=args.torch_dtype
            )
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, args.model_base, model_name
            )
    else:
        raise NotImplementedError("GPT-4V not supported yet.")

    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False

    # Initialize counters for total samples and correct answers
    total_samples = 0
    num_correct = 0

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.jsonl")

    input_jsonl = args.input_jsonl

    # Get the total number of lines in the input file
    with open(input_jsonl, 'r') as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_jsonl, 'r') as infile, open(answers_file, 'w') as ans_file:
        pbar = tqdm(infile, total=total_lines, desc="Processing")
        for line in pbar:
            data = json.loads(line.strip())

            # Extract fields from data
            idx = data.get('idx')
            type_ = data.get('type')
            task = data.get('task')
            question = data.get('question')
            choices = data.get('choices')
            gt_answer = data.get('gt_answer')
            filename = data.get('filename')
            source = data.get('source')

            sample_set = data.copy()

            # Increment total samples counter
            total_samples += 1

            # Construct full path to the video
            video_path = os.path.join(args.video_dir, filename)

            if not os.path.exists(video_path):
                print(f"Video file {video_path} not found.")
                continue

            # Load the video
            if args.model_path != "gpt4v":
                video, frame_time, video_time = load_video(
                    video_path, args.for_get_frames_num, 1, args.force_sample
                )
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().to(
                    dtype=getattr(torch, args.torch_dtype)
                )
                video = [video]
            else:
                # Handle GPT-4V case if needed
                raise NotImplementedError("GPT-4V not supported yet.")

            # Run inference on the video and add the output to the list
            if "gpt4v" != args.model_path:
                qs = question

                # Prepare the prompt
                if args.add_time_instruction:
                    time_instruction = (
                        f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are "
                        f"uniformly sampled from it. These frames are located at {frame_time}. "
                        "Please answer the following questions related to this video."
                    )
                    qs = f'{time_instruction}\n{qs}'

                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = copy.deepcopy(conv_templates[args.conv_mode])
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()
                if tokenizer.pad_token_id is None:
                    if "qwen" in tokenizer.name_or_path.lower():
                        print("Setting pad token to bos token for qwen model.")
                        tokenizer.pad_token_id = 151643

                attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                cur_prompt = question
            else:
                prompt = question
            if "gpt4v" != args.model_path:

                with torch.inference_mode():
                    if "mistral" in model.config._name_or_path.lower():
                        output_ids = model.generate(
                            inputs=input_ids,
                            images=video,
                            attention_mask=attention_masks,
                            modalities="video",
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=1024,
                            top_p=0.1,
                            num_beams=1,
                            use_cache=True
                        )
                    else:
                        output_ids = model.generate(
                            inputs=input_ids,
                            images=video,
                            attention_mask=attention_masks,
                            modalities="video",
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=1024,
                            top_p=0.1,
                            num_beams=1,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria]
                        )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            else:
                openai.api_key = args.api_key  # Your API key here

                max_num_retries = 0
                retry = 5
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            f"These are frames from a video that I want to upload. Answer me one question of this video: {prompt}",
                            *map(lambda x: {"image": x, "resize": 336}, video[0::interval]),
                        ],
                    },
                ]
                params = {
                    "model": "gpt-4-vision-preview", #gpt-4-1106-vision-preview
                    "messages": PROMPT_MESSAGES,
                    "max_tokens": 1024,
                }
                sucess_flag=False
                while max_num_retries < retry:
                    try:
                        result = openai.ChatCompletion.create(**params)
                        outputs = result.choices[0].message.content
                        sucess_flag = True
                        break
                    except Exception as inst :
                        if 'error' in dir(inst):
                            # import pdb;pdb.set_trace()
                            if  inst.error.code == 'rate_limit_exceeded':
                                if "TPM" in inst.error.message:
                                    time.sleep(30)
                                    continue
                                else:
                                    import pdb;pdb.set_trace()
                            elif inst.error.code == 'insufficient_quota':
                                print(f'insufficient_quota key')
                                exit()
                            elif inst.error.code == 'content_policy_violation':
                                print(f'content_policy_violation')
                                system_error = "content_policy_violation"

                                break
                            print('Find error message in response: ',str(inst.error.message), 'error code: ', str(inst.error.code))

                        continue
                if not sucess_flag:
                    raise RuntimeError(f'Calling OpenAI failed after retrying for {max_num_retries} times. Check the logs for details.')

                if system_error not in ['content_policy_violation', ""]:
                    import pdb;pdb.set_trace()


            if "mistral" not in cfg_pretrained._name_or_path.lower():
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            sample_set['pred'] = outputs

            # Compare 'pred' to 'gt_answer' to set 'correct'
            correct = is_correct_answer(outputs, gt_answer, choices)
            sample_set['correct'] = correct

            # Increment correct answers counter if applicable
            if correct:
                num_correct += 1

            # Write the sample_set to the output JSONL file
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()

            # Update the progress bar with running accuracy
            if total_samples > 0:
                running_accuracy = (num_correct / total_samples) * 100
                pbar.set_postfix(accuracy=f"{running_accuracy:.2f}%")

    # After processing all samples, print summary statistics
    print("\nEvaluation Results:")
    print(f"Total samples: {total_samples}")
    print(f"Number correct: {num_correct}")
    if total_samples > 0:
        accuracy = (num_correct / total_samples) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No samples were processed.")

    # Save summary statistics to a CSV file
    summary_stats_file = os.path.join(args.output_dir, "summary_stats.csv")
    with open(summary_stats_file, 'w') as csv_file:
        csv_file.write("Total samples,Number correct,Accuracy\n")
        if total_samples > 0:
            csv_file.write(f"{total_samples},{num_correct},{accuracy:.2f}\n")
        else:
            csv_file.write(f"{total_samples},{num_correct},N/A\n")
    
    print(f"\nSummary statistics saved to {summary_stats_file}")
    print(f"Output JSONL file saved to {answers_file}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
