import os
import argparse
from tqdm import tqdm
import time
from demo_model import RepairModel
from demo_prompt import *
from demo_utils import *
from defects4j_validation_test import validate_one_d4j_patches
from human_eval_validation_test import validate_one_human_eval


def repair_loop(model, prompt, file_name, folder, bug, t_chances, validator_func):
    generate_patch_path = os.path.join(folder,"generatePath")
    if not os.path.exists(generate_patch_path):
        os.makedirs(generate_patch_path, exist_ok=True)
    
    correct_patch_path = os.path.join(folder,"correctPatch")
    if not os.path.exists(correct_patch_path):
        os.makedirs(correct_patch_path, exist_ok=True)

    repair_result = []
    p_diff = {}

    start = time.time()
    print("Repairing bug {} ... ".format(file_name))

    well, all_generate_texts = model.model_predict(prompt, bug['buggy'], num_samples=t_chances)

    end = time.time()

    if well:
        for index, output in enumerate(all_generate_texts):
            diff = get_unified_diff(bug['buggy'], output)
            if diff in p_diff:
                repair_result[p_diff[diff]]['num'] += 1
                continue
            p_diff[diff] = len(repair_result)

            generate_text_file_name = file_name + "_" + str(len(repair_result)) + ".java"
            write_file(os.path.join(generate_patch_path, generate_text_file_name), output)

            is_valid = validator_func(file_name, output)
            if is_valid == "valid":
                write_file(os.path.join(correct_patch_path, generate_text_file_name), output)
            
            repair_result.append({'patch': output, 'diff': diff, 'valid': is_valid, 'num': 1})
    
    write_json(os.path.join(folder, "llm_repair.json"), repair_result)
    print("{} Unique Patches Generated for {} in {}s".format(len(repair_result), file_name, end - start))


def repair(args, model, bugs, folder, chances):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    for file_name, bug in tqdm(bugs.items(), desc='Processing bugs', unit='bug', position=0):
        if bug["dataset"] == "defects4j":
            validate_func = validate_one_d4j_patches
        elif bug["dataset"] == "human-eval-java":
            validate_func = validate_one_human_eval
        else:
            print("Unknown dataset: {}".format(args.dataset))
            return -1
        
        if args.use_retrieval == "false":
            folder_path = os.path.join(folder, file_name + "无检索")
            prompt = PROMPT_NO_RAG.format(buggy_code=bug["buggy"])
        else:
            folder_path = os.path.join(folder, file_name + "有检索")
            prompt = PROMPT_RAG.format(retrieval_buggy_code=bug["retrieval_buggy"], retrieval_fixed_code=bug["retrieval_fix"], buggy_code=bug["buggy"])
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "prompt.txt"), "w") as f:
            f.write(prompt)

        repair_loop(model, prompt, file_name, folder_path, bug, chances, validate_func)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Coder-7B")
    parser.add_argument("--chances", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset_path", type=str, default="/home/liutongxue/PycharmProject/MuGReT-DWA/Demo/demo.json")
    parser.add_argument("--folder", type=str, default="/home/liutongxue/PycharmProject/MuGReT-DWA/Demo/Results")
    parser.add_argument("--use_retrieval", type=str, default="true")
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()

    dataset = read_json(args.dataset_path)

    set_seed(args.seed)

    model = RepairModel(batch_size=args.batch_size, model_name=args.model_name)

    repair(args, model, dataset, args.folder, args.chances)


if __name__ == '__main__':
    main()
