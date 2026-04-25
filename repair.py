import os
import argparse
from tqdm import tqdm
import time
from model import RepairModel, RepairModelOllama
from prompt import *
from utils import *
from Dataset.parse import clean_parse_d4j, clean_parse_human_eval
from Dataset.validate_d4j import validate_one_d4j_patches
from Dataset.validate_human_eval_java import validate_one_human_eval


def repair_loop(model, prompt, file_name, folder, bug, t_chances, validator_func):
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
            is_valid = validator_func(file_name, file_name + "_" + str(len(repair_result)) + ".java", folder, output)
            repair_result.append({'output': output,
                                  'generate_text': all_generate_texts[index],
                                  'diff': diff,
                                  'valid': is_valid,
                                  'num': 1
                                  })

    print("{} Unique Patches Generated for {} in {}s".format(len(repair_result), file_name, end - start))

    return len(all_generate_texts), repair_result


def repair(args, model, bugs, folder, correct_patch_folder, prompt, chances, validate_func):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(correct_patch_folder):
        os.makedirs(correct_patch_folder, exist_ok=True)

    parent_dir = os.path.dirname(folder)
    with open(os.path.join(parent_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join(parent_dir, "args.txt"), "w") as f:
        f.write(str(args))

    lm_repair_path = os.path.join(parent_dir, "lm_repair.json")
    if os.path.exists(lm_repair_path):
        result = read_json(lm_repair_path)
    else:
        result = {}

    total_generated = 0
    total_unique = 0
    total_plausible = 0
    total_project_plausible = 0

    start_t = time.time()
    for file_name, bug in tqdm(bugs.items(), desc='Processing bugs', unit='bug', position=0):
        combined_prompt = prompt.format(retrieval_buggy_code=bug["C2LLM_retrieval"]["buggy_function"], retrieval_fixed_code=bug["C2LLM_retrieval"]["fixed_function"], buggy_code=bug["buggy"])
        n_generated, result[file_name] = repair_loop(model, combined_prompt, file_name, folder, bug, chances, validate_func)

        if n_generated >= 1:
            total_generated += n_generated
            total_unique += len(result[file_name])
            for patch in result[file_name]:
                if patch['valid'] == 'valid':
                    file_path = os.path.join(correct_patch_folder, file_name)
                    if not os.path.exists(file_path):
                        os.makedirs(file_path, exist_ok=True)
                    patch_name = file_name + "_" + str(len(os.listdir(str(file_path)))) + ".java"
                    try:
                        with open(file_path + "/" + patch_name, 'w') as f:
                            f.write(patch['output'])
                    except Exception as e:
                        print(f"Error: {e}")
                    total_plausible += 1
            for patch in result[file_name]:
                if patch['valid'] == 'valid':
                    total_project_plausible += 1
                    break

        write_json(lm_repair_path, result)
    end_t = time.time()

    with open(os.path.join(parent_dir, "stats.txt"), "w") as f:
        f.write("Total generated: {}\n".format(total_generated))
        f.write("Total unique: {}\n".format(total_unique))
        f.write("Total plausible: {}\n".format(total_plausible))
        f.write("Total Project plausible: {}\n".format(total_project_plausible))
        f.write("Total time: {}\n".format(end_t - start_t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Coder-7B")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="defects4j")
    parser.add_argument("--chances", type=int, default=20)
    parser.add_argument("--folder", type=str, default="/home/liutongxue/PycharmProject/MuGReT-DWA/Results/generatePatch")
    parser.add_argument("--correct_patch_folder", type=str, default="/home/liutongxue/PycharmProject/MuGReT-DWA/Results/correctPatch")
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()
    if args.dataset == "defects4j":
        dataset = clean_parse_d4j()
        validate_func = validate_one_d4j_patches
        prompt = PROMPT_RAG
    elif args.dataset == "human-eval-java":
        dataset = clean_parse_human_eval()
        validate_func = validate_one_human_eval
        prompt = PROMPT_RAG
    else:
        print("Unknown dataset: {}".format(args.dataset))
        return -1

    set_seed(args.seed)

    model = RepairModel(batch_size=args.batch_size, model_name=args.model_name)
    # model = RepairModelOllama(batch_size=args.batch_size, model_name=args.model_name)

    repair(args, model, dataset, args.folder, args.correct_patch_folder, prompt, args.chances, validate_func)


if __name__ == '__main__':
    main()
