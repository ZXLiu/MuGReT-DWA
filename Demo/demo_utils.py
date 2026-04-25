import os
import torch
import json
import logging
import random
import numpy as np
from difflib import unified_diff


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def merge_json(file_path_1, file_path_2, output_file):
    part_1 = read_json(file_path_1)
    part_2 = read_json(file_path_2)

    merge_data = part_1 + part_2
    write_json(output_file, merge_data)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remove_empty_lines(source):
    lines = source.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(non_empty_lines)


def remove_comments(source):  # 适用于C/C++/Java
    try:
        # Define states
        NORMAL, SINGLE_COMMENT, MULTI_COMMENT, STRING_LITERAL, CHAR_LITERAL = range(5)

        state = NORMAL
        result = []
        i = 0

        while i < len(source):
            # Check the current state and process accordingly
            if state == NORMAL:
                if source[i : i + 2] == "//":
                    state = SINGLE_COMMENT
                    i += 2
                elif source[i : i + 2] == "/*":
                    state = MULTI_COMMENT
                    i += 2
                elif source[i] == '"':
                    state = STRING_LITERAL
                    result.append(source[i])
                    i += 1
                elif source[i] == "'":
                    state = CHAR_LITERAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1
            elif state == SINGLE_COMMENT:
                if source[i] == "\n":
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    i += 1
            elif state == MULTI_COMMENT:
                if source[i : i + 2] == "*/":
                    state = NORMAL
                    i += 2
                else:
                    i += 1
            elif state == STRING_LITERAL:
                if source[i] == "\\":
                    result.append(source[i])
                    i += 1
                    result.append(source[i])
                    i += 1
                elif source[i] == '"':
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1
            elif state == CHAR_LITERAL:
                if source[i] == "\\":
                    result.append(source[i])
                    i += 1
                    result.append(source[i])
                    i += 1
                elif source[i] == "'":
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1

        return "".join(result)
    except Exception as e:
        logging.warning(
            f"Failed to remove_java_comments from\n```n{source}\n```\nwith error: {e}"
        )
        return None


def clean_left_space(code):
    lines = code.splitlines()
    leading_white_space = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join([line[leading_white_space:] for line in lines])


def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm=''):
        output += line + "\n"
    return output


def process_data(code):
    code = remove_comments(code)
    code = remove_empty_lines(code)
    code = clean_left_space(code)
    return code


def get_folder_name(floder_path):
    all_items = os.listdir(floder_path)
    print(f"修复个数: {len(all_items)}")
    print(f"修复bug的项目名称: {all_items}")


def get_unique_count(file_path):
    project_count, unique_count, total_count = 0, 0, 0
    lm_repair = read_json(file_path)
    for id, lm_item in lm_repair.items():
        project_count += 1
        for item in lm_item:
            if item['diff'] != "":
                unique_count += 1
                total_count += item['num']
    print(f"total count: {total_count}, unique count: {unique_count}, project count: {project_count}")
