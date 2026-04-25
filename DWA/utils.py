import json


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
