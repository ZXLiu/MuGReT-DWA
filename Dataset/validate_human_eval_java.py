import os
import json
import time
import shutil
import subprocess


def command_with_timeout(cmd, timeout=60):
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()
            return 'TIMEOUT', 'TIMEOUT'
        time.sleep(1)
    out, err = p.communicate()
    return out, err


def humaneval_test_suite(algo, humaneval_dir):
    CUR_DIR = os.getcwd()
    FNULL = open(os.devnull, 'w')
    try:
        os.chdir(humaneval_dir)
        out, err = command_with_timeout(["mvn", "test", "-Dtest=TEST_" + algo.upper()], timeout=30)
        os.chdir(CUR_DIR)
        msg = (str(out) + str(err)).upper()
        if "compilation problems".upper() in msg or "compilation failure".upper() in msg:
            return 'uncompilable'
        elif "timeout".upper() in msg:
            return 'timeout'
        elif "build success".upper() in msg:
            return 'plausible'
        else:
            return "wrong"
    except Exception as e:
        print(e)
        os.chdir(CUR_DIR)
        return 'uncompilable'
    

def insert_fix(filename, start_line, end_line, patch):
    with open(filename, 'r') as file:
        data = file.readlines()

    with open(filename, 'w') as file:
        for i in range(start_line - 1):
            file.write(data[i] + '\n')
        file.write(patch.strip())
        for i in range(end_line, len(data)):
            file.write(data[i])


def validate_one_human_eval(buggy_file,patch_file,folder,patch):
    tmp_dir = "/home/liutongxue/human-eval-java/"
    
    with open("/home/liutongxue/PycharmProject/bishe/HumanEval/humaneval-java-sf.json", "r", encoding="utf-8") as f:
        bug_dict = json.load(f)
    
    command_with_timeout(['rm', '-rf', tmp_dir + 'src/main/java/humaneval/buggy/'])
    command_with_timeout(['mkdir', '-p', tmp_dir + 'src/main/java/humaneval/buggy/'])
    command_with_timeout(['rm', '-rf', tmp_dir + 'src/test/java/humaneval/'])
    command_with_timeout(['mkdir', '-p', tmp_dir + 'src/test/java/humaneval/'])

    try:
        with open(folder + "/" + patch_file, 'w') as f:
            f.write(patch)
    except Exception as e:
        print(f"Error: {e}")

    print("Validating patch ... ")

    command_with_timeout(['rm', '-rf', tmp_dir + 'src/main/java/humaneval/buggy/*.java'])
    command_with_timeout(['rm', '-rf', tmp_dir + 'src/test/java/humaneval/*.java'])
    shutil.copyfile(tmp_dir + 'src_bak/main/java/humaneval/buggy/' + buggy_file + '.java', tmp_dir + 'src/main/java/humaneval/buggy/' + buggy_file + '.java')
    shutil.copyfile(tmp_dir + 'src_bak/test/java/humaneval/TEST_' + buggy_file + '.java', tmp_dir + 'src/test/java/humaneval/TEST_' + buggy_file + '.java')
    
    filename = tmp_dir + 'src/main/java/humaneval/buggy/' + buggy_file + '.java'
    start_line, end_line = bug_dict[buggy_file]['begin'], bug_dict[buggy_file]['end']
    insert_fix(filename, start_line, end_line, patch)

    
    correctness = humaneval_test_suite(buggy_file, tmp_dir)
    if correctness == 'plausible':
        print("Patch is valid")
        return 'valid'
    else:
        print("Patch is invalid")
        return correctness
