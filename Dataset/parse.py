import json
from utils import *

d4j_bug_lists = '''
| Chart           | jfreechart                 |       26       | 1-26                | None                    |
| Cli             | commons-cli                |       39       | 1-5,7-40            | 6                       |
| Closure         | closure-compiler           |      174       | 1-62,64-92,94-176   | 63,93                   |
| Codec           | commons-codec              |       18       | 1-18                | None                    |
| Collections     | commons-collections        |        4       | 25-28               | 1-24                    |
| Compress        | commons-compress           |       47       | 1-47                | None                    |
| Csv             | commons-csv                |       16       | 1-16                | None                    |
| Gson            | gson                       |       18       | 1-18                | None                    |
| JacksonCore     | jackson-core               |       26       | 1-26                | None                    |
| JacksonDatabind | jackson-databind           |      112       | 1-112               | None                    |
| JacksonXml      | jackson-dataformat-xml     |        6       | 1-6                 | None                    |
| Jsoup           | jsoup                      |       93       | 1-93                | None                    |
| JxPath          | commons-jxpath             |       22       | 1-22                | None                    |
| Lang            | commons-lang               |       64       | 1,3-65              | 2                       |
| Math            | commons-math               |      106       | 1-106               | None                    |
| Mockito         | mockito                    |       38       | 1-38                | None                    |
| Time            | joda-time                  |       26       | 1-20,22-27          | 21                      |'''


def clean_parse_d4j():
    with open("/home/liutongxue/PycharmProject/bishe/filter_retrieval_dataset/top_1_defects4j.json", "r", encoding='utf-8') as f:
        result = json.load(f)
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        v['buggy'] = "\n".join([line[leading_white_space:] for line in lines])
        v['buggy'] = remove_empty_lines(remove_comments(v['buggy']))
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        v['fix'] = "\n".join([line[leading_white_space:] for line in lines])
        v['fix'] = remove_empty_lines(remove_comments(v['fix']))
    return result


def clean_parse_human_eval():
    with open("/home/liutongxue/PycharmProject/bishe/filter_retrieval_dataset/top_1_human_eval.json", "r", encoding='utf-8') as f:
        result = json.load(f)
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        v['buggy'] = "\n".join([line[leading_white_space:] for line in lines])
        v['buggy'] = remove_empty_lines(remove_comments(v['buggy']))
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        v['fix'] = "\n".join([line[leading_white_space:] for line in lines])
        v['fix'] = remove_empty_lines(remove_comments(v['fix']))
    return result
