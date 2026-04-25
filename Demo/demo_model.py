import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import re
import sys
from ollama import Client
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams


global_eof_stops = ['// Buggy Function', '// Fixed Function', '# Buggy Function', '# Fixed Function', '/* Buggy Function */', '/* Fixed Function */', '<|endoftext|>', '</s>', '// Provide a fix for the buggy function']


def extract_markdown_code(text):
    if not text:
        return ""

    pattern = r"```[^\n]*\n([\s\S]*?)```"
    
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[-1].strip()
    
    return text.strip()

# 模型路径检查
def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError("路径不存在: {}".format(path))


class RepairModel(object):
    def __init__(self, model_name, batch_size):
        print("Initializing a generate model: {} ...".format(model_name))
        self.model_path = '/home/liutongxue/LLM_Model/Base_Model/' + model_name
        try:
            check_path_exists(self.model_path)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        
        self.model_name = model_name
        self.batch_size = batch_size
        config = AutoConfig.from_pretrained(self.model_path)
        self.max_length = 1024  # default context size of 1024
        if hasattr(config, 'max_position_embeddings'):
            self.max_length = config.max_position_embeddings
        elif hasattr(config, 'n_positions'):
            self.max_length = config.n_positions
        print("Max length: {}".format(self.max_length))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.llm = LLM(model=self.model_path, gpu_memory_utilization=0.9, max_num_seqs=256)
        self.llm = LLM(model=self.model_path, tensor_parallel_size=4, gpu_memory_utilization=0.9, max_num_seqs=256)

    def check_input(self, prompt):
        input_tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        if len(input_tokens[0]) >= self.max_length:
            return False
        return True

    def model_predict(self, prompt, buggy_func, num_samples):
        if not self.check_input(prompt):
            return False, []  # If the input is too long, return False

        input_tokens = self.tokenizer.encode(prompt, return_tensors=None, add_special_tokens=True)
        buggy_tokens = self.tokenizer.encode(buggy_func, return_tensors=None, add_special_tokens=True)
        max_length = min(self.max_length - int(len(input_tokens)), int(2 * len(buggy_tokens)))

        sampling_params = SamplingParams(n=self.batch_size, temperature=0.8, top_p=0.95, max_tokens=max_length, stop=global_eof_stops)

        all_generate_texts = []
        n_times = num_samples // self.batch_size
        for i in range(n_times):
            outputs = self.llm.generate([prompt], sampling_params)
            for output in outputs:
                for candidate in output.outputs:
                    all_generate_texts.append(candidate.text)
        all_generate_texts = [generate_text.strip() for generate_text in all_generate_texts]
        return True, all_generate_texts
    

class RepairModelNew(object):
    def __init__(self, model_name, batch_size):
        print("Initializing a generate model: {} ...".format(model_name))
        self.model_path = '/home/liutongxue/LLM_Model/Base_Model/' + model_name
        try:
            check_path_exists(self.model_path)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        
        self.model_name = model_name
        self.batch_size = batch_size
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.max_length = 1024  # default context size of 1024
        
        if hasattr(config, 'max_position_embeddings'):
            self.max_length = config.max_position_embeddings
        elif hasattr(config, 'n_positions'):
            self.max_length = config.n_positions
        print("Max length: {}".format(self.max_length))

        # --- 新增：判断是否为 gpt-oss 模型 ---
        self.is_gpt_oss = "gpt-oss" in model_name.lower()

        if self.is_gpt_oss:
            # 使用 Harmony 编码器替代 HuggingFace Tokenizer
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.tokenizer = None 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # vLLM 引擎初始化 (trust_remote_code=True 对 gpt-oss 很重要)
        self.llm = LLM(
            model=self.model_path, 
            tensor_parallel_size=4, 
            gpu_memory_utilization=0.9, 
            max_num_seqs=256,
            trust_remote_code=True
        )

    # --- 新增：专门用于 gpt-oss 构建对话的辅助函数 ---
    def _build_harmony_convo(self, user_prompt):
        return Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("Output only the repaired code function, do not output anything else.")
            ),
            Message.from_role_and_content(Role.USER, user_prompt),
        ])

    def check_input(self, prompt):
        if self.is_gpt_oss:
            # 使用 Harmony 计算 token 长度
            convo = self._build_harmony_convo(prompt)
            input_tokens = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            if len(input_tokens) >= self.max_length:
                return False
            return True
        else:
            # 原有的 HuggingFace 逻辑
            input_tokens = self.tokenizer.encode(prompt, return_tensors='pt')
            if len(input_tokens[0]) >= self.max_length:
                return False
            return True

    def model_predict(self, prompt, buggy_func, num_samples):
        if not self.check_input(prompt):
            return False, []  # If the input is too long, return False

        if self.is_gpt_oss:
            # 1. 编码 Prompt
            convo = self._build_harmony_convo(prompt)
            prefill_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            input_length = len(prefill_ids)
            
            # 2. 估算 buggy_func 的 token 长度 (用于限制最大生成长度)
            temp_convo = self._build_harmony_convo(buggy_func)
            buggy_tokens_len = len(self.encoding.render_conversation_for_completion(temp_convo, Role.ASSISTANT))
            
            max_new_tokens = min(self.max_length - input_length, int(10 * buggy_tokens_len))
            stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

            sampling_params = SamplingParams(
                n=self.batch_size, 
                temperature=0.8, 
                top_p=0.95, 
                max_tokens=max_new_tokens, 
                # stop=global_eof_stops,
                stop_token_ids=stop_token_ids
            )
        else:
            # 原始逻辑
            input_tokens = self.tokenizer.encode(prompt, return_tensors=None, add_special_tokens=True)
            buggy_tokens = self.tokenizer.encode(buggy_func, return_tensors=None, add_special_tokens=True)
            max_new_tokens = min(self.max_length - int(len(input_tokens)), int(2 * len(buggy_tokens)))

            sampling_params = SamplingParams(
                n=self.batch_size, 
                temperature=0.8, 
                top_p=0.95, 
                max_tokens=max_new_tokens, 
                stop=global_eof_stops # 使用字符串作为停止条件
            )

        all_generate_texts = []
        n_times = num_samples // self.batch_size
        
        for i in range(n_times):
            # --- 区分不同模型的输入格式 ---
            if self.is_gpt_oss:
                outputs = self.llm.generate(prompts=[{"prompt_token_ids": prefill_ids}], sampling_params=sampling_params)
            else:
                outputs = self.llm.generate([prompt], sampling_params)

            for output in outputs:
                for candidate in output.outputs:
                    if self.is_gpt_oss:
                        # --- 提取 gpt-oss 的 Token IDs 并反解析 ---
                        entries = self.encoding.parse_messages_from_completion_tokens(candidate.token_ids, Role.ASSISTANT)
                        print(type(entries), len(entries))
                        for message in entries:
                            msg_dict = message.to_dict()
                            if msg_dict["channel"] == 'final':
                                generate_text = extract_markdown_code(msg_dict["content"][0]["text"])
                                all_generate_texts.append(generate_text)
                    else:
                        # 原有模型的纯文本提取
                        all_generate_texts.append(candidate.text)
                        
        all_generate_texts = [generate_text.strip() for generate_text in all_generate_texts]
        return True, all_generate_texts


class RepairModelOllama(object):
    def __init__(self, model_name, batch_size, host='http://localhost:11434'):
        print(f"Initializing Ollama client for model: {model_name} ...")
        self.model_path = '/home/liutongxue/LLM_Model/Base_Model/' + model_name
        try:
            check_path_exists(self.model_path)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        self.model_name = 'gpt-oss:20b' if model_name == 'GPT-OSS-20B' else model_name
        self.batch_size = batch_size
        self.client = Client(host=host)

        config = AutoConfig.from_pretrained(self.model_path)
        self.max_length = 1024  # default context size of 1024
        if hasattr(config, 'max_position_embeddings'):
            self.max_length = config.max_position_embeddings
        elif hasattr(config, 'n_positions'):
            self.max_length = config.n_positions
        print(f"Client-side Max length check: {self.max_length}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        

    def check_input(self, prompt):
        input_tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        if len(input_tokens[0]) >= self.max_length:
            return False
        return True

    def model_predict(self, prompt, buggy_func, num_samples):
        if not self.check_input(prompt):
            return False, []

        # 估算输出长度限制
        input_tokens = self.tokenizer.encode(prompt, return_tensors=None, add_special_tokens=True)
        buggy_tokens = self.tokenizer.encode(buggy_func, return_tensors=None, add_special_tokens=True)
        max_new_tokens = min(self.max_length - int(len(input_tokens)), int(10 * len(buggy_tokens)))

        # 1. 构造 Ollama 接受的 Messages 格式
        # 注意：Ollama 没有 Developer 角色，我们将强硬的指令合并到 System 角色中
        messages = [
            {'role': 'system', 'content': "Output only the repaired code function, do not output anything else."},
            {'role': 'user', 'content': prompt}
        ]

        # 2. 设置 Ollama 的生成参数
        options = {
            'temperature': 0.8,
            'top_p': 0.95,
            'num_predict': max_new_tokens,  # Ollama 中控制最大生成长度的参数
            'stop': global_eof_stops        # 传入你的字符串停止词
        }

        all_generate_texts = []
        
        # 3. 执行推理
        # 注意：Ollama 的 API 默认不支持像 vLLM 那样一次性传入 N 的参数来并发生成多个候选。
        # 如果需要生成多个样本 (num_samples)，通常需要循环调用。
        for i in range(num_samples):
            try:
                response = self.client.chat(
                    model=self.model_name, # 例如 'gpt-oss:20b'
                    messages=messages,
                    options=options
                )
                
                # Ollama 返回的是干净的文本，无需再像 Harmony 那样反解析
                text_content = response['message']['content'].strip()
                text_content = extract_markdown_code(text_content)
                all_generate_texts.append(text_content)
                
            except Exception as e:
                print(f"Ollama generation error: {e}")
                
        return True, all_generate_texts
