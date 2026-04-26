# MuGReT-DWA
Here is the open-source code repository for the paper "Bridging Structural and Semantic Similarity for Code RAG-Driven Program Repair "

It is structured as follows:
- [DWA](DWA) contains the specific implementation of the DWA algorithm, including training data construction, model network, and model training.
- [Dataset](Dataset) contains the processing logic for two benchmarks as well as the patch validation logic on these two benchmarks.
- [Defects4j](Defects4j) contains the single-function bugs we extracted from Defects4J 2.0.
- [Demo](Demo) contains the examples from the case study section of the paper, along with their specific repair situations.
- [HumanEval](HumanEval) contains the single-function bugs we extracted from HumanEval-Java.
- [MuGReT](MuGReT) contains the specific implementation of the MuGReT algorithm, including code property graph construction, code abstraction implementation, MuGReT network, and model training.
- [Retrieval](Retrieval) contains similarity retrieval using C2LLM, MuGReT, and DWA.
- [model.py](model.py) contains the initialization and generation settings of the foundation models selected in the paper (e.g., Qwen2.5-Coder-7B).
- [prompt.py](prompt.py) contains prompt settings for both with and without retrieval.
- [reapir.py](repair.py) contains the entire repair process and serves as the main program entry point.
- [utils.py](utils.py) contains common utility functions, such as reading and writing JSON files.

## Guide
### 1.Preparation
To run this code, you first need to install the two benchmarks and the C2LLM code sequence retrieval model used in the paper.
#### Benchmarks
- Defects4J: [https://github.com/rjust/defects4j](https://github.com/rjust/defects4j)<br>
- HumanEval-Java: [https://github.com/ASSERT-KTH/human-eval-java](https://github.com/ASSERT-KTH/human-eval-java)
#### Retrieval Model
- C2LLM: [https://huggingface.co/codefuse-ai/C2LLM-0.5B](https://huggingface.co/codefuse-ai/C2LLM-0.5B)
### 2.MuGReT Algorithm Implementation
Execute the following commands to train MuGReT. All training configurations and model parameters are located in 'MuGReT-DWA/MuGReT/configs/bcb.json':
```bash
cd MuGReT
python3 main.py --config ./configs/bcb.json
```
### 3.DWA Algorithm Implementation
Execute the following commands to train DWA.
```bash
cd DWA
python3 main.py
```
### 4.Retrieval Implementation
After training the MuGReT and DWA models, you can use them for retrieval. Following the settings described in the paper, retrieval is categorized into three types: code sequence retrieval (C2LLM), code graph retrieval (MuGReT), and hybrid retrieval (DWA). The corresponding operations are as follows:
```bash
cd Retrieval
# C2LLM Retrieval
python3 C2LLM.py
# MuGReT and DWA Retrieval
python3 MuGReT_And_DWA.py
```
### 5.Repair Implementation
After obtaining the retrieval results, run the following command to execute the final repair workflow.
```bash
python3 repair.py --model_name generative model name --batch_size batch_size --dataset defects4j or humanevaljava --chances beam search count --folder directory for storing model-generated patches --correct_patch_folder directory for storing plausible patches that pass test cases
```
## Demo
We provide a demo using two examples from the paper's Case Study: 'FRUIT_DISTRIBUTION' and 'Collections-26.' The following commands allow you to directly compare the LLM's repair results with and without retrieval.
```bash
cd Demo
# Without Retrieval
python3 demo_repair.py --use_retrieval false
# With Retrieval
python3 demo_repair.py --use_retrieval true
```
