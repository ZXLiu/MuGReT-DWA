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
