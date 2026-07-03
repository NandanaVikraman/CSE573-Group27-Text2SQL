# Traditional Text-to-SQL Baseline (DyNet)

This repository contains a **neural sequence-to-sequence (pre-LLM) Text-to-SQL baseline** for the Spider task, implemented with DyNet.    
The model is a **BiLSTM encoder-decoder with attention and a pointer-generator (copy) mechanism**.

## What is included

- BiLSTM encoder
- Attention-based decoder
- Pointer-generator copy mechanism (extended vocabulary)
- Greedy decoding and beam search
- Training and evaluation scripts
- Preprocessing and vocabulary-building scripts
- Processed Spider train/dev files
- Checkpoint saving and resume support
- SLURM job script for SOL cluster

## Requirements

This project was developed and tested with:

- Python 3.8
- NumPy
- NLTK
- Cython < 3
- DyNet 2.1.2 (CUDA-enabled build from source)
- NVIDIA CUDA Toolkit
- Eigen headers
- GPU-enabled machine (recommended)

### System requirements

To run the CUDA version of DyNet, you need:

- NVIDIA GPU
- NVIDIA driver installed
- CUDA toolkit available on the system
- Compatible compiler toolchain
- Eigen headers available locally
- Sufficient disk space for DyNet build


### Recommended environment setup

```bash
conda create -n dynet38 python=3.8 -y
conda activate dynet38
pip install -r requirements.txt
pip install "Cython<3"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## CUDA / DyNet setup

This project uses a CUDA-enabled DyNet build. On our cluster, the working setup was:

```bash
module load cuda-13.0.1-gcc-12.1.0
```

Then build DyNet from source:

```bash
cd ~
git clone https://github.com/clab/dynet.git dynet-cuda
cd dynet-cuda
git clone https://gitlab.com/libeigen/eigen.git ~/eigen

export EIGEN3_INCLUDE_DIR=$HOME/eigen
export CPLUS_INCLUDE_PATH=$HOME/eigen:$CPLUS_INCLUDE_PATH
export CPATH=$HOME/eigen:$CPATH

rm -rf build
mkdir build
cd build
cmake .. -DBACKEND=cuda -DCUDA_ARCH=80 -DEIGEN3_INCLUDE_DIR=$HOME/eigen -DCMAKE_CXX_STANDARD=14 -DCMAKE_CUDA_STANDARD=14
make -j4

cd ../python
cd ..
python setup.py build_ext --inplace
```

After building, make sure Python can find the local DyNet package:

```bash

cat > python/dynet.py <<'EOF'
from _dynet import *
EOF

export PYTHONPATH=$HOME/dynet-cuda:$HOME/dynet-cuda/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/dynet-cuda/build/dynet:$LD_LIBRARY_PATH
```

Verify installation

From the project directory:

```bash
python -c "import dynet as dy; print(dy.Model)"
```

If it prints a DyNet class reference, the installation is working.

## Data

The processed Spider data should be present in:

- processed/spider_train_processed.jsonl
- processed/spider_dev_processed.jsonl
- processed/src_vocab.json
- processed/tgt_vocab.json

If you need to regenerate these files, run the preprocessing script first.

## Model Overview

- **Encoder:** BiLSTM over input (question + schema tokens)

- **Decoder:** LSTM with attention

- **Copy Mechanism:** Pointer-generator  
  - Enables copying schema tokens (tables, columns) directly from input  
  - Uses extended vocabulary per example  

- **Decoding:**  
  - Greedy decoding  
  - Beam search (optional)

## Training

SLURM submission

Submit the full pipeline with:

```bash
sbatch full_pipeline_job.sh
```

The job script handles:

CUDA module loading
DyNet environment setup
Training
Local evaluation(Currently does not use Spider Official EM and EX)
Prediction export
Resume from checkpoints/model_latest.dy if present

## Evaluation

This implementation uses **local evaluation metrics**:

Metrics

- Token Exact Match (EM): Strict token-level match (used mainly for debugging)
- Normalized Exact Match (primary metric): Lowercased, Whitespace-normalized, Used for checkpoint selection
- Syntax Validity: SQL is executable in SQLite (checks both syntax and schema validity)
- Execution Accuracy (EX): Compares query results against ground truth

## Results

Evaluation on the Spider development set:

Metric Score
- Normalized Exact Match	4.74%
- Token Exact Match	4.74%
- Syntax Validity	30.08%
- Execution Accuracy	7.93%

Observations

- The model frequently predicts incorrect tables (e.g., stadium instead of singer).
- Majority of failures are due to: Incorrect schema linking, Missing or invalid columns.
- Large number of errors are of type: no such column.
- Performance is low due to: Lack of explicit schema understanding, No constrained decoding during generation, Difficulty generalizing to unseen databases (Spider setting)

## Outputs

After running the pipeline:

- train.log → training logs
- eval.log → evaluation results
- pred.txt → predicted SQL queries
- gold.txt → gold SQL + db_id
- results.jsonl → detailed evaluation results (optional)

Checkpoints:
- checkpoints/model_best.dy → best model (based on normalized EM)
- checkpoints/model_latest.dy → latest checkpoint
- checkpoints/model_final.dy → final model

## Summary

This baseline demonstrates the limitations of traditional neural seq2seq models for Text-to-SQL:

- Limited schema understanding
- Poor generalization to unseen databases
- Frequent schema-related errors
- Low exact match and execution accuracy

It serves as a reference baseline for comparison with more advanced approaches.
