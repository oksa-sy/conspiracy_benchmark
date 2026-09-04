# Conspiracy Benchmark

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Analysis and evaluation of artificial intelligence models for the detection of conspiracy theories.

## Project Organization

```text
conspiracy-benchmark/
├── config/
│   ├── canonical_taxonomy.yaml  # central source of truth for target labels
│   └── platform_credentials.env # template for API keys (e.g., Twitter/X, Reddit)
|
├── data/                        # ignored by git (.gitignore)
│   ├── 01_raw/                  # raw files (downloaded, hydrated, or placed manually)
│   ├── 02_standardized/         # intermediate harmonized JSONL files (original splits preserved)
│   └── 03_benchmark/            # final combined benchmark splits (train/val/test)
|
├── models/                      # trained models
|
├── reports/                     
│   ├── metrics/                 # evaluation metrics saved after training
│   └── predictions/             # predictions generated on the test sets
│
├── sample_data/                 # mock data to allow testers to test the pipeline
│
├── src/
│   ├── dataset_modules/         # self-contained dataset loaders + mappings
│   │   ├── __init__.py          # dynamic registry decorator
│   │   ├── base_loader.py       # abstract base class (strategy pattern)
│   │   ├── ccn_loader.py        
│   │   ├── coco_loader.py       
│   │   ├── donald_loader.py     
│   │   ├── loco_loader.py       
│   │   ├── semeval_loader.py    
│   │   └── voat_loader.py
|   ├── evaluation/              
│   │   └── evaluate.py          # model performance calculation module
|   ├── modeling/                
│   │   ├── baseline.py          # baselines script
│   │   └── transformer.py       # CT-BERT fine-tuning script
│   ├── harmonizer.py            # validates loader outputs against canonical taxonomy
│   └── builder.py               # merges harmonized splits into benchmark datasets
|
├── scripts/
│   ├── 01_download_public.py    # entrypoint to fetch public datasets
│   ├── 02_rehydrate_ids.py      # entrypoint to run API hydration
│   ├── 03_process_private.py    # entrypoint to validate manually placed private files
│   └── 04_build_benchmark.py    # entrypoint to unify and export harmonized dataset
|
├── run_build.sh                 # bash/SLURM script to start building the benchmark on the HPC system
├── run_ctbert.sh                # bash/SLURM script to start Transformer training on HPC
├── pyproject.toml               # package metadata and tool configurations
├── requirements.txt
└── README.md
```

## Prerequisites

Before starting, ensure your system meets the following requirements:
* **Supported OS:** macOS, Linux, or Windows (via WSL - Windows Subsystem for Linux).
* **Python:** Python 3.8 or higher.
* **Virtual Environment:** The `venv` module must be installed. If you are using Linux or WSL and it is missing, you can install it by running `sudo apt install python3-venv`.

## Setup and Testing (Mock Data)

Since the original datasets are too large to be tracked on GitHub, a `sample_data/` folder containing mock data is provided to allow testing the pipeline architecture.

### Step 1: Environment Setup
First, clone the repository, prepare your virtual environment, and load the mock data into your working directory:

```bash
git clone https://github.com/oksa-sy/conspiracy_benchmark
cd conspiracy_benchmark

# If you are on the HPC cluster, load the Python module first:
# module load python/3.10.8_10gcc

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy the mock data
mkdir -p data
cp -r sample_data/* data/
```

### Step 2: Parameters & Split Strategy (Info)
Once records are validated and serialized into intermediate standardized JSONL files (`data/02_standardized/`), researchers could choose between different combination strategies to build the final benchmark:
* **Preserve Original Splits:** `--split-strategy preserve_original_splits`
* **Leave One Dataset Out:** `--split-strategy leave_one_dataset_out --target-dataset <name>`
* **Stratified Re-split:** `--split-strategy stratified_re_split`

### Step 3: Execute the Pipeline
Depending on your environment, choose the appropriate command to build the benchmark.

**Local Execution (macOS / Linux / WSL):**
Execute the Python entrypoints directly from your terminal, passing your chosen strategy:

```bash
# Example 1: Preserving original splits
python scripts/04_build_benchmark.py --split-strategy preserve_original_splits

# Example 2: Building with Leave-One-Out (LOO) strategy
python scripts/04_build_benchmark.py --split-strategy leave_one_dataset_out --target-dataset semeval
```

**HPC Cluster Execution (Slurm):**
If you are running the pipeline on the cluster, use the `sbatch` command. 
> **Note:** The bash script `run_build.sh` is pre-configured with a default splitting strategy. If you want to change it, open the `run_build.sh` file and modify the Python arguments inside before submitting the job.

```bash
sbatch run_build.sh
```

### Step 4: Model Training (CT-BERT)
Once the datasets are built, you can launch the transformer fine-tuning.

**Local Execution:**
Pass the target dataset name (e.g., `semeval`, `voat`, `ccn`) as a positional argument. If you are running a LOO experiment, pass `loo` as the argument and add the `--loo` flag.
```bash
# Standard training
python src/modeling/transformer.py semeval

# LOO training
python src/modeling/transformer.py loo --loo
```

**HPC Cluster Execution (Slurm):**
Modify the dataset arguments directly inside `run_ctbert.sh` if necessary, then launch the training job:
```bash
sbatch run_ctbert.sh
```