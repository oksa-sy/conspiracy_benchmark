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

## Setup and Testing (Mock Data)

Since the original datasets are too large to be tracked on GitHub, a `sample_data/` folder containing mock data is provided to allow testing the pipeline architecture.

### Option A: Running on the HPC Cluster (SLURM)

To prepare the environment and set up the mock data on the cluster, run the following commands:

```bash
# 1. Clone the repository and setup the environment
git clone https://github.com/oksa-sy/conspiracy_benchmark
cd conspiracy_benchmark
module load python/3.10.8_10gcc
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy the mock data into the working directory
mkdir -p data
cp -r sample_data/* data/
```

The first step is to extract and harmonize data using the dataset loaders. The builder will automatically skip the datasets that are not present in the mock data folder, but it will successfully process the available mock test data.

> **Note:** The bash script `run_build.sh` is pre-configured with a default splitting strategy. If you want to change the strategy (e.g., to LOO or stratified) or the target dataset, you must open the `run_build.sh` and `run_ctbert.sh` files and modify the Python arguments inside them before running `sbatch`.

```bash
sbatch run_build.sh
```

Check the generated `.err` and `.out` logs to verify that the mock records were successfully processed and saved into `data/03_benchmark/benchmark_train.jsonl`.

Finally, launch the CT-BERT training script:

```bash
sbatch run_ctbert.sh
```

### Option B: Alternative Running Locally

If you don't have access to the HPC cluster, you can easily test the pipeline on your local machine using the mock data.

**1. Setup the local environment:**
```bash
git clone https://github.com/oksa-sy/conspiracy_benchmark
cd conspiracy_benchmark
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Copy the mock data:**
```bash
mkdir -p data
cp -r sample_data/* data/
```

**3. Run the pipeline directly:**
Instead of using SLURM, execute the Python entrypoints directly from your terminal. When building the benchmark, you must define the splitting strategy using the `--split-strategy` option. If you use the leave-one-out approach, you must also specify the `--target-dataset`:

```bash
# Example 1: Preserving original splits
python scripts/04_build_benchmark.py --split-strategy preserve_original_splits

# Example 2: Building with Leave-One-Out (LOO) strategy (e.g., leaving out 'semeval')
python scripts/04_build_benchmark.py --split-strategy leave_one_dataset_out --target-dataset semeval

# Example 3: Building with a stratified 80/10/10 re-split
python scripts/04_build_benchmark.py --split-strategy stratified_re_split
```

Finally, to launch the CT-BERT fine-tuning, you must pass the target dataset name (e.g., `semeval`, `voat`, `ccn`) as a positional argument. If you are running a LOO experiment, pass `loo` as the argument and add the `--loo` flag:

```bash
# Standard training on a specific dataset (e.g., semeval)
python src/modeling/transformer.py semeval

# LOO training
python src/modeling/transformer.py loo --loo
```