# datapruebas_analysis

Neuropsychological data analysis pipeline for processing cognitive task data from multiple experiments.

## Requirements

- Python 3.10.x
- [neurotask](https://github.com/NeuroLIAA/neurotask) (local installation required)

## Installation

```bash
# 1. Clone this repository
git clone <repo-url>
cd datapruebas_analysis

# 2. Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install neurotask (required for TMT analysis)
#    Clone the repository if you don't have it:
git clone https://github.com/NeuroLIAA/neurotask.git ~/Research/neurotask
#    Install in editable mode from the neurotask subdirectory:
cd ~/Research/neurotask/neurotask && pip install -e .

# 5. (Optional) Install development dependencies
pip install -r requirements-dev.txt
```

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run all analyses
python -m src.runner.run_all_analysis

# Run ML pipeline
python -m src.model.run_models --task regression
```

See [CLAUDE.md](CLAUDE.md) for detailed documentation.
