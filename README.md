
# ORExplainer

## Requirements

To ensure compatibility, we recommend using the following versions:

- **python**==3.10.10
- **spacy**==3.5.2
- **torch**==1.13.1
- **pytorch-geometric**==2.2.0

Additional dependencies and exact package versions can be found in the `environment.yml` file.

### Setting Up the Environment

To create the environment with all dependencies:

```shell
conda env create -f environment.yml
```

## Usage

The main scripts for running the model on synthetic and real datasets are provided below:

## Running the Pipelines
### Synthetic Datasets
```bash
python pipeline_syn.py --dataset=syn1 --ood=1 --date=0901
```
- Replace `syn1` with your target synthetic dataset.  
- The `--ood` flag specifies the OOD **structural shift level** (`0–3`).  

### Real-World Datasets
```bash
python pipeline_real.py --dataset=Cora --ood=2        # Featural OOD (level 2)
python pipeline_real.py --dataset=Citeseer --ood=label1  # Unseen label OOD
```
- **Structural / Featural OOD**: set `--ood` to an integer between `0–3` (higher = stronger OOD level).  
- **Unseen Label OOD**: set `--ood` to `label0` (disabled) or `label1` (enabled).  

#### Recommended Practice
For better traceability of results, we recommend appending a date-based identifier to your save directory or experiment name.  

## References

This project is based on methodologies from:
1. **DIG Library** - A [Deep Graph Library](https://github.com/divelab/DIG) that provides a framework for graph learning.
2. **RE-Parameterized Explainer** - From [LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks), an approach for graph neural network explanations.
