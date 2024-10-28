
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

### Synthetic Data

For synthetic datasets, use the following command. Replace `syn1` with your specific synthetic dataset and `str_60` with the desired OOD configuration:

```shell
python pipeline_syn.py --dataset=syn1 --ood=str_60
```

### Real Data

For real datasets, you can run:

```shell
python pipeline_real.py --dataset=Cora
```

## Explanation Modules

ORExplainer leverages OOD detection techniques to analyze model robustness in different graph settings. The model works with both synthetic and real-world graph data to provide insights into feature importance and node interactions.

- **Synthetic Data**: Use `pipeline_syn.py` for controlled experiments with synthetic datasets.
- **Real Data**: Use `pipeline_real.py` to test on real-world datasets like Cora.

## References

This project is based on methodologies from:
1. **DIG Library** - A [Deep Graph Library](https://github.com/divelab/DIG) that provides a framework for graph learning.
2. **RE-Parameterized Explainer** - From [LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks), an approach for graph neural network explanations.
