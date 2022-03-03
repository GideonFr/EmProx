# EmProx
Implementation of paper EmProx: Neural Network Performance Estimation For Neural Architecture Search

Adaption of [NASLib](https://github.com/automl/NASLib).

# Setup
Firstly, it is recommended to create a new conda environment.

```bash
conda create -n mvenv python=3.8
```

Secondly, install the packages from [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```

Thirdly, download the CIFAR-10 dataset [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and place it in the directory `EmProx/naslib/data`.

# Usage
To reproduce the findings in the paper, simply run the following command in the directory `EmProx/experiments`:

```bash
python run_experiments.py
```