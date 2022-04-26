# EmProx
Implementation of paper EmProx: Neural Network Performance Estimation For Neural Architecture Search

Adaption of [NASLib](https://github.com/automl/NASLib).

# Setup
1. It is recommended to create a new conda environment.

```bash
conda create -n mvenv python=3.8
```

2. Install the packages from [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```

3. Download the CIFAR-10 dataset [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and the file `nb201_cifar10_full_training.pickle` [here](https://drive.google.com/file/d/1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ/view) and place them in the directory `EmProx/naslib/data` (optional: [CIFAR100 data for NAS-Bench-201](https://drive.google.com/file/d/1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp/view?usp=sharing), [DARTS architectures](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510) and [CIFAR10 data for DARTS](https://drive.google.com/file/d/1YJ80Twt9g8Gaf8mMgzK-f5hWaVFPlECF/view?usp=sharing)).

# Usage
To reproduce the findings in the paper, simply run the following command in the directory `EmProx/experiments`:

```bash
python run_experiments.py
```

This evaluates the predictors listed below on architectures from NAS-Bench-201 on the CIFAR-10 dataset. Evaluation metrics are MAE and RMSE between the predicted and 
actual validation accuracy of the architectures. Additionally, since in many NAS algorithms the exact predicted performance is not of importance, rather than the rank 
of a certain architecture among other candidates, correlation-based performance measures Pearson, Spearman and Kendall’s Tau on the predicted and validation accuracies 
are included. Lastly, fit (training) times and query times are incorporated.

Results are averaged over 20 trials and outputted in seperate Excel files per predictor. 

* EmProx (with k=60)
* [SemiNAS](https://arxiv.org/abs/2002.10389)
* [BANANAS](https://arxiv.org/abs/1910.11858)
* [XGBoost](https://arxiv.org/abs/1603.02754)
* [NAO](https://arxiv.org/abs/1808.07233)
* [SoTL-E](https://arxiv.org/abs/2006.04492v1)
* MLP

To reproduce the additional experiments, the search space and dataset can be changed in the `experiment_config.yaml` file. 

# Results
Based on our own experiments, results are as follows:

| Predictor       | MAE        | RMSE        | Pearson    | Spearman   | Kendall    | Fit time    | Query time  |
|-----------------|------------|-------------|------------|------------|------------|-------------|-------------|
| EmProx (k = 10) | **4.4027** | **10.7163** | 0.4771     | 0.7304     | 0.5453     | **6.4498**  | **0.0009**  |
| EmProx (k = 60) | 4.5264     | 10.7953     | **0.5044** | **0.7332** | **0.5468** | 7.2310      | 0.0032      |
| NAO             | 4.7336     | 10.9394     | 0.4512     | 0.6433     | 0.4835     | 54.1674     | 0.0026      |
| SemiNAS         | **4.0283** | **10.1222** | **0.5307** | **0.7677** | **0.5822** | 152.2268    | 0.0012      |
| XGB             | 5.3989     | 12.2955     | 0.4008     | 0.6466     | 0.4719     | **31.6526** | 0.0004      |
| BANANAS         | 7.3007     | 12.2760     | 0.3793     | 0.4169     | 0.2910     | 507.3936    | 0.0002      |
| MLP             | 6.8584     | 11.3592     | 0.4417     | 0.5298     | 0.3759     | 471.7508    | **<0.0001** |

The upper two rows are the results of the method proposed in this study, below are several methods evaluated in White et al. (2021). In bold are the best
results of EmProx and the best results among the other predictors for ease of comparison. As can be seen, EmProx scores competitively regarding regression 
and correlation coefficients and scores much better regarding fit time.


