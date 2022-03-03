import logging
import sys
import os
from fvcore.common.config import CfgNode
import pandas as pd
import numpy as np
#import naslib as nl

from predictor_evaluator import PredictorEvaluator

from naslib.predictors import (
    # BayesianLinearRegression,
    # BOHAMIANN,
    # BonasPredictor,
    # DNGOPredictor,
    # EarlyStopping,
    EmProxPredictor,
    Ensemble,
    # GCNPredictor,
    # GPPredictor,
    # LCEPredictor,
    # LCEMPredictor,
    # LGBoost,
    MLPPredictor,
    # NGBoost,
    # OmniNGBPredictor,
    # OmniSemiNASPredictor,
    # RandomForestPredictor,
    # SVR_Estimator,
    SemiNASPredictor,
    # SoLosspredictor,
    # SparseGPPredictor,
    # VarSparseGPPredictor,
    XGBoost,
    # ZeroCostV1,
    # ZeroCostV2,
    # GPWLPredictor,
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    DartsSearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpace
)

from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import get_project_root


#with open(os.path.join(get_project_root(), "experiments", "experiment_config.yaml")) as f:
with open("experiment_config.yaml") as f:
    config = CfgNode.load_cfg(f)

utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

predictor_list = ["emprox", "nao", "seminas", "xgb", "bananas", "mlp"]
for pred in predictor_list:
    print(f'Predictor = {pred}')
    supported_predictors = {
        "bananas": Ensemble(predictor_type="bananas", num_ensemble=3, hpo_wrapper=True),
        "emprox": EmProxPredictor(encoding_type="seminas", semi=True, hpo_wrapper=False, k_nn=60, hidden_lay=32), #k, hidden_lay
        "mlp": MLPPredictor(encoding_type="adjacency_one_hot", hpo_wrapper=True),
        "nao": SemiNASPredictor(encoding_type="seminas", semi=False, hpo_wrapper=False),
        "seminas": SemiNASPredictor(encoding_type="seminas", semi=True, hpo_wrapper=False),
        "xgb": XGBoost(encoding_type="adjacency_one_hot", hpo_wrapper=False),
        # "bayes_lin_reg": BayesianLinearRegression(encoding_type="adjacency_one_hot"),
        # "bohamiann": BOHAMIANN(encoding_type="adjacency_one_hot"),
        # "bonas": BonasPredictor(encoding_type="bonas", hpo_wrapper=True),
        # "dngo": DNGOPredictor(encoding_type="adjacency_one_hot"),
        # "fisher": ZeroCostV2(config, batch_size=64, method_type="fisher"),
        # "gcn": GCNPredictor(encoding_type="gcn", hpo_wrapper=True),
        # "gp": GPPredictor(encoding_type="adjacency_one_hot"),
        # "gpwl": GPWLPredictor(
        #     ss_type=config.search_space,
        #     kernel_type="wloa",
        #     optimize_gp_hyper=True,
        #     h="auto",
        # ),
        # "grad_norm": ZeroCostV2(config, batch_size=64, method_type="grad_norm"),
        # "grasp": ZeroCostV2(config, batch_size=64, method_type="grasp"),
        # "jacov": ZeroCostV1(config, batch_size=64, method_type="jacov"),
        # "lce": LCEPredictor(metric=Metric.VAL_ACCURACY),
        # "lce_m": LCEMPredictor(metric=Metric.VAL_ACCURACY),
        # "lcsvr": SVR_Estimator(
        #     metric=Metric.VAL_ACCURACY, all_curve=False, require_hyper=False
        # ),
        # "lgb": LGBoost(encoding_type="adjacency_one_hot", hpo_wrapper=False),
        # "ngb": NGBoost(encoding_type="adjacency_one_hot", hpo_wrapper=False),
        # "rf": RandomForestPredictor(encoding_type="adjacency_one_hot", hpo_wrapper=False),
        # "snip": ZeroCostV2(config, batch_size=64, method_type="snip"),
        # "sotl": SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option="SoTL"),
        # "sotle": SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option="SoTLE"),
        # "sotlema": SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option="SoTLEMA"),
        # "sparse_gp": SparseGPPredictor(
        #     encoding_type="adjacency_one_hot", optimize_gp_hyper=True, num_steps=100
        # ),
        # "synflow": ZeroCostV2(config, batch_size=64, method_type="synflow"),
        # "valacc": EarlyStopping(metric=Metric.VAL_ACCURACY),
        # "valloss": EarlyStopping(metric=Metric.VAL_LOSS),
        # "var_sparse_gp": VarSparseGPPredictor(
        #     encoding_type="adjacency_one_hot", optimize_gp_hyper=True, num_steps=200
        # ),
        # path encoding experiments:
        # "bayes_lin_reg_path": BayesianLinearRegression(encoding_type="path"),
        # "bohamiann_path": BOHAMIANN(encoding_type="path"),
        # "dngo_path": DNGOPredictor(encoding_type="path"),
        # "gp_path": GPPredictor(encoding_type="path"),
        # "lgb_path": LGBoost(encoding_type="path", hpo_wrapper=False),
        # "ngb_path": NGBoost(encoding_type="path", hpo_wrapper=False),
        # # omni:
        # "omni_ngb": OmniNGBPredictor(
        #     encoding_type="adjacency_one_hot",
        #     config=config,
        #     zero_cost=["jacov"],
        #     lce=["sotle"],
        # ),
        # "omni_seminas": OmniSemiNASPredictor(
        #     encoding_type="seminas",
        #     config=config,
        #     semi=True,
        #     hpo_wrapper=False,
        #     zero_cost=["jacov"],
        #     lce=["sotle"],
        #     jacov_onehot=True,
        # ),
        # # omni ablation studies:
        # "omni_ngb_no_lce": OmniNGBPredictor(
        #     encoding_type="adjacency_one_hot", config=config, zero_cost=["jacov"], lce=[]
        # ),
        # "omni_seminas_no_lce": OmniSemiNASPredictor(
        #     encoding_type="seminas",
        #     config=config,
        #     semi=True,
        #     hpo_wrapper=False,
        #     zero_cost=["jacov"],
        #     lce=[],
        #     jacov_onehot=True,
        # ),
        # "omni_ngb_no_zerocost": OmniNGBPredictor(
        #     encoding_type="adjacency_one_hot", config=config, zero_cost=[], lce=["sotle"]
        # ),
        # "omni_ngb_no_encoding": OmniNGBPredictor(
        #     encoding_type=None, config=config, zero_cost=["jacov"], lce=["sotle"]
        # ),
    }

    supported_search_spaces = {
        "nasbench101": NasBench101SearchSpace(),
        "nasbench201": NasBench201SearchSpace(),
        "darts": DartsSearchSpace(),
        "nlp": NasBenchNLPSearchSpace(),
        'transbench101': TransBench101SearchSpace()
    }
    #    'transbench101_micro': TransBench101SearchSpace('micro'),
    #    'transbench101_macro': TransBench101SearchSpace('micro')}

    #}

    """
    If the API did not evaluate *all* architectures in the search space, 
    set load_labeled=True
    """
    load_labeled = True if config.search_space in ["darts", "nlp"] else False
    dataset_api = get_dataset_api(config.search_space, config.dataset)

    # initialize the search space and predictor
    utils.set_seed(config.seed)
    predictor = supported_predictors[pred] # config.predictor
    search_space = supported_search_spaces[config.search_space]

    ##########################################################################

    results_df = pd.DataFrame(columns=['trial', 'predictor', 'mae', 'rmse', 'pearson', 'spearman', 'kendalltau', 'fit_time', 'average_query_time'])

    # run evaluation trials
    for i in range(config.trials):
        print(f'Trial {i+1}')

        # re-initialize the PredictorEvaluator class, since results are object attribute
        predictor_evaluator = PredictorEvaluator(predictor, config=config)
        predictor_evaluator.adapt_search_space(
            search_space, load_labeled=load_labeled, dataset_api=dataset_api
        )

        results = predictor_evaluator.evaluate()[1]
        results_df = results_df.append({'trial': i+1, 
                                        'predictor':pred, # config.predictor 
                                        'mae': results['mae'],
                                        'rmse': results['rmse'],
                                        'pearson': results['pearson'],
                                        'spearman': results['spearman'],
                                        'kendalltau': results['kendalltau'],
                                        'fit_time': results['fit_time'],
                                        'average_query_time': results['average_query_time']
                                        }, ignore_index=True)

    # add mean values as last row
    results_df = results_df.append({'trial': 'average', 
                                    'predictor': pred,  #config.predictor
                                    'mae': np.mean(results_df['mae']),
                                    'rmse': np.mean(results_df['rmse']),
                                    'pearson': np.mean(results_df['pearson']),
                                    'spearman': np.mean(results_df['spearman']),
                                    'kendalltau': np.mean(results_df['kendalltau']),
                                    'fit_time': np.mean(results_df['fit_time']),
                                    'average_query_time': np.mean(results_df['average_query_time'])
                                    }, ignore_index=True)


    results_df.to_excel(f'results_{pred}.xlsx') # config.predictor


