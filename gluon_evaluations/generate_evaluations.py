# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Original file:
# https://github.com/awslabs/gluon-ts/blob/master/evaluations/generate_evaluations.py

"""
This example evaluates models in gluon-ts.
Evaluations are stored for each model/dataset in a json file and all results can then
be displayed with `show_results.py`.
"""
import json
import os
from pathlib import Path
from typing import Dict

from gluonts.model.estimator import Estimator
from gluonts.dataset.repository.datasets import get_dataset, dataset_names
from gluonts.evaluation import backtest_metrics
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.seq2seq import MQRNNEstimator, RNN2QRForecaster
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.transformer import TransformerEstimator
from gluonts.mx.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator

metrics_persisted = ["mean_wQuantileLoss", "ND", "RMSE", "sMAPE", "MASE"]

datasets = ["electricity_nips"]
# datasets = ["m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly"]
# datasets = ["tourism_monthly", "tourism_quarterly", "tourism_yearly"]
# datasets = dataset_names

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


def persist_evaluation(
    estimator_name: str,
    dataset: str,
    evaluation: Dict[str, float],
    evaluation_path: str = "./",
):
    """
    Saves an evaluation dictionary into `evaluation_path`
    """
    path = Path(evaluation_path) / dataset / f"{estimator_name}.json"

    os.makedirs(path.parent, exist_ok=True)

    print(evaluation)
    evaluation = {
        m: v for m, v in evaluation.items() if m in metrics_persisted
    }
    evaluation["dataset"] = dataset
    evaluation["estimator"] = estimator_name

    with open(path, "w") as f:
        f.write(json.dumps(evaluation, indent=4, sort_keys=True))


if __name__ == "__main__":

    for dataset_name in datasets:
        estimators = [
            # (DeepAREstimator, {}),
            # (RForecastPredictor, {
            #     "method_name": "arimaapprox",
            # }),
            # (RForecastPredictor, {
            #     "method_name": "arimans",
            # }),
            # (RForecastPredictor, {
            #     "method_name": "arima",
            # }),
            # (RForecastPredictor, {
            #     "method_name": "arimafast",
            # }),
            # (RForecastPredictor, {
            #     "method_name": "etsMAZ",
            # }),
            # (ProphetPredictor, {}),
            (SeasonalNaivePredictor, {}),
            # (MQRNNEstimator, {}),
            # (TransformerEstimator, {
                # "dropout_rate": 0.1,
                # "embedding_dimension": 20,
                # "model_dim": 32,
                # "act_type": "softrelu",
                # "batch_size": 32,
                # "trainer": Trainer(epochs=100),
            # }),
            # (model.simple_feedforward.SimpleFeedForwardEstimator, {}),
            # model.deepar.DeepAREstimator,
            # model.NPTSPredictor,
        ]

        for est_class, est_kwargs in estimators:
            dataset = get_dataset(
                dataset_name=dataset_name,
                regenerate=False,
                path=Path("../datasets/"),
            )

            estimator = est_class(
                prediction_length=dataset.metadata.prediction_length,
                freq=dataset.metadata.freq,
                **est_kwargs
            )

            estimator_name = type(estimator).__name__
            if "method_name" in est_kwargs:
                estimator_name += "_" + est_kwargs["method_name"]

            print(f"evaluating {estimator_name} on {dataset_name}")

            if isinstance(estimator, Estimator):
                predictor = estimator.train(training_data=dataset.train)
            else:
                predictor = estimator

            agg_metrics, item_metrics = backtest_metrics(
                test_dataset=dataset.test,
                predictor=predictor,
            )

            persist_evaluation(
                estimator_name=estimator_name,
                dataset=dataset_name,
                evaluation=agg_metrics,
                evaluation_path=dir_path,
            )
