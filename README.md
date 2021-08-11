## Running

```
cd trax_code
rm -rf out; PYTHONPATH=trax python code/run_trax.py --output_dir=./out --config_file=configs/transformer_srl_electricity.gin --config=specs/transformer_ts_conv.py
```
or
```
cd trax_code
mrunner run specs/transformer_ts_conv.py
```

## Datasets

To use a dataset from GluonTS:

- Have GluonTS installed
```
pip install git+https://github.com/awslabs/gluon-ts.git
```
- Choose a dataset from https://github.com/awslabs/gluon-ts/blob/v0.8.0/src/gluonts/dataset/repository/datasets.py
- Replace the `datasets` variable in `gluon_evaluations/generate_evaluations.py`
- Optional: add estimators to the `estimators` variable in that script to evaluate them on the datasets. Leave `SeasonalNaivePredictor` to just quickly download and process the dataset.
- Run `generate_evaluations.py` and the dataset will be downloaded to `../datasets`. Copy it to `trax_code/data` to use with the trax code.

### `electricity_nips`

Most testing was done on the `electricity_nips` dataset (hourly electricity consumption data). It's included in `trax_code/data` and doesn't need to be downloaded. The `electricity_nips_small` version has the same training data, but about half the test data to speed up the serialized evaluation which can be a bit slow. The `small` version has 1280 rows of data, which is 20 minibatches of 64 rows, so it's easy to always evaluate on the same data.

The script in `trax_code/data/data_stats.py` show very basic statistics about the data in a dataset and can help with picking the min/max range of the discretization.
