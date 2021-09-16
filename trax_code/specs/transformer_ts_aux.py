from mrunner.helpers.specification_helper import create_experiments_helper


def repeat(n):
    def callback(params_configurations, **kwargs):
        params_configurations[:] = params_configurations * n
    return callback


# 50, 100, 200, 400
CONTEXT_SIZE = 200

# 64, 128, 256, 512, 768, 1024
D_MODEL = 128


experiments_list = create_experiments_helper(
    experiment_name='Transformer TS with aux data',

    base_config={
        'train.steps': 5002,

        'BoxSpaceSerializer.max_range': (0.0, 50000.0),
        'TimeSeriesModelAux.low': 0.0,
        'TimeSeriesModelAux.high': 50000.0,

        'TimeSeriesModelAux.precision': 2,
        'TimeSeriesModelAux.vocab_size': 512,

        'CreateGluonTSInputs.max_length': CONTEXT_SIZE,
        'SerializedModelEvaluation.context_lengths': (CONTEXT_SIZE - 24,),
        'SerializedModelEvaluation.horizon_lengths': (24,),

        'SerializedModelEvaluation.normalize_context': False,
        'CreateGluonTSInputs.normalize_train': False,

        'TransformerLMAux.n_layers': 2,
        'TransformerLMAux.d_model': D_MODEL,
        'TransformerLMAux.d_ff': 2 * D_MODEL,
        'TransformerLMAux.n_heads': 2,
        'TransformerLMAux.dropout': 0.1,

        'CreateGluonTSInputs.aux_data_type': 'none', # none, hour, weekday
        'TimeSeriesModelAux.aux_vocab_size': 24,
    },
    params_grid={
        'CreateGluonTSInputs.aux_data_type': ['none', 'hour', 'weekday'],
    },
    script='python3 code/run_trax.py --output_dir=./out --config_file=configs/transformer_srl_electricity_aux.gin',
    exclude=['.git', '.pytest_cache', 'alpacka.egg-info', 'out', 'env', 'traces', 'alpacka/.git', 'alpacka/tools', 'alpacka/traces', 'alpacka/out', 'singularity', 'exps', 'alpacka/fixtures', 'trax/.git', 'trax/trax/supervised/testdata'],
    python_path=':trax',
    tags=[globals()['script'][:-3]],
    env={'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/usr/local/cuda-11'},
    with_neptune=False,
    project_name='lraszkiewicz/sequences',
    callbacks=[repeat(1)],
)
