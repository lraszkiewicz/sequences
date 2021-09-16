from mrunner.helpers.specification_helper import create_experiments_helper


def repeat(n):
    def callback(params_configurations, **kwargs):
        params_configurations[:] = params_configurations * n
    return callback


# 50, 100, 200, 400
CONTEXT_SIZE = 600

# 64, 128, 256, 512, 768, 1024
D_MODEL = 128


experiments_list = create_experiments_helper(
    experiment_name='Transformer TS with norm',

    base_config={
        'train.steps': 5002,

        'BoxSpaceSerializer.max_range': (-20.0, 20.0),
        'TimeSeriesModel.low': -20.0,
        'TimeSeriesModel.high': 20.0,

        'TimeSeriesModel.precision': 2,
        'TimeSeriesModel.vocab_size': 512,

        'CreateGluonTSInputs.max_length': CONTEXT_SIZE,
        'SerializedModelEvaluation.context_lengths': (CONTEXT_SIZE - 24,),
        'SerializedModelEvaluation.horizon_lengths': (24,),

        'SerializedModelEvaluation.normalize_context': True,
        'CreateGluonTSInputs.normalize_train': True,

        'CreateGluonTSInputs.aux_data_type': 'none',

        'TransformerLM.n_layers': 2,
        'TransformerLM.d_model': D_MODEL,
        'TransformerLM.d_ff': 2 * D_MODEL,
        'TransformerLM.n_heads': 2,
        'TransformerLM.dropout': 0.1,

        'TransformerLM.conv_before_dropout': False,
        'TransformerLM.conv_after_dropout': False,
        'TransformerLM.conv_relu': False,
        'TransformerLM.conv_kernel_size': 8,
    },
    params_grid={
        # 'TimeSeriesModel.precision': [1],
        # 'TimeSeriesModel.vocab_size': [512, 1024, 2048],
    },
    script='python3 code/run_trax.py --output_dir=./out --config_file=configs/transformer_srl_electricity.gin',
    exclude=['.git', '.pytest_cache', 'alpacka.egg-info', 'out', 'env', 'traces', 'alpacka/.git', 'alpacka/tools', 'alpacka/traces', 'alpacka/out', 'singularity', 'exps', 'alpacka/fixtures', 'trax/.git', 'trax/trax/supervised/testdata'],
    python_path=':trax',
    tags=[globals()['script'][:-3]],
    env={'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/usr/local/cuda-11'},
    with_neptune=True,
    callbacks=[repeat(1)],
)
