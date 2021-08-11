import argparse
import atexit
import collections
import datetime
import neptune
import os
import pickle

import cloudpickle
import gin

import numpy as np

from trax.supervised import callbacks
from trax.supervised import trainer_lib


neptune_experiment = None


def get_configuration(spec_path):
    """Get mrunner experiment specification and gin-config overrides."""
    try:
        with open(spec_path, 'rb') as f:
            specification = cloudpickle.load(f)
    except pickle.UnpicklingError:
        with open(spec_path) as f:
            vars_ = {'script': os.path.basename(spec_path)}
            exec(f.read(), vars_)  # pylint: disable=exec-used
            specification = vars_['experiments_list'][0].to_dict()
            print('NOTE: Only the first experiment from the list will be run!')

    parameters = specification['parameters']
    gin_bindings = []
    for key, value in parameters.items():
        if key == 'imports':
            for module_str in value:
                binding = f'import {module_str}'
                gin_bindings.append(binding)
            continue

        if isinstance(value, str) and not value[0] in ('@', '%', '{', '(', '['):
            binding = f'{key} = "{value}"'
        else:
            binding = f'{key} = {value}'
        gin_bindings.append(binding)

    return specification, gin_bindings


def extract_bindings(config_str):
    """Extracts bindings from a Gin config string.

    Args:
        config_str (str): Config string to parse.

    Returns:
        List of (name, value) pairs of the extracted Gin bindings.
    """
    # Really crude parsing of gin configs.
    # Remove line breaks preceded by '\'.
    config_str = config_str.replace('\\\n', '')
    # Remove line breaks inside parentheses. Those are followed by indents.
    config_str = config_str.replace('\n    ', '')
    # Indents starting with parentheses are 3-space.
    config_str = config_str.replace('\n   ', '')
    # Lines containing ' = ' are treated as bindings, everything else is
    # ignored.
    sep = ' = '

    bindings = []
    for line in config_str.split('\n'):
        line = line.strip()
        if sep in line:
            chunks = line.split(sep)
            name = chunks[0].strip()
            value = sep.join(chunks[1:]).strip()
            bindings.append((name, value))
    return bindings


@gin.configurable
class NeptuneCallback(callbacks.TrainingStepCallback):

    def __init__(self, loop, log_every=100):
        super().__init__(loop)

        global neptune_experiment
        self._experiment = neptune_experiment

        self._log_every = log_every
        # Nested dict {mode: {metric: int}}, containing the lengths of metric
        # history buffers.
        self._last_lengths = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

    def call_at(self, step):
        return step == 1 or step % self._log_every == 0

    def on_step_begin(self, step):
        pass

    def on_step_end(self, step):
        history = self._loop.history

        if step == 1:
            # Save the operative config after the first step, so once we're sure
            # everything has initialized.
            self._save_operative_config()

        # Send the metrics starting from the last recorded point.
        for mode in history.modes:
            metric_lengths = self._last_lengths[mode]
            for metric in history.metrics_for_mode(mode):
                length = metric_lengths[metric]
                steps_and_values = history.get(mode, metric)
                for (step, value) in steps_and_values[length:]:
                    print(f'{mode}/{metric}', step, value)
                    self._experiment.send_metric(
                        f'{mode}/{metric}', step, value)
                metric_lengths[metric] = len(steps_and_values)

        self._last_step = step

    def _save_operative_config(self):
        # Save the Gin operative config (the part of the config that is actually
        # used) to a file and to Neptune.
        config_path = os.path.join(self._loop.output_dir, 'config.gin')
        config_str = gin.operative_config_str()
        with open(config_path, 'w') as f:
            f.write(config_str)

        for (name, value) in extract_bindings(config_str):
            self._experiment.set_property(name, value)


def configure_neptune(specification):
    """Configures the Neptune experiment."""
    if 'NEPTUNE_API_TOKEN' not in os.environ:
        raise KeyError('Environment variable NEPTUNE_API_TOKEN is not set!')

    git_info = specification.get('git_info', None)
    if git_info:
        git_info.commit_date = datetime.datetime.now()

    neptune.init(project_qualified_name=specification['project'])

    # Set pwd property with path to experiment.
    properties = {'pwd': os.getcwd()}
    global neptune_experiment
    neptune_experiment = neptune.create_experiment(
        name=specification['name'],
        tags=specification['tags'],
        params=specification['parameters'],
        properties=properties,
        git_info=git_info)
    atexit.register(neptune.stop)


def add_callback(callback_class):
    callbacks = gin.query_parameter('train.callbacks')
    callbacks = tuple(callbacks) + (callback_class,)
    gin.bind_parameter('train.callbacks', callbacks)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory.')
    parser.add_argument(
        '--config_file', action='append',
        help='Gin config files.')
    parser.add_argument(
        '--config', action='append',
        help='Gin config overrides.')
    return parser.parse_args()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    args = _parse_args()
    gin_bindings = args.config

    spec_path = gin_bindings.pop()

    specification, overrides = get_configuration(spec_path)
    gin_bindings = overrides + gin_bindings

    gin.parse_config_files_and_bindings(args.config_file, gin_bindings, finalize_config=False)

    configure_neptune(specification)

    add_callback(NeptuneCallback)

    gin.finalize()

    trainer_lib.train(output_dir=args.output_dir)
