<div align="center">

[![Mlshell logo](https://github.com/nizaevka/mlshell/blob/master/docs/source/_static/images/logo.png?raw=true)](https://github.com/nizaevka/mlshell)

**Simple ML framework**

[![Build Status](https://travis-ci.org/nizaevka/mlshell.svg?branch=master)](https://travis-ci.org/nizaevka/mlshell)
[![PyPi version](https://img.shields.io/pypi/v/mlshell.svg)](https://pypi.org/project/mlshell/)
[![PyPI Status](https://pepy.tech/badge/mlshell)](https://pepy.tech/project/mlshell)
[![Docs](https://readthedocs.org/projects/mlshell/badge/?version=latest)](https://mlshell.readthedocs.io/en/latest/)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/nizaevka)

</div>

- Auto ML.
- Unified pipeline.
- Stable CV scheme.
- One configuration file.
- Multi-stage optimization.

Mlshell based on [Pycnfg](https://github.com/nizaevka/pycnfg) library.
All parameters controlled from single Python configuration.

![Workflow](https://github.com/nizaevka/mlshell/blob/master/docs/source/_static/images/scheme.png?raw=true)

For details, please refer to
 [Concepts](https://mlshell.readthedocs.io/en/latest/Concepts.html).

## Installation

#### PyPi [![PyPi version](https://img.shields.io/pypi/v/mlshell.svg)](https://pypi.org/project/mlshell/) [![PyPI Status](https://pepy.tech/badge/mlshell)](https://pepy.tech/project/mlshell)

```bash
pip install mlshell
```

<details>
<summary>Development installation (tests and docs): </summary>
<p>

```bash
pip install mlshell[dev]
```
</p>
</details>

#### Docker [![Docker Pulls](https://img.shields.io/docker/pulls/nizaevka/mlshell)](https://hub.docker.com/r/nizaevka/mlshell/tags)

```bash
docker run -it nizaevka/mlshell
```
Tested on: Python 3.6+.

## Docs
[![Docs](https://readthedocs.org/projects/mlshell/badge/?version=latest)](https://readthedocs.org/mlshell/en/latest/?badge=latest)

## Getting started

<details>
<summary>example</summary>
<p>

```python
"""Configuration example - tune LGBM on iris dataset."""
import lightgbm
import mlshell
import pycnfg
import sklearn.datasets


# Optimization hp ranges.
hp_grid = {
    'reduce_dimensions__skip': [False, True],  # PCA on/off
    # 'estimate__classifier__n_estimators': np.linspace(50, 1000, 10, dtype=int),
    # ...
}

"""
The single configuration CNFG controls whole ml task.
Each section sub-configurations produce object (pipeline/metric/dataset/workflow)
pipeline-wise:
    object init state
        => transform object with steps (producer methods)
            => store result
Sub-configuration with greater priority (workflow) could utilize previously
created objects.
"""
CNFG = {
    # Pipeline section - make pipeline object(s).
    'pipeline': {
        'lgbm': {
            'init': mlshell.Pipeline,
            'producer': mlshell.PipelineProducer,
            'priority': 3,
            'steps': [
                ('make', {
                    'estimator_type': 'classifier',
                    'steps': mlshell.pipeline.Steps,
                    'estimator': lightgbm.sklearn.LGBMClassifier(
                        num_leaves=5, max_depth=5, n_estimators=100,
                        random_state=42),  # last stage of pipeline.
                }),
            ],
        }
    },
    # Metric section - make scorer object(s).
    'metric': {
        'accuracy': {
            'init': mlshell.Metric,
            'producer': mlshell.MetricProducer,
            'priority': 4,
            'steps': [
                ('make', {
                    'score_func': sklearn.metrics.accuracy_score,
                    'greater_is_better': True,
                }),
            ],
        },
        'confusion_matrix': {
            'init': mlshell.Metric,
            'producer': mlshell.MetricProducer,
            'priority': 4,
            'steps': [
                ('make', {
                    'score_func': sklearn.metrics.confusion_matrix,
                }),
            ],
        },
    },
    # Dataset section - dataset loading/preprocessing/splitting.
    'dataset': {
        'train': {
            'init': mlshell.Dataset({
                'data': sklearn.datasets.load_iris(as_frame=True).frame
            }),
            'producer': mlshell.DatasetProducer,
            'priority': 5,
            'steps': [
                ('preprocess', {'targets_names': ['target']}),
                ('split', {'train_size': 0.75, 'shuffle': True,
                           'random_state': 42}),
            ],
        },
    },
    # Workflow section
    # - fit/predict pipelines on datasets,
    # - optimize/validate metrics,
    # - predict/dump predictions on datasets.
    'workflow': {
        'conf': {
            'init': {},
            'producer': mlshell.Workflow,
            'priority': 6,
            'steps': [
                # Optimize 'lgbm' pipeline on 'train' subset of 'train' dataset
                # on hp combinations from 'hp_grid'. Score and refit on
                # 'accuracy' scorer.
                ('optimize', {
                    'pipeline_id': 'pipeline__lgbm',
                    'dataset_id': 'dataset__train',
                    'subset_id': 'train',
                    'metric_id': ['metric__accuracy'],
                    'hp_grid': hp_grid,
                    'gs_params': {
                        'n_iter': None,
                        'n_jobs': 1,
                        'refit': 'metric__accuracy',
                        'cv': sklearn.model_selection.KFold(n_splits=3,
                                                            shuffle=True,
                                                            random_state=42),
                        'verbose': 1,
                        'pre_dispatch': 'n_jobs',
                        'return_train_score': True,
                    },
                }),
                # Validate 'lgbm' pipeline on 'train' and 'test' subsets of
                # 'train' dataset with 'accuracy' and 'confusion_matrix'.
                ('validate', {
                    'pipeline_id': 'pipeline__lgbm',
                    'dataset_id': 'dataset__train',
                    'subset_id': ['train', 'test'],
                    'metric_id': ['metric__accuracy',
                                  'metric__confusion_matrix'],
                }),
            ],
        },
    },
}


if __name__ == '__main__':
    # mlshell.CNFG contains default section / configuration keys for typical ml
    # task, including pretty logger and project path detection.
    objects = pycnfg.run(CNFG, dcnfg=mlshell.CNFG)
```
</p>
</details>


## Examples
Check **[examples folder](https://github.com/nizaevka/mlshell/blob/master/examples)**.

## Contribution guide
- [contribution guide](https://github.com/nizaevka/mlshell/blob/master/CONTRIBUTING.md).

## License
Apache License, Version 2.0 [LICENSE](https://github.com/nizaevka/mlshell/blob/master/LICENSE).
[![License](https://img.shields.io/github/license/nizaevka/mlshell.svg)](LICENSE)