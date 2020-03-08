ALPHA UNDERCONSTRUCTED
<div align="center">

[![Mlshell logo](pictures/mlshell_logo.PNG?raw=true)](https://github.com/nizaevka/mlshell)

**Unified ML framework**

[![Build Status](https://travis-ci.org/nizaevka/mlshell.svg?branch=master)](https://travis-ci.org/nizaevka/mlshell)
[![PyPi version](https://img.shields.io/pypi/v/mlshell.svg)](https://pypi.org/project/mlshell/)
[![PyPI Status](https://pepy.tech/badge/mlshell)](https://pepy.tech/project/mlshell)
[![Docs](https://readthedocs.org/projects/mlshell/badge/?version=latest)](https://mlshell.readthedocs.io/en/latest/)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/nizaevka)

</div>

**MLshell** is a framework for ML research and development:
- Fast and simple pipeline prototyping and parameters tuning.
- Unified ml pipeline.
- Stable CV scheme.
- Production ready.
- One conf file rule all.
- Simple result analyse.
- Unified plots.
- Common EDA techniques.
- Pure python.

[![Workflow](docs/source/_static/images/workflow.JPG?raw=true)]()

For details, please refer to
 [Concepts](https://mlshell.readthedocs.io/en/latest/Concepts.html>).

--

## Installation

#### PyPi [![PyPi version](https://img.shields.io/pypi/v/mlshell.svg)](https://pypi.org/project/mlshell/) [![PyPI Status](https://pepy.tech/badge/mlshell)](https://pepy.tech/project/mlshell)

```bash
pip install -U mlshell
```

<details>
<summary>Specific versions with additional requirements</summary>
<p>

```bash
pip install catalyst[dev]        # installs dependencies for development
```
</p>
</details>

#### Docker [![Docker Pulls](https://img.shields.io/docker/pulls/nizaevka/mlshell)](https://hub.docker.com/r/nizaevka/mlshell/tags)

```bash
docker run -it nizaevka/mlshell
```

MLshell is compatible with: Python 3.6+.


## Getting started

```python
import mlshell
```
see Docs for details ;)

## Docs and examples
An overview and API documentation can be found here
[![Docs](https://readthedocs.org/projects/mlshell/badge/?version=latest)](https://readthedocs.org/mlshell/en/latest/?badge=latest)

Check **[examples folder](examples)** of the repository:
- For regression example please follow [Allstate claims severity](examples/regression).
- For classification example please follow [IEEE-CIS Fraud Detection](examples/classification).

## Contribution guide

We appreciate all contributions.
If you are planning to contribute back bug-fixes,
please do so without any further discussion.
If you plan to contribute new features, utility functions or extensions,
please first open an issue and discuss the feature with us.

- Please see the [contribution guide](CONTRIBUTING.md) for more information.
- By participating in this project, you agree to abide by its [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the Apache License, Version 2.0 see the [LICENSE](LICENSE) file for details
[![License](https://img.shields.io/github/license/nizaevka/mlshell.svg)](LICENSE)