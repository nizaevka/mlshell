"""A setuptools based setup module."""


import setuptools
import os


NAME = "mlshell"
DESCRIPTION = "Catalyst. PyTorch framework for DL & RL research and development."
URL = "https://github.com/nizaevka/mlshell"
REQUIRES_PYTHON = ">=3.6"
PATH = os.path.abspath(os.path.dirname(__file__))


# get text
def parse_text(filename, splitlines=False):
    with open(os.path.join(PATH, filename), "r", encoding='utf-8') as f:
        if splitlines:
            return f.read().splitlines()
        else:
            return f.read()


# get version
version = {}
with open("src/{}/version.py".format(NAME)) as fp:
    exec(fp.read(), version)


setuptools.setup(
    name=NAME,
    version=version['__version__'],
    author="nizaevka",
    author_email="nizaevka@gmail.com",
    description="Shell around Ml libraries.",
    keywords='ml sklearn',
    long_description=parse_text('README.md', splitlines=False),
    long_description_content_type="text/markdown",
    url=URL,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    install_requires=parse_text('requirements.txt', splitlines=True),
    # $ pip install package[dev]
    extras_require={
        'dev': parse_text('requirements_dev.txt', splitlines=True),
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=REQUIRES_PYTHON,
    project_urls={
        'Documentation': 'https://mlshell.readthedocs.io/',
        'Source': "https://github.com/nizaevka/mlshell",
        'Tracker': "https://github.com/nizaevka/mlshell/issues",
    },
)
