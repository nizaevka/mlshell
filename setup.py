"""A setuptools based setup module."""


import setuptools
import os


NAME = "mlshell"
DESCRIPTION = "MLshell. ML framework."
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
with open("src/{}/__version__.py".format(NAME)) as fp:
    exec(fp.read(), version)

# prevent install dependencies for readthedoc build (there is no way to set --no-deps in pip install)
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd and False:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = parse_text('requirements.txt', splitlines=True)


setuptools.setup(
    name=NAME,
    version=version['__version__'],
    author="nizaevka",
    author_email="knizaev@gmail.com",
    description="Ml framework.",
    keywords='ml sklearn workflow',
    long_description=parse_text('README.md', splitlines=False),
    long_description_content_type="text/markdown",
    url=URL,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    install_requires=INSTALL_REQUIRES,
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
