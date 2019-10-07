# Weakly Supervised Learning

This is the source code for the book *Weakly Supervied Learning: Doing More with Less Data* (O'Reilly, 2020) by Russell Jurney. The book itself is open source and can be found at <https://github.com/rjurney/weakly_supervised_learning> :)

## Install

### Prerequisites

* Linux
* Mac OS X can work if you have an NVIDIA GPU and remove `cupy-cuda100` from the environment
* [Git](https://git-scm.com/download) is used to check out the book’s source code
* Python 3.7+ - I recommend [Anaconda Python](https://www.anaconda.com/distribution/), but any Python will do
* An NVIDIA graphics card - you can work the examples without one, but CPU training is painfully slow
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) - for GPU acceleration in CuPy and Tensorflow
* [cuDNN](https://developer.nvidia.com/cudnn) - for GPU acceleration in Tensorflow

### requirements.txt

Unfortunately not all of the code will run on OS X because it does not have support for NVIDIA graphics cards. Accordingly I have used [`requirements.in`](requirements.in) to generate
[`requirements.linux.txt`](requirements.linux.txt) using  `pip-compile`, but you will have to remove `cupy-cuda100` and use `pip` or build your own `requirements.mac.txt` and then use `pip` to install the requirements.

```bash
pip-compile --output-file requirements.<os>.txt requirements.in
```

### Conda Environment

There is also a conda environment defined in [`environment.linux.yml`](environment.linux.yml) that can be used to instantiate a `conda` environment.

```bash
conda env create -n weak -f environment.yml
conda activate
```

## Jupyter Notebooks

Each chapter directory has one or more Jupyter Notebooks containing the complete text of the book and its examples for you to follow along with. 

You will want to run Jupyter from this directory.

```bash
jupyter notebook
```

Or, what I use:

```bash
nohup jupyter notebook &
```

You may then monitor the notebook with:

```bash
tail -f nohup.out
```

and can exit that log at any time without shutting down the Jupyter server.
