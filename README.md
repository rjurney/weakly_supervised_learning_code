# Weakly Supervised Learning

This is the source code for the book *Weakly Supervied Learning: Doing More with Less Data* (O'Reilly, 2020) by Russell Jurney. The book itself is open source and can be found at <https://github.com/rjurney/weakly_supervised_learning> :)

In my previous book, [Agile Data Science 2.0](http://shop.oreilly.com/product/0636920051619.do) (O’Reilly Media, 2017), I [setup EC2 and Vagrant environments](https://github.com/rjurney/Agile_Data_Code_2) in which to run the book’s code but since 2017 the Python ecosystem has developed to the point that I am going to refrain from providing thorough installation documentation for every requirement. In this book I provide a Docker setup that is easy to use and also provide Anaconda and PyPi environments if you wish the run the code yourself locally. The website for each library is a better resource than I can possibly create, and they are updated and maintained more frequently than this book. I will instead list requirements, link to the project pages and let the reader install the requirements themselves. If you want to use a pre-built environment, use the `Dockerfile` and  `docker-compose.yml` files included in the [code repository for the book](https://github.com/rjurney/weakly_supervised_learning_code) will “just work” on any operating system that Docker runs on: Linux, Mac OS X, Windows.

## Software Prerequisites

* Linux, Mac OS X or Windows
* [Git](https://git-scm.com/download) is used to check out the book’s source code
* [Docker](https://www.docker.com/get-started) is used to run the book’s examples in the same environment I wrote them in

## Running Docker via `docker-compose`

To run the examples using `docker-compose` simply run:

```bash
docker-compose up --build -d
```

The `--build` builds the container using the local directory the first time you run it. The `-d` puts the Jupyter web server in the background, and is optional. 

Now visit [http://localhost:8888](http://localhost:8888)

If you run into problems, remove the `-d` argument to run it in the foreground and [file an issue](https://github.com/rjurney/weakly_supervised_learning_code/issues/new) on Github with the command you used and the complete error output.

## Running Docker directly via the `Dockerfile`

You can also build and run the docker image directly via the `docker` command and the `Dockerfile` :

```bash
docker build --tag weakly_supervised_learning .
docker container run \
    --publish 8888:8888 \
    --detach \
    --name weakly_supervised_learning \
    -v .:/weakly_supervised_learning_code \
        weakly_supervised_learning
```

Now visit [http://localhost:8888](http://localhost:8888)

If you run into problems, remove the `--detach` argument to run it in the foreground and [file an issue](https://github.com/rjurney/weakly_supervised_learning_code/issues/new) on Github with the command you used and the complete error output.

## Running via Docker Hub

You can also use Docker Hub to pull and run the image directly:

```bash
docker pull rjurney/weakly_supervised_learning
docker run weakly_supervised_learning # add a volume for .
```

Now visit [http://localhost:8888](http://localhost:8888)

## Bugs, Errors or other Problems

If you run into problems, make sure you have the latest code with `git pull origin master` and if it persist then [search the Github issues](https://github.com/rjurney/weakly_supervised_learning_code/issues?utf8=%E2%9C%93&q=is%3Aissue+) for the error. If a fix isn’t in the issues, then [create a ticket](https://github.com/rjurney/weakly_supervised_learning_code/issues/new) and include the command you ran and the complete output of that command. You can find the Book’s issues on Github here: [https://github.com/rjurney/weakly_supervised_learning_code/issues](https://github.com/rjurney/weakly_supervised_learning_code/issues).

## Running the Code Locally

I’ve defined two Python environments for the book using Conda and a Virtual Environment. Once you have setup the requirements, you can easily reproduce the environment in which the book was written and tested.

### Software Prerequisites

The following requirements are needed if you run the code locally:

* Python 3.7+ - I recommend [Anaconda Python](https://www.anaconda.com/distribution/), but any Python will do
* `conda` or `virtualenv` to recreate the Python environment I wrote the examples in
* Recommended: An NVIDIA graphics card - you can work the examples without one, but CPU training is painfully slow
* Recommended: [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) - for GPU acceleration in CuPy and Tensorflow
* Recommended: [cuDNN](https://developer.nvidia.com/cudnn) - for GPU acceleration in Tensorflow

The file `environment.yml` lists them for the `conda` environment system used by [Anaconda Python](https://www.anaconda.com/distribution/). The library dependencies for the book are also defined in `requirements.in`, which [PyPi - the Python Package Index](https://pypi.org/) can use via the `pip` command to install them. I recommend PyPi users create a [Virtual Environment](https://virtualenv.pypa.io/en/latest/user_guide.html#introduction) to ensure you replicate the book’s environment accurately. 

The examples in the book are run as [Jupyter Notebooks](https://jupyter.org/). Jupyter is included in both `conda` and `pip` environments.

### Anaconda Python 3

To create a `conda` environment for the book, run:

```bash
conda env create -f environment.yml
conda activate weak
```

To deactivate the environment, run:

```bash
conda deactivate
```

### Virtual Environment

To create a Virtual Environment in which to install the PyPi dependencies, run:

```bash
pip install --upgrade virtualenv
virtualenv -p `which python3` weak
source weak/bin/activate
pip install -r requirements.in
```

To deactivate the Virtual Environment, run:

```bash
source deactivate
```

### Running Jupyter

If you’re using Docker, the image will install and run Jupyter for you. If you’re using your own Python environment, you need to run Jupyter:

```bash
cd </path/to/weakly_supervised_learning_code>
jupyter notebook &
```

Then visit [http://localhost:8888](http://localhost:8888) and open [Introduction.ipynb](https://github.com/rjurney/weakly_supervised_learning_code/blob/master/Introduction.ipynb) or select the chapter file you want to read and run.
