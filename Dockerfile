FROM nvidia/cuda:10.0-cudnn7-runtime

# Install some utilities
RUN apt-get update && \
    apt-get install -y -q wget git vim libxrender1 libsm6 bzip2 && \
    apt-get clean

# Install miniconda
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA && \
    echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /miniconda/bin:$PATH

SHELL ["/bin/bash", "-l", "-c"]

# install deepchem with master branch
RUN conda update -n base conda && \
    git clone --depth=1 https://github.com/deepchem/deepchem.git && \
    cd deepchem && \
    gpu=1 bash scripts/install_deepchem_conda.sh deepchem && \
    source activate deepchem && python setup.py install

# install jax
RUN source activate deepchem && \
    PYTHON_VERSION=cp36 && \
    CUDA_VERSION=cuda100 && \
    PLATFORM=linux_x86_64 && \
    BASE_URL='https://storage.googleapis.com/jax-releases' && \
    pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.47-$PYTHON_VERSION-none-$PLATFORM.whl && \
    pip install --upgrade jax

# install additonal package
RUN source activate deepchem && \
    yes | pip install flake8 autopep8 torch torchvision

WORKDIR /root
