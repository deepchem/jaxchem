# JAXChem

[![](https://github.com/deepchem/jaxchem/workflows/main/badge.svg)](https://github.com/deepchem/jaxchem/actions?query=workflow%3Amain)

JAXChem is a JAX-based deep learning library for complex and versatile chemical modelings.  
We welcome to contribute any chemical modelings for JAX.

## (WIP) Installation

### pip installation

JAXChem requires the following packages.

- JAX (jax==0.1.69, jaxlib==0.1.47)
- Haiku (==0.0.1)
- typing-extensions (>=3.7.4)

First, you have to install JAX. Please confirm how to install JAX [here](https://github.com/google/jax/tree/jax-v0.1.69#installation).  
After installing JAX, please run the following commands.

```bash
// install haiku v0.0.1
$ pip install dm-haiku==0.0.1

// install jaxchem
$ pip install git+https://github.com/deepchem/jaxchem
```

### docker installation

```bash
$ git clone https://github.com/deepchem/jaxchem.git
$ cd jaxchem
$ docker build . -t jaxchem
```
