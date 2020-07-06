# JAXChem

[![](https://github.com/deepchem/jaxchem/workflows/main/badge.svg)](https://github.com/deepchem/jaxchem/actions?query=workflow%3Amain)

JAXChem is a JAX-based deep learning library for complex and versatile chemical modelings.  
We welcome to contribute any chemical modelings with JAX.

NOTE : This is a 2020 GSoC project with Open Chemistry. Please confirm the details from [here](https://summerofcode.withgoogle.com/projects/#4840359860895744).

## (WIP) Installation

### pip installation

JAXChem requires the following packages.

- JAX (jax==0.1.69, jaxlib==0.1.47)
- Haiku (==0.0.1)
- typing-extensions (>=3.7.4)

First, you have to install JAX. Please confirm how to install JAX [here](https://github.com/google/jax/tree/jax-v0.1.69#installation).  
After installing JAX, please run the following commands.

```bash
// install jaxchem
$ pip install git+https://github.com/deepchem/jaxchem
```

### docker installation

Please run the following commands.

```bash
$ git clone https://github.com/deepchem/jaxchem.git
$ cd jaxchem
$ docker build . -t jaxchem
```