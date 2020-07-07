Installation
============

pip installation
-------------------------

JAXChem requires the following packages.

- JAX (jax==0.1.69, jaxlib==0.1.47)
- Haiku (==0.0.1)
- typing-extensions (>=3.7.4)

| First, you have to install JAX. Please confirm how to install JAX from `here`_.
| After installing JAX, please run the following commands.

.. code-block:: bash

    // install jaxchem
    $ pip install git+https://github.com/deepchem/jaxchem


docker installation
-------------------------

Please run the following commands.

.. code-block:: bash

    $ git clone https://github.com/deepchem/jaxchem.git
    $ cd jaxchem
    $ docker build . -t jaxchem


.. _`here`: https://github.com/google/jax/tree/jax-v0.1.69#installation
