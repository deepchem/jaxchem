from setuptools import setup, find_packages


def _get_version():
    with open('jaxchem/__init__.py') as fp:
        for line in fp:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                return g['__version__']
        raise ValueError('`__version__` not defined in `jaxchem/__init__.py`')


def _parse_requirements(requirements_txt_path):
    with open(requirements_txt_path) as fp:
        return fp.read().splitlines()


setup(
    name='jaxchem',
    version=_get_version(),
    url='https://github.com/deepchem/jaxchem',
    maintainer='DeepChem contributors',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    license='MIT',
    description='An experimental repository to work on some Jax models for chemistry',
    keywords=[
        'jax'
        'deepchem',
        'life-science',
        'drug-discovery',
    ],
    packages=find_packages(),
    project_urls={
        'Bug Tracker': 'https://github.com/deepchem/jaxchem/issues',
        'Source': 'https://github.com/deepchem/jaxchem',
    },
    install_requires=_parse_requirements('requirements.txt'),
    python_requires='>=3.6'
)
