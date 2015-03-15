from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

setup(
    name='apsis',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.1',

    description='Toolkit for hyperparameter optimization for machine learning algorithms.',
    long_description='Our goal is to provide a flexible, simple and scaleable approach - parallel, on clusters and/or on your own machine. Check out our usage tutorials to get started or the design pages to understand how apsis works.',

    # The project's main homepage.
    url='https://github.com/FrederikDiehl/apsis',

    # Author details
    author='Frederik Diehl, Andreas Jauch',
    author_email= 'apsis-users@googlegroups.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='bayesian optimization machine learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'documentation', 'diagrams']),

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['scipy', 'numpy', 'gpy>=0.6.0', 'matplotlib'],
    
    package_data={
    'apsis': ['config/*'],
    },


    # List additional groups of dependencies here (e.g. development dependencies).
    # You can install these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    extras_require = {
        'test': ['coverage', 'nosetests'],
        'demos': ['sklearn']
    },
)
