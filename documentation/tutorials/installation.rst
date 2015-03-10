Installing Apsis 
****************

This guide provides instructions for how to get apsis running on your system. The guide is manily targetted at Ubuntu/Debian and Mac OS users, however as a user of another linux based OS you should easily to be able to follow this guide with the methods used in your distro.


Prerequisites
=============

Apsis requires the following **python frameworks** and their dependencies to be installed.

    * numpy

    * scipy
    
    * sklearn
    
    * gpY, versions >= 0.6.0
    
    * matplotlib
    
    .. note:: 

        For apsis versions newer than December 2014 older gpY versions will no longer work. It has been developed and tested to work with gpY version 0.6.0.


**Operating Systems**

    * developed on Ubuntu 14.04. Tested on Mac OS X Yosemite.
    * most unix based operating systems for which the dependencies listed above are available should work.
    
    * no support for non-unix systems right now.
 
Installation using PIP
======================

apsis can easiest be installed using PIP by just executing ::

    $ pip install apsis --pre

If the installation fails then you most lieky do not have the appropriate non-python requirements for one of the packages above installed. See the corresponding section under Manual Installation.
 
Manual Installation
===================

Installing Non-Python Requirements by Operating System
------------------------------------------------------
    
Installing Non-Python Prerequisites on Debian/Ubuntu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The compilation of matplotlib and scipy have several non-python dependencies such as C and fortran compilers or linear algebra libraries. Also you should install ``pip`` to install the newest versions of the python dependencies.

Tested on Ubuntu 14.04 the following command should give you what you need. If you run on another OS please check out the documentation of the listed prerequesites above for how to install them. ::

    $ sudo apt-get install git build-essential python-pip gfortran libopenblas-dev liblapack-dev libfreetype6-dev libpng12-dev python-dev
    
*Optional* In order to be able to use the Markov Chain Monte Carlo sampling for integrating over GP Hyperparameters you need to install a HDFS distribution on your system. For Ubuntu 14.04 the 
following will do the trick. ::

    $ sudo apt-get install libhdf5-serial-dev

    
Installing Non-Python Prerequesites on Mac OS X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to update your python version to a later version than the one distributed with your OS.

Installation works easy when using homebrew package manager, please see the homebrew page for how to install it.  

  http://brew.sh/

When homebrew is installed follow these instructions.
  
1. Install another and up to date Python distribution.

    $ brew install python
    $ brew linkapps python
    
2. Install pip

    $ brew install pip
    $ brew linkapps pip
        
    
Installing Python Prerequisites with PIP
------------------------------------------------------

1. Make sure you have ``pip`` and the non-python prerequisites for the libraries listed above installed on your system

2. Install numpy. ::

    $ pip install --upgrade numpy

2. Install scikit learn. ::

    $ pip install --upgrade scikit-learn

3. Install matplotlib. ::
    
    $ pip install --upgrade matplotlib

4. Install gpY. It will also install the required scipy version for you. ::

    $ pip install --upgrade gpy==0.6.0
    
5. *Optional* If you want to use MCMC sampling for the hyperparameters of the acquisition functions in bayesian optimization then you need to install pymc. The installation is easy and you only need to clone the git repository and run the setup script. See the following link for details. ::

    https://github.com/ebilionis/py-mcmc

Installing and Running Apsis
-----------------------------

Apsis doesn't have an installation routine yet. To be ready to use you need to

1. Pull the code repository ::

    $ git clone https://github.com/FrederikDiehl/apsis.git
    
2. Set the PYTHONPATH environment variable to include th apsis folder ::

    $ export PYTHONPATH=[WHEREVER]/apsis/code/apsis

Finally run the test suite to see if everything works alright::

    $ cd apsis/code/apsis
    $ nosetests

Which should print something like ::

    $ nosetests
    .
    ----------------------------------------------------------------------
    Ran XX tests in YYs
    
    OK
    



