Installing Apsis
****************

This guide provides instructions for how to get apsis running on your system. The guide is manily targetted for Ubuntu/Debian users, however as a user of another linux based OS you should easily to be able to follow this guide with the methods used in your distro.



Prerequesites
=============

Apsis requires the following **python frameworks** and their dependencies to be installed.

    * numpy

    * scipy
    
    * sklearn
    
    * gpY, versions <= 0.4.9
    
    * matplotlib
    
    .. note:: 

        The newest gpY verisons will not worky any more. It has been developed tested to work with gpY version 0.4.9. Adoption to the new gpY interface is on the todo list.


**Operating Systems**

    * most unix based operating systems for which the dependencies listed above are available should work.
    
    * no support for non-unix systems right now.
    
Installing Non-Python Prerequesites on Debian/Ubuntu
====================================================

The compilation of matplotlib and scipy have several non-python dependencies such as C and fortran compilers or linear algebra libraries. Also you should install ``pip`` to install the newest versions of the python dependencies.

Tested on Ubuntu 14.04 the following command should give you what you need. If you run on another OS please check out the documentation of the listed prerequesites above for how to install them. ::

    $ apt-get install git python-pip gfortran libopenblas-dev liblapack-dev libfreetype6-dev libpng12-dev

    
Installing Python Prerequesites with PIP
====================================

    1. Make sure you have ``pip`` and the non-python prequesites for the libraries listed above installed on your system

    2. Install scikit learn. It will also install numpy and scipy for you. ::

        $ pip install scikit-learn
    
    3. Install matplotlib. ::
        
        $ pip install matplotlib
    
    4. Install gpY. **Attention** Newer versions of gpY do not work at the moment. ::
    
        $ pip install gpy==0.4.9
        

Installing and Running Apsis
================

Apsis doesn't have an installation routine yet. To be ready to use you need to

    1. Pull the code repository ::
    
        $ git clone https://github.com/FrederikDiehl/apsis.git
        
    2. Set the PYTHONPATH environment variable to include th apsis folder ::

        $ export PYTHONPATH=[WHEREVER]/BayOpt/code/apsis

    3. Set the CSV writing target folder to tell apsis where to store results when using EvaluationFramework. E.g. something like::
    
        $ export APSIS_CSV_TARGET_FOLDER=/tmp/apsiswriting
    
Finally run the test suite to see if everything works alright.::

        $ cd BayOpt/code/apsis
        $ nosetests

Which should print something like ::

        $ nosetests
        .
        ----------------------------------------------------------------------
        Ran 25 tests in 80.655s
        
        OK
    



