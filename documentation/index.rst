.. apsis documentation master file, created by
   sphinx-quickstart on Fri Jan 23 11:11:04 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../diagrams/apsis_logo.png

Welcome to apsis's documentation!
=================================



A toolkit for hyperparameter optimization for machine learning algorithms.

Our goal is to provide a flexible, simple and scaleable approach - parallel, on clusters and/or on your own machine. Check out our usage tutorials to get started, or the design pages to understand how apsis works.


Contents
--------

.. toctree::
   :maxdepth: 2

   ./tutorials/usage/usage
   ./tutorials/installation
   design
   modules
   evaluation

Example
-------
The following is an example on how to do a simple evaluation routine in apsis (after starting a server)::

    conn = Connection(server_address)

    param_defs = {
        "x": {"type": "MinMaxNumericParamDef", "lower_bound": -5, "upper_bound": 10},
        "y": {"type": "MinMaxNumericParamDef", "lower_bound": 0, "upper_bound": 15},
    }
    
    exp_id = conn.init_experiment(name="first_experiment", optimizer="BayOpt", param_defs=param_defs, minimization=True)
    
    for i in range(10):
        to_eval = conn.get_next_candidate(exp_id)
        result = branin_func(to_eval["params"]["x"], to_eval["params"]["y"])
        to_eval["result"] = result
        conn.update(exp_id, to_eval, "finished")
    print(conn.get_best_candidate(exp_id))

Check out our usage tutorials for more information.
    
        
Project State
-------------

We are currently in beta state. Most of the structure has been implemented, as have been RandomSearch and an initial Bayesian Optimization.

Scientific Project Description
------------------------------

If you want to learn more on the project and are interested in the theoretical background on hyperparameter optimization used in apsis you may want to check out the `scientific project documentation <https://github.com/FrederikDiehl/apsis/raw/master/paper.pdf>`_. This is currently still referencing v0.1.

Furthermore, a presentation slide deck is available at `slideshare <http://www.slideshare.net/andi1400/apsis-automatic-hyperparameter-optimization-framework-for-machine-learning>`_. This presentation is currently still referencing v0.1.

License
-------

The project is licensed under the MIT license, see the `License file <https://github.com/FrederikDiehl/apsis/blob/master/License.txt>`_ on github.

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

