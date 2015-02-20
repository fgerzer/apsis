Bayesian Optimization with apsis - Advanced Tutorial
*********************************

apsis implements the technique called Bayesian Optimization for optimizing your hyperparameters. There are several problems involved with optimizing hyperparameters, why it took up to now for automated methods to become available. First, you usually have any assumption on the form and behaviour of the loss function you want to optimize over. Second, you are actually running a surrounding optimization loop around some other sort of optimization loop. This means that in every iteration your optimization for the hyperparameters usually includes another optimization, which is your acutal algorithm - e.g. a machine learning algorithm - for which you optimize the parameters. This makes it necessary to carefully select the next hyperparameter vector to try, since that migh involve to train your machine learning algorithm for a couple of hours or more. 

Hence in Bayesian Optimization a Gaussian Process is trained as a surrogate model to approximate the loss function or whatever measure of your algorithm shall be optimized. A central role here is played by the so called acquisition function that is responsible for interpreting the surrogate model and suggest new hyperparameter vectors to sample from your original algorithm. This function is maximized and the hyperparameter vector maximizing it will be the next one to be tested.

A comprehensive introduction to the field of hyperparameter optimization for machine learning algorithms can be found in 

.. [1] James Bergstra, R ́emy Bardenet, Yoshua Bengio, and Balazs Kegl. Algorithms for hyper-parameter optimization. In NIPS’2011, 2011. `See here for downloading the paper. <http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_

In there you can also find a very short introduction to Bayesian Optimization. For a more clear and in-depth understanding how Bayesian Optimization itself works the following great tutorial is a must reader.

.. [2] Eric Brochu, Vlad M. Cora, and Nando de Freitas. A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning. IEEE Transactions on Reliability, 2010. `See here for downloading the paper. <http://arxiv.org/abs/1012.2599>`_

Finally the following paper provides a comprehensive introduction in the usage of Bayesian Optimization for hyperparameter optimization.

.. [3] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. In NIPS, pages 2960–2968, 2012. `See here for downloading the paper. <http://arxiv.org/pdf/1206.2944.pdf>`_

This tutorial gives an overview on how to switch around between several possibilities implemented in apsis, including how to switch acquisition functions, kernels or the way acquisition functions are optimized.


Choosing Acquisition Functions
===============================

So far apsis contains only two acquisition functions, the :class:`ProbabilityOfImprovement <apsis.optimizers.bayesian.acquisition_functions.ProbabilityOfImprovement>` function, as well as :class:`ExpectedImprovement <apsis.optimizers.bayesian.acquisition_functions.ExpectedImprovement>`. You can easily provide your own acquisition functions by extending to the :class:`AcquisitionFunction <apsis.optimizers.bayesian.acquisition_functions.AcquisitionFunction>` class. While Probability Of Improvement is mainly included for the sake of completeness the Expected Improvement function is the one most commonly used in Bayesian Optimization. It is said to have a good balance between exploration of unknown regions and exploitation of well-known but promising regions. 

To choose which acquisition function to use you can do so using the :class:`LabAssistant <apsis.assistants.lab_assistant.BasicLabAssistant>` or :class:`ExperimentAssistant <apsis.assistants.experiment_assistant.PrettyExperimentAssistant>` interface and passing the acquisition function name in the optimizer_arguments.::

    from apsis.optimizers.bayesian.acquisition_functions import ExpectedImprovement, ProbabilityOfImprovement
    from apsis.assistants.lab_assistant import PrettyLabAssistant
    
    LAss = PrettyLabAssistant()
    LAss.init_experiment("bay_EI", "BayOpt", param_defs, minimization=True, optimizer_arguments={"acquisition": ExpectedImprovement, "initial_random_runs": 5} )
    LAss.init_experiment("bay_POI", "BayOpt", param_defs, minimization=True, optimizer_arguments={"acquisition": ProbabilityOfImprovement, "initial_random_runs": 5} )
    
Furthermore an acquisition function can receive hyperparameters, e.g. for telling apsis how to optimize this function. These hyperparameters are specific to the acquisition function. :class:`ExpectedImprovement <apsis.optimizers.bayesian.acquisition_functions.ExpectedImprovement>` for example can be told to use another optimization method. More on this in the section on Acquisition Optimization.

    LAss.init_experiment("bay_EI_BFGS", "BayOpt", param_defs, minimization=True, optimizer_arguments={"acquisition": ExpectedImprovement, "initial_random_runs": 5, "acquisition_hyperparams":{"optimization": "BFGS"}} )
  
Choosing Kernels
=================

Another central point to tweak your bayesian optimization is the kernel used. apsis supports the Matern 5-2 and the RBF kernel, whereas the first one is selected as standard choice. For both kernels the implementation of the gpY package is used. Choosing your kernel works similar to choosing your acquisition function.

You can either specify the kernel as one of those two strings ["matern52", "rbf"] or supply a class inheriting from the GPy.kern.Kern class.::

    from apsis.assistants.lab_assistant import PrettyLabAssistant
    impoort GPy
    
    LAss = PrettyLabAssistant()
    LAss.init_experiment("bay_RBF", "BayOpt", param_defs, minimization=True, optimizer_arguments={"kernel": "rbf", "initial_random_runs": 5} )
    LAss.init_experiment("bay_Matern52", "BayOpt", param_defs, minimization=True, optimizer_arguments={"kernel": GPy.kern.Matern52, "initial_random_runs": 5} )
  
A kernel can also be given parameters if necessary. For example a frequent parameter to the gpY kernels is if automatic relevance determination (ARD) shall be used or not.::

    LAss.init_experiment("bay_Matern52", "BayOpt", param_defs, minimization=True, optimizer_arguments={"kernel": GPy.kern.Matern52, "kernel_params": {"ARD": True}, "initial_random_runs": 5} )
  
By default the Matern 5-2 kernel with ARD will be used.

Minimizing or Maximizing your Objective Function
================================================

By default apsis assumes you want to minimize your objective function, e.g. that it represents the error of your machine learning algorithm. However, apsis can easily be switched around to be used for maximization when specifying the minimization property of :class:`LabAssistant <apsis.assistants.lab_assistant.BasicLabAssistant>` or :class:`ExperimentAssistant <apsis.assistants.experiment_assistant.PrettyExperimentAssistant>`.::

    from apsis.assistants.lab_assistant import PrettyLabAssistant
    LAss = PrettyLabAssistant()
    LAss.init_experiment("bay_Matern52", "BayOpt", param_defs, minimization=False, optimizer_arguments={"kernel": GPy.kern.Matern52, "initial_random_runs": 5} )
  
Dealing with the GPs Hyperparameter
===================================

In addition to those hyperparameters that are subject of optimization the Gaussian Process used to approximate the underlying model also has hyperparameters. Especially the kernel usually has a relevance parameter influencing the shape of the distribution. This can be one parameter or several depending on if an ARD kernel is used or not. By default apsis uses maximum likelyhood as implemented by the gpY package to optimize these parameters. 

Additionally you can switch to use Hybrid Monte Carlo sampling provided by the gpY package to integrate these parameters out. This will only apply for the GP and kernel hyperparameters, not for those of the acquisition function. To do so simply switch the mcmc parameter to True. ::

    from apsis.assistants.lab_assistant import PrettyLabAssistant
    LAss = PrettyLabAssistant()
    LAss.init_experiment("bay_rand", "BayOpt", param_defs, minimization=True, optimizer_arguments={"initial_random_runs": 5, "mcmc": True})

Note that using the Monte Carlo sampling takes considerably more than time than not. You should consider this option only of you are optimizing a several ours or more running ml algorithm.
    
Expected Improvement
====================

This section describes how Expected Improvement is implemented in apsis. You migh also want to see the `source code. <https://github.com/FrederikDiehl/apsis/blob/master/code/apsis/optimizers/bayesian/acquisition_functions.py>`_.

The Expected Improvement function implemented in apsis has a couple of places that can be tuned

    * maximization method of ExpectedImprovement
    * exploration/exploitation tradeoff
    * minimization or maximization
    
    
Closed Form Computation and Gradient
------------------------------------

Expected Improvement (EI) is generally defined as the expectation value of the improvement, hence being the integral of the improvement times its probability for every possible hyperparameter vector, called :math:`\lambda` here.

.. math::

  u_{\text{EI}}(\lambda| M_{t}) = \underset{-\infty}{\int}^{\infty} \underbrace{max(y^{*} - y, 0)}_{\text{value of improvement}} \text{  }\cdot \underbrace{p_M(y|\lambda)}_\text{probability of improvement}\text{  }dy

:math:`y` represents the GP model's prediction for the value of the objective function if the hyperparameter vector is set to :math:`\lambda` and :math:`y^{*}` marks the best value measured on the true objective function so far. Fortunately there is a closed form of this equation available.

.. math::

  u_{\text{EI}}(\lambda| M_{t}) = \sigma(\lambda) \cdot \left( z(\lambda) \cdot \Phi(\lambda) + \phi(\lambda) \right)

with 

.. math::

  z(\lambda) = \frac{\left( f(\lambda^{*}) - \mu(\lambda)\right)}{\sigma(\lambda)}

In apsis there is an adopted version in use that allows for switching maximization and minimization of the objective function, and adds an additional parameter :math:`\zeta` used to balance the exploitation/exploration tradeoff in EI. :math:`MAX` is assumed to be a binary value of either :math:`0` if the function is being minimized or :math:`1` for maxmimization of the objective function.
  
.. math::

  z(\lambda) = \frac{(-1)^{MAX} \cdot \left( f(\lambda^{*}) - \mu(\lambda) + \zeta\right)}{\sigma(\lambda)}
  
Also the gradient has been derived for EI in order to be able to apply gradient based optimization methods.

.. math::

  \nabla EI(\lambda) &= \frac{\nabla \sigma^{2}(\lambda)}{2\sigma(\lambda)}  - (-1)^{MAX} \cdot \nabla\mu(\lambda) \cdot \Phi(z(\lambda)) -  \nabla \sigma^{2}(\lambda) \cdot \Phi(z(\lambda)) \cdot \frac{z(\lambda)}{2\sigma(\lambda)}

EI Optimization
---------------

No matter if the underlying objective function is to be maximized or minimized EI always has to be maximized since we want to do the maximum possible improvement in every step. 

apsis provides the following possibilities for maximization of EI. The value in ["XX"] denotes the key for activating the respective method.

    * random search ["random"] 
    * Quasi-Newton optimization using the inverse BFGS method. ["BFGS"]
    * Nelder-Mead method ["Nelder-Mead"]
    * Powell method ["Powell"]
    * Conjugate Gradient method ["CG"]
    * inexact/truncated Newton method using Conjugate Gradient to solve the Newton Equation ["Newton-CG"]

For the latter 5 it shall be referred to the `docs of the scipy project <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_ since their implementation is used in apsis. The first one is implemented directly in apsis.

To switch the optimization method simply specify the acquisition hyperparameter optimization when initializing your experiments.::

    from apsis.assistants.lab_assistant import PrettyLabAssistant
    LAss = PrettyLabAssistant()
    LAss.init_experiment("bay_RBF", "BayOpt", param_defs, minimization=True, optimizer_arguments={"acquisition": ExpectedImprovement, "initial_random_runs": 5, "acquisition_hyperparams":{"optimization": "BFGS"}} )

Since the gradient of EI can also be computed in closed form it is desirable to make use of that first order information during optimization. Hence BFGS optimization is set as default method since it generally performs better than the others when gradients are available. For all of the optimization methods above a random search is performed first and the best samples from random search will be used as initializers for the more sophisticated optimization methods. 

To prevent keeping stuck in local extrema too much optimization can use multiple restarts. By default random search uses 1000 iterations.::

    from apsis.assistants.lab_assistant import PrettyLabAssistant
    LAss = PrettyLabAssistant()
    LAss.init_experiment("bay_RBF", "BayOpt", param_defs, minimization=True, optimizer_arguments={"acquisition": ExpectedImprovement, "initial_random_runs": 5, "acquisition_hyperparams":{"optimization_random_steps": 100000}} )

Also the number of function evaluations for random search can be specified as follows. This will have an effect on all optimizations methods you select since in every case a random search is done at first place. By default random search uses 10 random restarts will be done.::

    from apsis.assistants.lab_assistant import PrettyLabAssistant
    LAss = PrettyLabAssistant()
    LAss.init_experiment("bay_RBF", "BayOpt", param_defs, minimization=True, optimizer_arguments={"acquisition": ExpectedImprovement, "initial_random_runs": 5, "acquisition_hyperparams":{"optimization_random_restarts": 10}} )
