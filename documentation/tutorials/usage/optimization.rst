Bayesian Optimization with apsis - Advanced Tutorial
*********************************

apsis implements a technique called Bayesian Optimization for optimizing hyperparameters.

There are several reasons why hyperparameter optimization is difficult: Your machine learning algorithm usually takes a long time to run, punishing an excessive number of evaluations. The fitness landscape usually has a very high dimensionality. It is impossible (or very difficult) to compute gradients of the hyperparameters with respect to the final performance. And, lastly, there's little to no previous knowledge for a truly novel ML algorithm.

Bayesian Optimization uses Gaussian processes to optimize hyperparameters. Intuitively, a point in the fitness landscape (that is, a certain hyperparameter configuration for your ML algorithm) should tell you something about points which are closeby - you expect the weather in London to tell you more about the weather in Oxford than the weather in Sidney. By leveraging this locality through Gaussian processes, we are able to predict the performance of any hyperparameter configuration. One nice aspect of using Gaussian Processes is that we get both a predicted mean and a predicted variance for each point on our fitness landscape - even those we haven't seen yet. In this way, our Gaussian functions acts as a surrogate function to our real fitness landscape.

We then try to choose points in a way that both tells us more about our fitness landscape (exploration) and minimizes or maximizes our performance (exploitation). Choosing the correct point is done by maximizing the so-called acquisition function, which uses the mean and variance at each point to compute its score. One example computes the expected improvement compared to the current best result. This acquisition function is far, far cheaper to evaluate than our machine learning algorithm and usually differentiable. We therefore can easily use the usual optimization methods like LBFSG.

To sum up: We use Gaussian processes as a surrogate function to model our expensive function, then optimize on this cheap surrogate function. Whether we want to explore or exploit is defined by the acquisition function.

A comprehensive introduction to the field of hyperparameter optimization for machine learning algorithms can be found in 

.. [1] James Bergstra, Rémy Bardenet, Yoshua Bengio, and Balazs Kegl. `Algorithms for hyper-parameter optimization. <http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_ In NIPS’2011, 2011.

It also includes a very short introduction to Bayesian Optimization. 

For a more in-depth explanation of Bayesian Optimization, the following tutorial is a must-read:

.. [2] Eric Brochu, Vlad M. Cora, and Nando de Freitas. `A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning. <http://arxiv.org/abs/1012.2599>`_ IEEE Transactions on Reliability, 2010. 

Finally the following paper provides a comprehensive introduction in the usage of Bayesian Optimization for hyperparameter optimization. It includes some tricks which we use in apsis.

.. [3] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. `Practical bayesian optimization of machine learning algorithms. <http://arxiv.org/pdf/1206.2944.pdf>`_ In NIPS, pages 2960–2968, 2012.


The remainder of this tutorial gives an overview on how to switch around between several possibilities implemented in apsis, including how to switch acquisition functions, kernels or the way acquisition functions are optimized.


Choosing Acquisition Functions
===============================

So far apsis contains two acquisition functions, the :class:`ProbabilityOfImprovement <apsis.optimizers.bayesian.acquisition_functions.ProbabilityOfImprovement>` function, as well as :class:`ExpectedImprovement <apsis.optimizers.bayesian.acquisition_functions.ExpectedImprovement>`. You can easily provide your own acquisition functions by extending the :class:`AcquisitionFunction <apsis.optimizers.bayesian.acquisition_functions.AcquisitionFunction>` class. 
Probability Of Improvement is mainly included for the sake of completeness - the Expected Improvement function is the one most commonly used in Bayesian Optimization. It is said to have a good balance between exploration of unknown regions and exploitation of well-known but promising regions. 

You can choose the acquisition function - or, indeed, most of the optimizer parameters - by passing it to ``init_experiment`` using the ``optimizer_arguments`` parameter.
As an example, you can use::

    optimizer_params = {
        "acquisition": "ExpectedImprovement"
    }
    
    exp_id = conn.init_experiment(name, optimizer, 
                              param_defs, optimizer_arguments=optimizer_params,
                              minimization=minimization)

Furthermore an acquisition function can receive hyperparameters, e.g. for telling apsis how to optimize this function. These hyperparameters are specific to the acquisition function. :class:`ExpectedImprovement <apsis.optimizers.bayesian.acquisition_functions.ExpectedImprovement>` for example can be told to use another optimization method. More on this in the section on Acquisition Optimization.::

    optimizer_params = {
        "acquisition": "ExpectedImprovement",
        "acquisition_hyperparams": {
            "max_searcher": "random"
        }
    }
    exp_id = conn.init_experiment(name, optimizer, 
                              param_defs, optimizer_arguments=optimizer_params,
                              minimization=minimization)

  
Choosing Kernels
=================

Another central point of tweaking the performance of bayesian optimization is the kernel. apsis supports the Matern 5-2 and the RBF kernel. The first one is the standard choice. Both kernels use the GPy package. Choosing your kernel works similarly to choosing your acquisition function.

You can either specify the kernel as one of those two strings ``["matern52", "rbf"]`` or supply a class inheriting from the GPy.kern.Kern class.::

    optimizer_params = {
        "kernel": "Matern52",
        "acquisition": "ExpectedImprovement",
        "acquisition_hyperparams": {
            "max_searcher": "random"
        }
    }
    exp_id = conn.init_experiment(name, optimizer, 
                              param_defs, optimizer_arguments=optimizer_params,
                              minimization=minimization)

By default the Matern 5-2 kernel with ARD will be used.

Minimizing or Maximizing your Objective Function
================================================

By default apsis assumes you want to minimize your objective function, e.g. that it represents the error of your machine learning algorithm. However, apsis can easily be switched to assume maximization by specifying so when initializing an experiment::

    exp_id = conn.init_experiment(name, optimizer, 
                              param_defs, minimization=False)
      
Expected Improvement
====================

The Expected Improvement function implemented in apsis has a couple of places that can be tuned.
    
The parameter ``exploitation_exploration_tradeoff`` has been suggested in the Brochu paper (originally from Lisotte 2008). It is a positive number, and has been suggested to be set to ``0.01``.

By default, Expected Improvement uses LBFGSB to optimize the function. It is suggested you keep this.

    
    
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


############################
TODO: Done up to this point.
############################




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
