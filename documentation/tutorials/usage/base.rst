First steps
***********

Once we have installed apsis, we can continue by doing our first project.

Generally speaking, apsis is divided into two parts: The server and clients. The client package can be installed independent of the server, and has only one dependency, making it easy to deploy on any system.

Our most important interface, and the only interface you will generally see in apsis is therefore the :class:`Connections <apsis_client.apsis_connection.Connection>` class in apsis_client.apsis_connection.

It communicates to an apsis server somewhere in the network (or on your local computer). The default port is set to 5000. Starting the server is done via the REST_start_script, located in apsis.webservice.REST_start_script, either by running it from the terminal or by starting it from a python interpreter or script, here running on port 5000::

    from apsis.webservice.REST_start_script import start_rest
    start_rest(port=5000)
    
Next, we can start a connection to that in another python process::

    from apsis_client.apsis_connection import Connection
    conn = Connection(server_address="http://localhost:5000")

Now, let's talk about :class:`Experiments <apsis.models.experiment.Experiment>`. Experiments are one of the building blocks of apsis.
Each Experiment represents a series of trials, each with a different parameter configuration. These trials are called :class:`Candidates <apsis.models.candidate.Candidate>`. Each Candidate stores the parameter configuration used in the corresponding trial and - once evaluated - the evaluation's result.

An Experiment, aside from the Candidates, also stores some details on the experimental setup. It stores the experiment name, whether the experiment's goal is to minimize or maximize the result and the parameter definitions. The latter is probably the most important part of an experiment. It defines whether a parameter is nominal, ordinal or numeric.

A complete list of parameter definitions can be found :class:`here <apsis.models.parameter_definition>`, but the two most useful are the :class:`MinMaxNumericParamDef <apsis.models.parameter_definition.MinMaxNumericParamDef>` and the :class:`NominalParamDef <apsis.models.parameter_definition.NominalParamDef>`. The first one represents a numeric parameter whose prior is uniformly distributed between a minimum and a maximum value, while the second is just an unordered list of nominal values.

Initializing an experiment is done via the :func:`init_experiment  <apsis_client.apsis_connection.Connection.init_experiment>` function of the Conncetion object, which takes several parameters for which you can find more specific explanations in the function doc. The important parameter is `param_defs`. `param_defs` is a dictionary used to construct the ParamDefs mentioned above. To keep the client as slim as possible, it is created from dictionaries, whose `"type"` field is the name of the ParamDef class. For example, creating a simple experiment called test_experiment with two parameters x and y between [-5, 10] and [0, 15] can be done like this::

    
    param_defs = {
        "x": {"type": "MinMaxNumericParamDef", "lower_bound": -5, "upper_bound": 10},
        "y": {"type": "MinMaxNumericParamDef", "lower_bound": 0, "upper_bound": 15},
    }
    exp_id = conn.init_experiment("my_first_exp", "RandomSearch", param_defs,
                         minimization=True)

The returned exp_id can be used to refer to that experiment in any other functions.

Now that we have defined an experiment, we want to begin optimizing! For this, we first want to get a candidate.::

    cand = conn.get_next_candidate(exp_id)
    
cand now is a dictionary, whose "param" field contains one entry per parameter, whose key is the parameter name, and whose value is the value of the parameter. This is basically the same format as the kwargs for scikit-learn, for example. Our candidate may now look like this (yours will look slightly different, of course)::

    >>> print(cand)
    {
    u'cost': None, 
    u'params': {
        u'y': 4.600118314143332, 
        u'x': -4.334962029067592}, 
    u'id': u'1a75ab29091b4e17ba1fc59c3da8caf0', 
    u'worker_information': None, 
    u'result': None}

    
For our optimization, we'll use a very simple function, which is just a sine with a linear function added::

    import math
    def f(params):
        x = params["x"]
        y = params["y"]
        return math.cos(x) + x/4 + math.sin(y) -x*y
        
    result = f(cand["params"])

To update apsis with the new result, we can simply change the dictionary and return it via the :func:`update <apsis_client.apsis_connection.Connection.update>` function. We can also change the notes, or the worker_information. The latter could be used to refer to a directory for additional information, for example.::

    cand["result"] = result
    conn.update(exp_id, cand, "finished")
    
And we're done, and have evaluated a single candidate.

Once you've done that a few times, you probably want to get your best result. The get_best_candidate function does so, returning the candidate in the same format as above::

    best_cand = conn.get_best_candidate(exp_id)


TODO DONE UNITL HERE.

But of course, we want to see how we performed over time! For this, the PrettyLabAssistant has the ability to plot experiment results over time, and to compare them. Currently, we just need one, though: ::

    assistant.plot_result_per_step(['tutorial_experiment'])
    
My plot looks like this:

.. image:: ./pictures/base_result_per_step.png
   :width: 50%

On the y-axis is the step, on the x axis the result. The line represents the best result found for each step, while the dots are the hypothesis tested in that step. Since the standard values for BayesianOptimization define a ten-step random search, we can see the following: First, we test ten points at random. Beginning with the first step where bayesian optimization begins, we find a very good solution, which is then improved in step 11. The following steps find only slight improvements to this.

That's it! We have optimized our first problem. Further tutorials will follow.