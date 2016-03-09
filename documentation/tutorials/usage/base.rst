First steps
***********

Once we have installed apsis, we can continue by doing our first project.

Generally speaking, apsis is divided into two parts: The server and clients. The client package can be installed independent of the server, and has only one dependency, making it easy to deploy on any system.

Our most important interface, and the only interface you will generally see in apsis, is therefore the :class:`Connection <apsis_client.apsis_connection.Connection>` class in apsis_client.apsis_connection.

Each client communicates (via a REST API) with the server. The server can be anywhere on a network, or on your local computer.

The general use is therefore to first, start the server and then start one or several workers which communicate with the server to get new parameters. Starting the server is done via the REST_start_script, located in apsis.webservice.REST_start_script, either by running it from the terminal or by starting it from a python interpreter or script, here running on port 5000::

    from apsis.webservice.REST_start_script import start_rest
    start_rest(port=5000)
    
Next, we can start a connection to that in another python process::

    from apsis_client.apsis_connection import Connection
    conn = Connection(server_address="http://localhost:5000")

Now, let's talk about :class:`Experiments <apsis.models.experiment.Experiment>`. Experiments are one of the building blocks of apsis.
Each Experiment represents a series of trials, each with a different parameter configuration. These trials are called :class:`Candidates <apsis.models.candidate.Candidate>`. Each Candidate stores the parameter configuration used in the corresponding trial and - once evaluated - the evaluation's result. It can also store any information you need to for its evaluation, and is able to store a cost of the evaluation.

An Experiment, aside from the Candidates, also stores some details on the experimental setup. It stores the experiment name, whether the experiment's goal is to minimize or maximize the result and the parameter definitions. The latter is probably the most important part of an experiment. It defines whether a parameter is nominal, ordinal, numeric or any of the subclasses of these.

A complete list of parameter definitions can be found :class:`here <apsis.models.parameter_definition>`, but the two most useful are the :class:`MinMaxNumericParamDef <apsis.models.parameter_definition.MinMaxNumericParamDef>` and the :class:`NominalParamDef <apsis.models.parameter_definition.NominalParamDef>`. The first one represents a numeric parameter whose prior is uniformly distributed between a minimum and a maximum value, while the second is just an unordered list of nominal values.

Initializing an experiment is done via the :func:`init_experiment  <apsis_client.apsis_connection.Connection.init_experiment>` function of the Conncetion object, which takes several parameters for which you can find more specific explanations in the function doc. The important parameter is `param_defs`. `param_defs` is a dictionary used to construct the ParamDefs mentioned above. To keep the client as slim as possible, it is created from dictionaries, whose `"type"` field is the name of the ParamDef class. For example, creating a simple experiment called test_experiment with two parameters x and y between [-5, 10] and [0, 15] can be done like this::

    
    param_defs = {
        "x": {"type": "MinMaxNumericParamDef", "lower_bound": -5, "upper_bound": 10},
        "y": {"type": "MinMaxNumericParamDef", "lower_bound": 0, "upper_bound": 15},
    }
    exp_id = conn.init_experiment("my_first_exp", "RandomSearch", param_defs,
                         minimization=True)

The returned exp_id can be used to refer to that experiment in any other functions. It is important that you store it, since it's used to refer to to the experiment, and used in almost every other function.

Now that we have defined an experiment, we want to begin optimizing! For this, we first want to get a candidate.::

    cand = conn.get_next_candidate(exp_id)
    
cand is a dictionary representing one such Candidate. Let's inspect it::

    >>> print(cand)
    {
    u'cost': None, 
    u'params': {
        u'y': 4.600118314143332, 
        u'x': -4.334962029067592}, 
    u'id': u'1a75ab29091b4e17ba1fc59c3da8caf0', 
    u'worker_information': None, 
    u'result': None}

These fields are as follows:
* `cost`: The cost of the single evaluation. This can be wallclock time, or number of operations, etc. This will later be used for expected_improvement_per_cost. This has not been implemented yet, but you can use it for statistics etc.
* `params`: A dictionary closely mirroring the `param_defs` dictionary as defined above. It contains one entry per parameter defined, with the key being its name and the value being its proposed value. Note that this format allows you to use a sklearn-like initialization, see the example below.
* `cand_id`: The id of the candidate. This id is unique, and allows identification of the candidate.
* `worker_information`: A field you can fill with arbitrary information. You can use it, for example, to refer to a path where model information is stored. apsis will never modify this value. It is probably useful to use a dictionary for this.
* `result`: The result of the evaluation. It is set to `None` initially. You have to set it before returning.

    
For our optimization, we'll use a very simple function, which is just a sine with a linear function added::

    import math
    def f(x, y):
        return math.cos(x) + x/4 + math.sin(y) -x*y
        
    result = f(**cand["params"])

As you can see, this uses the kwarg feature of python to simplify the code. This feature assures that your existing machine learning code should be easy to integrate.

To update apsis with the new result, we can simply change the dictionary and return it via the :func:`update <apsis_client.apsis_connection.Connection.update>` function. We can also change the notes, or the worker_information.::

    cand["result"] = result
    conn.update(exp_id, cand, "finished")
    
And we're done, and have evaluated a single candidate. In a loop, this looks like this::
    for i in range(10):
        cand = conn.get_next_candidate(exp_id)
        result = f(**cand["params"])
        cand["result"] = result
        conn.update(exp_id, cand, "finished")
        

This loop is all that has to run on your worker instances.

Once you've evaluated a few candidates, you probably want to get your best result. The get_best_candidate function does so, returning the candidate in the same format as above::

    best_cand = conn.get_best_candidate(exp_id)


But of course, we want to have an ability to inspect our results! For this, there's a web interface. Open `localhost:5000` in your browser. You'll be able to select the experiment you want to inspect (in this case, there's only one). Clicking on the link reveals the overview page.

This overview page first tells you about the experiment itself (the id, the parameter definitions etc), then offers you a plot of the current state, and shows you the currently best candidate and every candidate that has been evaluated, is currently being evaluated or has been generated but not evaluated.

The graph is fairly simple: The result is plotted on the y axis, the steps on the x axis. Each point represents the result of one of the experiments in order of their update. The line represents the best result for each step.

Some evaluations are not shown - by default, the plot only encompasses the best half. The reasoning for this is that most users are only interested in the best points and adding bad points would make the discernation of high-quality points more difficult. The worse points are represented by black arrows at the top of the plot. These are bigger the better the result was (that is, the closer to the cutoff), and smaller the further away.

That's it! You have optimized your first problem! How about reading about Bayesian optimization using apsis? .. TODO! Add links.
