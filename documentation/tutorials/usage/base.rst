First steps
***********

Once we have installed apsis, we can continue by doing our first project.

Our most important interface to apsis is the :class:`PrettyLabAssistant <apsis.assistants.lab_assistant.PrettyLabAssistant>` or - if we don't need plots, storage and so on - the :class:`BasicLabAssistant <apsis.assistants.lab_assistant.BasicLabAssistant>`.

Firstly, let's talk about :class:`Experiments <apsis.models.experiment.Experiment>`. Experiments are one of the building blocks of apsis.
Each Experiment represents a series of trials, each with a different parameter configuration. These trials are called :class:`Candidates <apsis.models.candidate.Candidate>`. Each Candidate stores the parameter configuration used in the corresponding trial and - once evaluated - the evaluation's result.

An Experiment, aside from the Candidates, also stores some details on the experimental setup. It stores the experiment name, whether the experiment's goal is to minimize or maximize the result and the parameter definitions. The latter is probably the most important part of an experiment. It defines whether a parameter is nominal, ordinal or numeric.

A complete list of parameter definitions can be found :class:`here <apsis.models.parameter_definition>`, but the two most useful are the :class:`MinMaxNumericParamDef <apsis.models.parameter_definition.MinMaxNumericParamDef>` and the :class:`NominalParamDef <apsis.models.parameter_definition.NominalParamDef>`. The first one represents a numeric parameter whose prior is uniformly distributed between a minimum and a maximum value, while the second is just an unordered list of nominal values.

Each Experiment has a dictionary of parameter definitions, which have to be designated to define that Experiment. For example, let us try to optimize a one-dimensional function, :math:`f(x) = cos(x) + x/4` for x between 0 and 10::

    import math
    def f(x):
        return math.cos(x) + x/4
        
As said above, we are now defining our parameter space. Our only parameter is a numeric parameter between 0 and 10, called x.::
    
    from apsis.models.parameter_definition import MinMaxNumericParamDef
    param_defs = {
        'x': MinMaxNumericParamDef(0, 10)
    }

Now, let's initialize the LabAssistant and the first experiment::

    from apsis.assistants.lab_assistant import PrettyLabAssistant
    assistant = PrettyLabAssistant()
    assistant.init_experiment("tutorial_experiment", "BayOpt", param_defs, minimization=True)   
    
As you can see, we have first initialized the LabAssistant, then the first experiment. The experiment is called tutorial_experiment (each name must be unique). It uses the BayOpt optimizer, is defined by the param_defs we have set above and the goal is one of minimization. We might also give the experiment's optimizer further parameters, but don't do so in this case.

Now, there are two main functionalities of the LabAssistant we usually use: getting the next candidate to try, and returning its result. First, our first proposal::

    candidate = assistant.get_next_candidate("tutorial_experiment")
    
As usual, the first argument specifies which experiment we want to get the next candidate from. There are two important fields in such a Candidate: params and result. We use the first one to set our next evaluation, and the second one to report our evaluation.::

    x = candidate.params['x']
    candidate.result = f(x)
    assistant.update("tutorial_experiment", candidate)

We can continue doing so until we have reached a break criterium, for example a certain number of steps or a sufficiently good result::
    
    for i in range(30):
        candidate = assistant.get_next_candidate("tutorial_experiment")
        x = candidate.params['x']
        candidate.result = f(x)
        assistant.update("tutorial_experiment", candidate)
        
Afterwards, we probably want to get the best result and the parameters we'd used to get there. This is quite simple: ::

    best_cand = assistant.get_best_candidate("tutorial_experiment")
    
This gives us the best candidate we have found. We can then check result and params.::

    print("Best result: " + str(best_cand.result))
    print("with parameters: " + str(best_cand.params))
    
In my case, the result was -0.245, with an x of 2.87. Yours will vary depending on the randomization.

But of course, we want to see how we performed over time! For this, the PrettyLabAssistant has the ability to plot experiment results over time, and to compare them. Currently, we just need one, though: ::

    assistant.plot_result_per_step(['tutorial_experiment'])
    
My plot looks like this:

.. image:: ./pictures/base_result_per_step.png
   :width: 50%

On the y-axis is the step, on the x axis the result. The line represents the best result found for each step, while the dots are the hypothesis tested in that step. Since the standard values for BayesianOptimization define a ten-step random search, we can see the following: First, we test ten points at random. Beginning with the first step where bayesian optimization begins, we find a very good solution, which is then improved in step 11. The following steps find only slight improvements to this.

That's it! We have optimized our first problem. Further tutorials will follow.