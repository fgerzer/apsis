__author__ = 'Frederik Diehl'

from apsis.models.experiment import Experiment
from apsis.models.candidate import Candidate
from apsis.utilities.optimizer_utils import check_optimizer
from apsis.utilities.file_utils import ensure_directory_exists
import datetime
import os
import time
from apsis.utilities.logging_utils import get_logger
from apsis.utilities.plot_utils import plot_lists, write_plot_to_file
import matplotlib.pyplot as plt

AVAILABLE_STATUS = ["finished", "pausing", "working"]


class ExperimentAssistant(object):
    """
    This class represents an assistant assisting with a single experiment.

    It stores an experiment (a data structure which stores the parameter
    definition, the candidates evaluated and current and whether it's to be
    minimized) and an optimizer for optimizing the experiment.
    It also contains functions for writing out results and for plotting.

    Parameters
    ----------
    _optimizer : Optimizer
        The Optimizer used to find new points for the experiment. It has to be
        an apsis.optimizers.optimizer.Optimizer instance.
    _optimizer_arguments : dict
        Dictionary of the arguments for optimizer.
    _experiment : Experiment
        The experiment storing the evaluated points and parameter definition.
    _csv_write_frequency : int, strictly positive.
        This sets the frequency with which the csv file is written. If set to
        1, it writes every step. If set to 2, every second and so on. Note that
        it still writes out every step eventually.
    _csv_steps_written : int
        Stores the number of steps already stored to csv file.
    _experiment_directory_base : string
        The folder to which the csv intermediary results and the plots will be
        written. Default is dependant on the OS. On windows, it is set to
        ./APSIS_WRITING/<exp_id>. On Linux, it is set to
        /tmp/APSIS_WRITING/<exp_id>.
    _logger : logger
        The logger instance for this class.

    """

    _optimizer = None
    _optimizer_arguments = None
    _experiment = None

    _csv_write_frequency = None
    _csv_steps_written = 0
    _experiment_directory_base = None
    _write_directory_base = None

    _logger = None

    def __init__(self, optimizer_class, optimizer_arguments=None,
                 write_directory_base=None, experiment_directory=None,
                 csv_write_frequency=1):
        """
        Initializes this experiment assistant.

        Note that calling this function does not yet create an experiment, for
        that, use init_experiment. If there is an already existing experiment,
        you can just set self._experiment.

        Parameters
        ----------
        optimizer_class : subclass of Optimizer
            The class of the optimizer, used to initialize it.
        optimizer_arguments : dict, optional
            The dictionary of optimizer arguments. If None, default values will
            be used.
        experiment_directory_base : string, optional
            The folder to which the csv intermediary results and the plots will
            be written. Default is <write_directory_base>/exp_id.
        write_directory_base : string, optional
            The base directory. In the default case, this is dependant on the
            OS. On windows, it is set to ./APSIS_WRITING/. On Linux,
            it is set to /tmp/APSIS_WRITING/. If an
            experiment_directory has been given, this will be ignored.
        csv_write_frequency : int, optional
            This sets the frequency with which the csv file is written. If set
            to 1 (the default), it writes every step. If set to 2, every second
            and so on. Note that it still writes out every step eventually.
        """
        self._logger = get_logger(self)
        self._logger.info("Initializing experiment assistant.")
        self._csv_write_frequency = csv_write_frequency
        self._optimizer = optimizer_class
        self._optimizer_arguments = optimizer_arguments
        if self._csv_write_frequency != 0:
            if experiment_directory is not None:
                self._experiment_directory_base = experiment_directory
                ensure_directory_exists(self._experiment_directory_base)
            else:
                if write_directory_base is None:
                    if os.name == "nt":
                        self._write_directory_base = \
                            os.path.relpath("APSIS_WRITING")
                    else:
                        self._write_directory_base = "/tmp/APSIS_WRITING"
                else:
                    self._write_directory_base = write_directory_base
        self._logger.info("Experiment assistant for successfully "
                         "initialized.")

    def init_experiment(self, name, param_defs, exp_id=None, notes=None,
                        minimization=True):
        """
        If not existing, initializes the _experiment.

        Parameters
        ----------
        name : string
            The name of the experiment.
        param_defs : dict of ParamDefs
            A dictionary with string keys, with each string being the name of
            a parameter, and the value being its parameter definition.
        exp_id : string, optional
            The id of this experiment. Can be set manually, or (if None, which
            is the default) will be generated by Experiment, which, at the
            moment, is uuid4.
        notes : string or None, optional
            The notes for the experiment. Can be any string (or, actually,
            jsonable object). Used for human-readable notes. Can be None, the
            default.
        minimization : bool, optional
            Whether this experiment's goal is to minimize the score function
            (the default assumption).

        Raises
        ------
        ValueError
            Iff creating a new experiment although this instance already has
            one set.
        """
        if self._experiment is None:
            experiment = Experiment(name, param_defs, exp_id, notes,
                                    minimization)
            self._experiment = experiment
            self._init_optimizer()
            if self._experiment_directory_base is None:
                self._create_experiment_directory()
        else:
            raise ValueError("Created a new experiment with one already "
                             "existing.")

    def set_experiment(self, experiment):
        """
        Sets the experiment property of this instance.

        Parameters
        ----------
        experiment : Experiment
            The experiment with which to set this.


        Raises
        ------
        ValueError
            Iff creating a new experiment although this instance already has
            one set.
        """
        if self._experiment is None:
            self._experiment = experiment
            self._init_optimizer()
            if self._experiment_directory_base is None:
                self._create_experiment_directory()
        else:
            raise ValueError("Set a new experiment with one already "
                             "existing.")


    def _init_optimizer(self):
        """
        Initializes the optimizer if it does not exist.
        """
        self._optimizer= check_optimizer(self._optimizer, self._experiment,
            optimizer_arguments=self._optimizer_arguments)



    def get_next_candidate(self):
        """
        Returns the Candidate next to evaluate.

        Internally, it first tries to return the most recent pending candidate
        of this experiment. If there is none, it generates one from optimizer.

        Returns
        -------
        next_candidate : Candidate or None
            The Candidate object that should be evaluated next. May be None,
            which is equivalent to no candidate generated.
        """

        self._logger.info("Returning next candidate.")
        if not self._experiment.candidates_pending:
            candidates = self._optimizer.get_next_candidates(num_candidates=1)
            if candidates is None:
                return None
            if len(candidates) > 0:
                self._experiment.add_working(candidates[0])
                return candidates[0]
            return None
        else:
            cand = self._experiment.candidates_pending.pop()
            self._experiment.add_working(cand)
            return cand

    def get_experiment_as_dict(self):
        """
        Returns the dictionary describing this EAss' experiment.

        Signature is equivalent to Experiment.to_dict()

        Returns
        -------
            exp_dict : dict
                The experiment dictionary.
        """
        return self._experiment.to_dict()

    def update(self, candidate, status="finished"):
        """
        Updates the experiment_assistant with the status of an experiment
        evaluation.

        Parameters
        ----------
        candidate : Candidate
            The Candidate object whose status is updated.
        status : {"finished", "pausing", "working"}
            A string defining the status change. Can be one of the following:
            - finished: The Candidate is now finished.
            - pausing: The evaluation of Candidate has been paused and can be
                resumed by another worker.
            - working: The Candidate is now being worked on by a worker.

        """
        if status not in AVAILABLE_STATUS:
            message = ("status not in %s but %s."
                             %(str(AVAILABLE_STATUS), str(status)))
            self._logger.error(message)
            raise ValueError(message)

        if not isinstance(candidate, Candidate):
            message = ("candidate %s not a Candidate instance."
                             %str(candidate))
            self._logger.error(message)
            raise ValueError(message)

        self._logger.info("Got new %s of candidate %s with parameters %s"
                         " and result %s" % (status, candidate,
                                            candidate.params,
                                            candidate.result))

        if status == "finished":
            self._experiment.add_finished(candidate)

            # invoke the writing to files
            step = len(self._experiment.candidates_finished)
            if self._csv_write_frequency != 0 and step != 0 \
                    and step % self._csv_write_frequency == 0:
                self._append_to_detailed_csv()
                self.write_plots()
            # And we rebuild the new optimizer.
            self._optimizer.update(self._experiment)

        elif status == "pausing":
            self._experiment.add_pausing(candidate)
        elif status == "working":
            self._experiment.add_working(candidate)

    def get_best_candidate(self):
        """
        Returns the best candidate to date.

        Returns
        -------
        best_candidate : candidate or None
            Returns a candidate if there is a best one (which corresponds to
            at least one candidate evaluated) or None if none exists.
        """
        return self._experiment.best_candidate

    def _create_experiment_directory(self):
        """
        Generates an experiment directory from the base write directory.
        """
        global_start_date = time.time()
        date_name = datetime.datetime.utcfromtimestamp(
                global_start_date).strftime("%Y-%m-%d_%H:%M:%S")
        self._experiment_directory_base = os.path.join(
                                    self._write_directory_base,
                                    self._experiment.exp_id)
        ensure_directory_exists(self._experiment_directory_base)

    def _best_result_per_step_dicts(self, color="b", plot_up_to=None,
                                    cutoff_percentage=1.):
        """
        Returns a dict to use with plot_utils.
        Parameters
        ----------
        color="b": string
            A pyplot-color representing string. Both plots will have that
            color.
        Returns
        -------
        dicts: list of dicts
            Two dicts, one for step_eval, one for step_best, and their
            corresponding definitions.
        """
        x, step_eval, step_best = self._best_result_per_step_data(plot_up_to=
                                                                  plot_up_to)

        step_eval_dict = {
            "x": x,
            "y": step_eval,
            "type": "scatter",
            "label": "%s" % (str(self._experiment.name)),
            "color": color,
            "cutoff_percent": cutoff_percentage
        }

        step_best_dict = {
            "x": x,
            "y": step_best,
            "type": "line",
            "color": color,
        }

        return [step_eval_dict, step_best_dict]

    def _best_result_per_step_data(self, plot_up_to=None):
        """
        This internal function returns goodness of the results by step.
        This returns an x coordinate, and for each of them a value for the
        currently evaluated result and the best found result.
        Returns
        -------
        x: list of ints
            The results per step. Should usually be [0, ..., maxSteps]
        step_evaluation: list of floats
            The result of the evaluated candidate during the corresponding step
        step_best: list of floats
            The best result that has been found until then.
        """
        x = []
        step_evaluation = []
        step_best = []
        best_candidate = None
        if plot_up_to is None:
            plot_up_to = len(self._experiment.candidates_finished)
        for i, e in enumerate(self._experiment.candidates_finished
                              [:plot_up_to]):
            x.append(i)
            step_evaluation.append(e.result)
            if self._experiment.better_cand(e, best_candidate):
                best_candidate = e
                step_best.append(e.result)

            else:
                step_best.append(best_candidate.result)
        return x, step_evaluation, step_best

    def _append_to_detailed_csv(self):
        """
        Appends the currently, non-written results to the csv summary.

        In the first step, includes the header.
        """
        if len(self._experiment.candidates_finished) <= \
                self._csv_steps_written:
            return

        # create file and header if
        wHeader = False
        if self._csv_steps_written == 0:
            #set use header
            wHeader = True

        csv_string, steps_included = self._experiment.to_csv_results(
                                            wHeader=wHeader,
                                            fromIndex=self._csv_steps_written)

        # write
        filename = os.path.join(self._experiment_directory_base,
                                self._experiment.exp_id + "_results.csv")

        with open(filename, 'a+') as detailed_file:
            detailed_file.write(csv_string)

        self._csv_steps_written += steps_included

    def get_candidates(self):
        """
        Returns the candidates of this experiment in a dict.

        Returns
        -------
        result : dict
            A dictionary of three lists with the keys finished, pending and
            working, with the corresponding candidates.
        """
        result = {"finished": self._experiment.candidates_finished,
                  "pending": self._experiment.candidates_pending,
                  "working": self._experiment.candidates_working}
        return result


    def plot_result_per_step(self, ax=None, color="b",
                             plot_min=None, plot_max=None):
        """
        Returns the plt.figure plotting the results over the steps.

        Parameters
        ----------
        ax : None or matplotlib.Axes, optional
            The ax to update. If None, a new figure will be created.
        color : string, optional
            A string representing a pyplot color.
        plot_min : float, optional
            The smallest value to plot on the y axis.
        plot_max : float, optional
            The biggest value to plot on the y axis.
        Returns
        -------
        ax: plt.Axes
            The Axes containing the results over the steps.
        """


        plots = self._best_result_per_step_dicts(color, cutoff_percentage=0.5)
        if self._experiment.minimization_problem:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'

        plot_options = {
            "legend_loc": legend_loc,
            "x_label": "steps",
            "y_label": "result",
            "title": "Plot of %s result over the steps."
                     % (str(self._experiment.name)),
            "minimizing": self._experiment.minimization_problem
        }
        fig, ax = plot_lists(plots, ax=ax, fig_options=plot_options,
                             plot_min=plot_min, plot_max=plot_max)

        return fig

    def write_plots(self):
        """
        Writes out the plots of this assistant.
        """
        fig = self.plot_result_per_step()
        filename = "result_per_step_%i" \
                   % len(self._experiment.candidates_finished)

        path = self._experiment_directory_base + "/plots"
        ensure_directory_exists(path)
        write_plot_to_file(fig, filename, path)
        write_plot_to_file(fig, "cur_state", self._experiment_directory_base)
        plt.close(fig)

    def set_exit(self):
        """
        Exits this assistant.

        Currently, all that is done is that the optimizer is exited.
        """
        self._optimizer.exit()
