__author__ = 'Frederik Diehl'

import requests
import time


class Connection(object):
    """
    This is a (very slim) connection to an apsis server.

    It can be used for a minimum installation of apsis on a worker node,
    for example. It can also be used to start experiments.

    In general, all of the functions defined here take two additional
    parameters:
    blocking : bool, optional
        If True, retries the query until it receives an acceptable answer, at
        most timeout seconds.
        If False, tries the query only once.
        Default is True.
    timeout : float, optional
        The maximum time to retry the connection. If it is <= 0 or None, this
        is interpreted as a an infinitely long wait.
         Default is None.

    Therefore, with the default settings, it is assured that the return of each
    function is a valid value. Otherwise, None may be returned. This means
    that, with the default settings, any program using this class can be
    assured that no extra handling of non-standard answers will be necessary.

    Attributes
    ----------
    server_address : string
        The address (including port) on which the apsis server is reachable.

    repeat_time : float, optional
        The minimum time in seconds between repeat attempts to retry a failed
        request. The real time may be slightly longer.
        Default is 0.1s
    """
    server_address = None
    repeat_time = None

    def __init__(self, server_address, repeat_time=0.1):
        """
        Initializes the apsis connection.

        Parameters
        ----------
        server_address : string
        The address (including port) on which the apsis server is reachable.

        repeat_time : float, optional
            The minimum time in seconds between repeat attempts to retry a failed
            request. The real time may be slightly longer.
            Default is 0.1s
        """
        self.server_address = server_address
        self.repeat_time = repeat_time

    def _request(self, request, url, json=None, blocking=True, timeout=None):
        """
        Internal function to handle requests including timeouts and retries.

        In general, the function reattempts a connection as long as the
        time has not been longer than specified by timeout. If blocking, and
        the "result" field of the returned json is None or "failed", both of
        which indicate a non-successful request, the connection is reattempted.
        Otherwise, or if the connection was successful, the json "result" field
        is returned.

        Parameters
        ----------
        request : requests.request
            The function to use to contact the server. In general, should be
            one of get, post, put, delete etc.
        url : string
            The url of the server, including port. Will ususally consist of
            self.url + some string defining the entry point of the function.
        json : json object, optional
            The json-converted object for this request. Can be None.
        blocking : bool, optional
            If True, retries the query until it receives an acceptable answer, at
            most timeout seconds.
            If False, tries the query only once.
            Default is True.
        timeout : float, optional
            The maximum time to retry the connection. If it is <= 0 or None, this
            is interpreted as a an infinitely long wait.
             Default is None.
        """
        start_time = time.time()
        while timeout is None or timeout <= 0 or time.time()-start_time < timeout:
            if json is None:
                r = request(url=url)
            else:
                r = request(url=url, json=json)
            if blocking:
                if r.json()["result"] is None or r.json()["result"] == "failed":
                    time.sleep(self.repeat_time)
                    continue
            return r.json()["result"]

    def init_experiment(self, name, optimizer, param_defs, optimizer_arguments,
                        minimization=True, blocking=False, timeout=None):
        """
        Initializes an experiment on the apsis server.

        Note that, since failure can regularly happen here if the experiment
        names have not been checked previously, blocking is by default set to
        False.

        Parameters
        ----------
        name : string
            name of the experiment. Must be unique in all experiments.
        optimizer : string
            String representation of the optimizer.
        param_defs : dict of parameter definitions
            Dictionary representing the parameter definitions. Must have the
            following format:
            For each parameter, one entry whose key is the name of the
            parameter as a string. The value is a dictionary whose "type" field
            is the name of the ParamDef class, and whose other fields are the
            kwarg fields of that constructor.
        optimizer_arguments : dict, optional
            A dictionary defining the operation of the optimizer. See the
            respective documentation of the optimizers.
            Default is None, which are default values.
        minimization : bool, optional
            Whether the problem is one of minimization. Defaults to True.
        blocking : bool, optional
            If True, retries the query until it receives an acceptable answer,
            at most timeout seconds.
            If False, tries the query only once.
            Default is False.
        timeout : float, optional
            The maximum time to retry the connection. If it is <= 0 or None,
            this is interpreted as a an infinitely long wait.
             Default is None.

        Returns
        -------
        success : string
            String representing the success of the operation.
            "success" if successful,
            "failed" if failed.
        """
        msg = {
            "name": name,
            "optimizer": optimizer,
            "param_defs": param_defs,
            "optimizer_arguments": optimizer_arguments,
            "minimization": minimization
        }
        url = self.server_address + "/experiments"
        success = self._request(requests.post, url=url, json=msg,
                                blocking=blocking, timeout=timeout)
        return success

    def get_all_experiment_names(self, blocking=True, timeout=None):
        """
        Returns the names of all experiments.

        Parameters
        ----------
        blocking : bool, optional
            If True, retries the query until it receives an acceptable answer, at
            most timeout seconds.
            If False, tries the query only once.
            Default is True.
        timeout : float, optional
            The maximum time to retry the connection. If it is <= 0 or None, this
            is interpreted as a an infinitely long wait.
             Default is None.

        Returns
        -------
        experiment_names : list of strings
            Returns one entry per existing experiment, containing its name.
            If blocking is False, may return None or "failed".
        """
        url = self.server_address + "/experiments"
        return self._request(requests.get, url, blocking=blocking, timeout=timeout)

    def get_next_candidate(self, exp_name, blocking=True, timeout=None):
        """
        Returns the next candidate of an experiment.

        Parameters
        ----------
        exp_name : string
            The name of the experiment to return.
        blocking : bool, optional
            If True, retries the query until it receives an acceptable answer, at
            most timeout seconds.
            If False, tries the query only once.
            Default is True.
        timeout : float, optional
            The maximum time to retry the connection. If it is <= 0 or None, this
            is interpreted as a an infinitely long wait.
             Default is None.

        Returns
        -------
        next_candidate : dict representing a candidate.
            The returned dictionary represents a candidate. It consists of the
            following fields:
            "cost" : float or None
                The cumulative cost of the evaluations of this candidate.
                Must be set by the worker. Default is None, representing no
                cost being set.
            "params" : dict of parameters
                The parameter values this candidate has. The format is
                analogous to the parameter defintion of init_experiment,
                with each entry being an acceptable value according to
                param_def.
            "id" : string
                An id uniquely identifying this candidate.
            "worker_information" : arbitrary
                A field usable for setting worker information, for example a
                directory in which intermediary results are stored. Any
                json-able information can be stored in it (though, since
                it's transferred via network, it is probably better to keep it
                fairly small), and apsis guarantees never to change it.
                By default, it's None.
            "result" : float
                The result of the process we want to optimize.
                Is None by default
            May also return "failed" or None if blocking is false and
            timeout > 0, which represents a failed request.
        """
        url = self.server_address + "/experiments/%s/get_next_candidate" %exp_name
        return self._request(requests.get, url=url, blocking=blocking, timeout=timeout)

    def update(self, exp_name, candidate, status, blocking=True, timeout=None):
        """
        Updates the result of the candidate.

        Parameters
        ----------
        exp_name : string
            The name of the experiment to return.
        candidate : dict representing a candidate
            Represents a candidate. Usually a modified candidate received from
             get_next_candidate.
            It consists of the following fields:
            "cost" : float or None
                The cumulative cost of the evaluations of this candidate.
                Must be set by the worker. Default is None, representing no
                cost being set.
            "params" : dict of parameters
                The parameter values this candidate has. The format is
                analogous to the parameter defintion of init_experiment,
                with each entry being an acceptable value according to
                param_def.
            "id" : string
                An id uniquely identifying this candidate.
            "worker_information" : arbitrary
                A field usable for setting worker information, for example a
                directory in which intermediary results are stored. Any
                json-able information can be stored in it (though, since
                it's transferred via network, it is probably better to keep it
                fairly small), and apsis guarantees never to change it.
                By default, it's None.
            "result" : float
                The result of the process we want to optimize.
                Is None by default
        status : string
            One of "finished", "working" and "pausing".
            "finished": The evaluation is finished.
            "working": The evaluation is still in progress. Later, it will be
            used to ensure that the worker is still working, allowing us to
            reschedule the candidate to other workers if necessary.
            "pausing": Signals that this candidate has paused the execution,
            meaning that we are allowed to reschedule it to another worker.
        blocking : bool, optional
            If True, retries the query until it receives an acceptable answer, at
            most timeout seconds.
            If False, tries the query only once.
            Default is True.
        timeout : float, optional
            The maximum time to retry the connection. If it is <= 0 or None, this
            is interpreted as a an infinitely long wait.
             Default is None.

        Returns
        -------
        result : string
            Returns "success" iff successful, "failed" otherwise.
        """
        url = self.server_address + "/experiments/%s/update" %exp_name
        msg = {
            "status": status,
            "candidate": candidate
        }
        return self._request(requests.post, url, json=msg, blocking=blocking,
                            timeout=timeout)

    def get_best_candidate(self, exp_name, blocking=True, timeout=None):
        """
        Returns the best finished candidate for an experiment.

        Parameters
        ----------
        exp_name : string
            The name of the experiment to return.
        blocking : bool, optional
            If True, retries the query until it receives an acceptable answer, at
            most timeout seconds.
            If False, tries the query only once.
            Default is True.
        timeout : float, optional
            The maximum time to retry the connection. If it is <= 0 or None, this
            is interpreted as a an infinitely long wait.
             Default is None.

        Returns
        -------
        best_candidate : dict as a candidate representation
            Dictionary candidate representation (see get_next_candidate for the
            exact format). May be None if no such candidate exists.
            If blocking is True and timeout > 0, this may return None or
            "Failed".
        """
        url = self.server_address + "/experiments/%s/get_best_candidate" %exp_name
        return self._request(requests.get, url, blocking=blocking, timeout=timeout)

    def get_all_candidates(self, exp_name, blocking=True, timeout=None):
        """
        Returns the candidates for an experiment.

        Parameters
        ----------
        exp_name : string
            The name of the experiment to return.
        blocking : bool, optional
            If True, retries the query until it receives an acceptable answer, at
            most timeout seconds.
            If False, tries the query only once.
            Default is True.
        timeout : float, optional
            The maximum time to retry the connection. If it is <= 0 or None, this
            is interpreted as a an infinitely long wait.
             Default is None.

        Returns
        -------
        candidates : dict of lists
            Returns a dictionary of three lists of candidates.
            Each of the lists contains dictionary candidate representation
            (see get_next_candidate for the exact format). Each list may be
            empty.
            The three lists are:
            "finished": The list of finished candidates.
            "workign": The list of candidates on which workers are currently
            working.
            "pending": The list of not-yet finished candidates on which no
            worker is currently working.
            If blocking is True and timeout > 0, this may return None or
            "Failed".
        """
        url = self.server_address + "/experiments/%s/candidates" %exp_name
        return self.request(requests.get, url, blocking=blocking, timeout=timeout)