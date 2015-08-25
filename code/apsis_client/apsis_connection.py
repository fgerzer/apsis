__author__ = 'Frederik Diehl'

import requests
import time
from apsis.utilities.plot_utils import plot_lists

class Connection(object):
    server_address = None
    repeat_time = None

    def __init__(self, server_address, repeat_time=0.1):
        self.server_address = server_address
        self.repeat_time = repeat_time

    def request(self, request, url, json=None, blocking=True, timeout=0):
        start_time = time.time()
        while timeout <= 0 or time.time()-start_time < timeout:
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
                        minimization=True, blocking=True, timeout=0):
        msg = {
            "name": name,
            "optimizer": optimizer,
            "param_defs": param_defs,
            "optimizer_arguments": optimizer_arguments,
            "minimization": minimization
        }
        url = self.server_address + "/experiments"
        r = self.request(requests.post, url=url, json=msg, blocking=blocking,
                         timeout=0)
        return r

    def get_all_experiment_names(self, blocking=True, timeout=0):
        url = self.server_address + "/experiments"
        return self.request(requests.get, url, blocking=blocking, timeout=timeout)

    def get_next_candidate(self, exp_name, blocking=True, timeout=0):
        url = self.server_address + "/experiments/%s/get_next_candidate" %exp_name
        return self.request(requests.get, url=url, blocking=blocking, timeout=timeout)

    def update(self, exp_name, candidate, status, blocking=True, timeout=0):
        #TODO candidate currently as a dict.
        #candidate.to_dict()
        url = self.server_address + "/experiments/%s/update" %exp_name
        msg = {
            "status": status,
            "candidate": candidate
        }
        return self.request(requests.post, url, json=msg, blocking=blocking,
                            timeout=timeout)

    def get_best_candidate(self, exp_name, blocking=True, timeout=0):
        url = self.server_address + "/experiments/%s/get_best_candidate" %exp_name
        return self.request(requests.get, url, blocking=blocking, timeout=timeout)

    def get_all_candidates(self, exp_name, blocking=True, timeout=0):
        url = self.server_address + "/experiments/%s/candidates" %exp_name
        return self.request(requests.get, url, blocking=blocking, timeout=timeout)

    def get_figure_results_per_step(self, exp_name, title=None, blocking=True, timeout=0):
        url = self.server_address + "/experiments/%s/fig_results_per_step" %exp_name
        list = self.request(requests.get, url, blocking=blocking, timeout=timeout)
        if title is None:
            title = "Result for %s" %exp_name
        plot_options = {
            "legend_loc": "upper left",
            "x_label": "steps",
            "y_label": "result",
            "title": title
        }
        return plot_lists(list, plot_options)