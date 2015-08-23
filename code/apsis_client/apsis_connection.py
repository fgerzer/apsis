__author__ = 'Frederik Diehl'

import requests

class Connection(object):
    server_address = None

    def __init__(self, server_address):
        self.server_address = server_address

    def init_experiment(self, name, optimizer, param_defs, optimizer_arguments,
                        minimization=True):
        msg = {
            "name": name,
            "optimizer": optimizer,
            "param_defs": param_defs,
            "optimizer_arguments": optimizer_arguments,
            "minimization": minimization
        }
        url = self.server_address + "/experiments"
        r = requests.post(url=url, json=msg)
        #TODO add error parsing

    def get_all_experiment_names(self):
        url = self.server_address + "/experiments"
        r = requests.get(url=url)
        return r.json()["result"]


    def get_next_candidate(self, exp_name):
        url = self.server_address + "/experiments/%s/get_next_candidate" %exp_name
        r = requests.get(url=url)
        if r.json()["result"] == "None":
            return None
        return r.json()["result"]

    def update(self, exp_name, candidate, status):
        #TODO candidate currently as a dict.
        #candidate.to_dict()
        url = self.server_address + "/experiments/%s/update" %exp_name
        msg = {
            "status": status,
            "candidate": candidate
        }
        r = requests.post(url, json=msg)
        print(r)

    def get_best_candidate(self, exp_name):
        url = self.server_address + "/experiments/%s/get_best_candidate" %exp_name
        r = requests.get(url)
        print(r)
        if r.json()["result"] == "None":
            return None
        return r.json()["result"]

    def get_all_candidates(self, exp_name):
        url = self.server_address + "/experiments/%s/candidates" %exp_name
        r = requests.get(url)
        if r.json()["result"] == "None":
            return None
        return r.json()["result"]