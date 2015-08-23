__author__ = 'Frederik Diehl'

import requests
import time

class Connection(object):
    server_address = None

    def __init__(self, server_address):
        self.server_address = server_address

    def request(self, request, url, json=None, blocking=True, timeout=0):
        start_time = time.time()
        while timeout <= 0 or time.time()-start_time < timeout:
            if json is None:
                r = request(url=url)
            else:
                r = request(url=url, json=json)
            if blocking:
                if r.json()["result"] is None or r.json()["result"] == "failed":
                    time.sleep(0.1)
                    continue
            #if not blocking:
            #    if r.json()["result"] is None:
            #        return None
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