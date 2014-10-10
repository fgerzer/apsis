#!/usr/bin/python

from apsis.OptimizationCoreInterface import OptimizationCoreInterface


class RandomSearchCore(OptimizationCoreInterface):
    def __init__(self, lower_bound, upper_bound):
        print("Initializing Random Search Core for bounds..." + str(lower_bound) + " and " + str(upper_bound))

    def working(self, candidate, status, worker_id=None, can_be_killed=False):
        print("Worker seinding worker informatiojn")

    def next_candidate(self, worker_id=None):
        return None

