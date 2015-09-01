__author__ = 'Frederik Diehl'

import REST_interface
import sys

def start_rest(validation=False, cv=5):
    print("Initializing apsis. Val is %s, cv is %s" %(validation, cv))
    REST_interface.start_apsis(validation)
    print("Initialized apsis.")

if __name__ == "__main__":
    validation = False
    cv = 5
    if sys.argv > 1:
        if sys.argv[1] == "val":
            validation = True
        if sys.argv > 2:
            cv = int(sys.argv[2])
    start_rest(validation, cv)