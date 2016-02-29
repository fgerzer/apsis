__author__ = 'Frederik Diehl'

import REST_interface
import argparse

# Fix for TclError: no display name and no $DISPLAY environment variable
import matplotlib
matplotlib.use('Agg')

def start_rest(port=5000, validation=False, cv=5, continue_path=None):
    print("Initializing apsis. Val is %s, cv is %s" %(validation, cv))
    print("Initialized apsis on port %s" %port)
    print("in start_rest: %s" %continue_path)
    REST_interface.start_apsis(port, validation, continue_path=continue_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Set the apsis server's port.")
    parser.add_argument("--cross_validation", help="Set the number of cvs if "
                                                   "using --validation.")
    parser.add_argument("--validation", help="Use ValidationLabAssistant",
                    action="store_true")
    parser.add_argument("--continue_path", help="Continue a previous experiment.")
    args = parser.parse_args()

    validation = False
    if args.validation:
        validation = True
    cv = 5
    if args.cross_validation:
        cv = args.cross_validation
    port = 5000
    if args.port:
        port = args.port
    continue_path = None
    if args.continue_path:
        continue_path = args.continue_path
    print("Continue path: %s" %continue_path)
    start_rest(port, validation, cv, continue_path)