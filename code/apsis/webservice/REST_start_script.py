__author__ = 'Frederik Diehl'

import REST_interface
import argparse

# Fix for TclError: no display name and no $DISPLAY environment variable
import matplotlib
matplotlib.use('Agg')

def start_rest(port=5000, continue_path=None):
    print("Initialized apsis on port %s" %port)
    REST_interface.start_apsis(port, continue_path=continue_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Set the apsis server's port.")
    parser.add_argument("--continue_path", help="Continue a previous "
                                                "experiment.")
    args = parser.parse_args()

    port = 5000
    if args.port:
        port = args.port
    continue_path = None
    if args.continue_path:
        continue_path = args.continue_path
    start_rest(port, continue_path)
