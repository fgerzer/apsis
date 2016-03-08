__author__ = 'Frederik Diehl'

# Fix for TclError: no display name and no $DISPLAY environment variable
import matplotlib
matplotlib.use('Agg')


import REST_interface
import argparse

def start_rest(port=5000, continue_path=None, fail_deadly=False):
    print("Initialized apsis on port %s" %port)
    print("Fail_deadly is %s" %fail_deadly)
    REST_interface.start_apsis(port, continue_path=continue_path,
                               fail_deadly=fail_deadly)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Set the apsis server's port.")
    parser.add_argument("--continue_path", help="Continue a previous "
                                                "experiment.")
    parser.add_argument("--fail_deadly", help="Fails with every exception "
                                              "instead of catching them. "
                                              "Warning! Dangerous. Do not use "
                                              "unless you know what you do.")
    args = parser.parse_args()
    print(args)
    port = 5000
    if args.port:
        port = args.port
    continue_path = None
    fail_deadly = False
    if args.continue_path:
        continue_path = args.continue_path
    if args.fail_deadly:
        fail_deadly = True
    start_rest(port, continue_path, fail_deadly)
