__author__ = 'Frederik Diehl'

import REST_interface
import argparse


def start_rest(port=5000):
    print("Initialized apsis on port %s" %port)
    REST_interface.start_apsis(port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Set the apsis server's port.")
    args = parser.parse_args()

    port = 5000
    if args.port:
        port = args.port
    start_rest(port)