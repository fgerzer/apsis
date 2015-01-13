import os

def ensure_directory_exists(directory):
        """
        Creates the given directory if not existed.
        """
        if not os.path.exists(directory):
                os.makedirs(directory)