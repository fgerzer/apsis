import os

def ensure_directory_exists(directory):
        """
        Creates the given directory if not existed.

        Parameters
        ----------
        directory : String
            The name of the directory that shall be created if not exists.
        """
        if not os.path.exists(directory):
                os.makedirs(directory)