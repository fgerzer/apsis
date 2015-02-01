from apsis.utilities.import_utils import import_if_exists
import nose.tools as nt

class TestImporter(object):
    def test_import_non_existant_module(self):
        worked, module = import_if_exists("os_wrong")

        assert worked == False

    def test_import_existant_module(self):
        worked, os = import_if_exists("os")

        assert worked == True

