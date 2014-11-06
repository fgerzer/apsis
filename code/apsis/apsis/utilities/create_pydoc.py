#!/usr/bin/python

"""
Script for generating the pydoc of this project.

Needs to be invoked with path to the folder containg apsis and optionally a destination dir

python apsis/utilities/create_pydoc.py <path_to_folder_containing_apsis> [<destination_dir>]
"""

import os
import sys
import commands
import shutil

#assuming we run from project root
path = sys.argv[1]
target = None
if len(sys.argv) >= 3:
    target = sys.argv[2]
    print ("setting target to " + str(target))

os.chdir(path)
cwd = os.getcwd()

python_file_names = []
python_module_names = []

shutil.rmtree('pydoc', ignore_errors=True)
os.mkdir("pydoc")

for (dirpath, dirnames, filenames) in os.walk(cwd):
   for dirname in dirnames:
       full_module_path = os.sep.join([dirpath, dirname])

       if full_module_path.startswith(cwd):
           #remove the trailing path and one more sign which is the beginning slash
           full_module_path = full_module_path[len(cwd) + 1:]

       module_name = full_module_path.replace("/", ".")
       python_module_names.append(module_name)

       print ("calling pydoc -w " + module_name  + " from " + cwd)
       commands.getoutput("pydoc -w " + module_name)

       if os.path.isfile(module_name + ".html"):
           shutil.move(module_name + ".html", "pydoc")

   for filename in filenames:
       if filename[-3:] == '.py':
           full_file_name = os.sep.join([dirpath, filename])

           #don't pydoc init files
           if "__init__.py" in full_file_name:
               continue

           if full_file_name.startswith(cwd):
               #remove the trailing path and one more sign which is the beginning slash
               full_file_name = full_file_name[len(cwd) + 1:]

           module_name = full_file_name.replace("/", ".")
           module_name = module_name.replace(".py", "")

           python_file_names.append(full_file_name)
           python_module_names.append(module_name)

           print ("adding " + module_name + " for " + filename)
           print ("calling pydoc -w " + module_name  + " from " + cwd)
           commands.getoutput("pydoc -w " + module_name)

           if os.path.isfile(module_name + ".html"):
               shutil.move(module_name + ".html", "pydoc")

if not target is None and not os.path.abspath("pydoc") == os.path.abspath(target + "/pydoc"):
    shutil.move("pydoc", target)





