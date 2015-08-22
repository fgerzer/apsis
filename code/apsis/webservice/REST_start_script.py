__author__ = 'Frederik Diehl'

from REST_interface import app
import REST_interface
from apsis.assistants.lab_assistant import LabAssistant
from multiprocessing.queues import Queue
import multiprocessing

REST_interface.start_apsis()
print("Initialized apsis.")