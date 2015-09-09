import sys; 
sys.path.append("//srv-file.brml.tum.de/nthome/fdiehl/apsis/code/"); 
from apsis_client.apsis_connection import Connection
import math
#server_address="http://10.162.85.138:5116"
server_address="http://pc-hiwi6:5116"
exp_id = "42ec6aa0a427482da402ee5818a52be6"
import requests
import socket

def branin_func(x, y, a=1, b=5.1/(4*math.pi**2), c=5/math.pi, r=6, s=10,
                t=1/(8*math.pi)):
        """
        Branin hoo function.

        This is the same function as in
        http://www.sfu.ca/~ssurjano/branin.html. The default parameters are
        taken from that same site.

        With the default parameters, there are three minima with f(x)=0.397887:
        (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475).

        Parameters
        ---------
        x : float
            A real valued float
        y : float
            A real valued float
        a, b, c, r, s, t : floats, optional
            Parameters for the shape of the Branin hoo function. Thier default
            values are according to the recommendations of the above website.
        Returns
        -------
        result : float
            A real valued fkloat.
        """
        result = a*(y-b*x**2+c*x-r)**2 + s*(1-t)*math.cos(x)+s
        return result



conn = Connection(server_address)
to_eval = conn.get_next_candidate(exp_id, blocking=True, timeout=30)
result = branin_func(to_eval["params"]["x"], to_eval["params"]["y"])
to_eval["result"] = result
to_eval["worker_information"] = socket.gethostname()
conn.update(exp_id, to_eval, "finished", blocking=True, timeout=30)
