import math

def branin_func(x, y):
        """
        Branin hoo function according to http://www.sfu.ca/~ssurjano/branin.html

        Paramters
        ---------
        x: float
            A real valued float
        y: float
            A real valued float

        Returns
        -------
        result: float
            A real valued float.
        """
        a = 1
        b = 5.1/(4*math.pi**2)
        c = 5/math.pi
        r = 6
        s = 10
        t = 1/(8*math.pi)
        result = a*(y-b*x**2+c*x-r)**2 + s*(1-t)*math.cos(x)+s
        #print("Branin: %f" %result)
        return result
