import math

def branin_func(x, y, a=1, b=5.1/(4*math.pi**2), c=5/math.pi, r=6, s=10,
                t=1/(8*math.pi)):
        """
        Branin hoo function.

        This is the same function as in
        http://www.sfu.ca/~ssurjano/branin.html. The default parameters are
        taken from that same site.

        With the default parameters, there are three minima with f(x)=0.397887:
        (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475).

        Paramters
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
            A real valued float.
        """
        result = a*(y-b*x**2+c*x-r)**2 + s*(1-t)*math.cos(x)+s
        return result
