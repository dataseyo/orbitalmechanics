import numpy as np
from scipy.integrate import solve_ivp
import math

'''
The Sat class takes in either the Keplerian elements or the state vector of a satellite in perifocal coordinates,
as well as the gravitational parameter mu. 

It provides various useful methods for working with satellite trajectories, such as converting the elements to state vectors
in various coordinate systems, 
'''

class Sat():
    def __init__(self, elements=[], state=[], mu=3.986004418e5):
        self.elements = elements
        self.state = state
        self.mu = mu

        if elements == [] and state ==[]:
            raise Exception("Please provide the Keplerian elements or state vector.")
        elif not elements:
            self.elements = self.state_to_elem()
        elif not state:
            self.state = self.elem_to_state("ECI")

    def state_to_elem(self):
        r = self.state[:3]
        v = self.state[3:]
        h_vec = np.cross(r, v)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        h_mag = np.linalg.norm(h_vec)

        e = (np.dot((v_mag**2 - self.mu / r_mag), r) - (np.dot(np.dot(r, v), v))) / self.mu
        e_mag = np.linalg.norm(e)

        # semi maajor axis
        a = 1 / (2/r_mag - v_mag**2/self.mu)
        
        # inclination
        i = np.rad2deg(np.arccos(h_vec[2] / h_mag))

        # RAAN
        k = np.array((0, 0, 1))
        n_vec = np.cross(k, h_vec)
        n_mag = np.linalg.norm(n_vec)
        O = 0
        if n_vec[1] > 0: 
            O = np.rad2deg(np.arccos(n_vec[0] / n_mag))
        else: 
            O = 360 - np.rad2deg(np.arccos(n_vec[0] / n_mag))
            
        # argument of perigee
        w = 0
        if e[2] > 0: 
            w = np.rad2deg(np.arccos(np.dot(n_vec, e) / (n_mag * e_mag)))
        else: 
            w = 360 - np.rad2deg(np.arccos(np.dot(n_vec, e) / (n_mag * e_mag)))

        # true anomaly
        f = 0
        if np.dot(r, v) > 0:
            f = np.rad2deg(np.arccos(np.dot(e, r) / (e_mag*r_mag)))
        else: 
            f = 360 - np.rad2deg(np.arccos(np.dot(e, r) / (e_mag*r_mag)))
        return [a, e_mag, i, O , w, f]
    
    def elem_to_state(self, type):
        '''
        Converts Keplerian orbital elements into a state vector represented in { type } coordinates.

        Type can be: 
            - PQW (perifocal)
            - ECI (Earth centered inertial)
            - ECEF (Earth-centered Earth-fixed)
        '''

        # get elements from elements vector
        a = self.elements[0]
        e = self.elements[1]
        i = self.elements[2]
        O = self.elements[3]
        w = self.elements[4]
        F = self.elements[5]

        # compute initial constants
        # run inverse Kepler if evolving the orbital elements
        #     nu = np.sqrt(mu / a**3)
        #     M_0 = np.deg2rad(10)
        #     M_t = M_0 + nu * dt
        #     E = inverse_kepler(M_0, e, 10)

        #     # compute true anomaly
        #     F = 2 * math.atan(math.sqrt((1+e)/(1-e)) * math.tan(E/2))
        #     F = F % (2*np.pi)
        #     F = np.rad2deg(F)

        theta = w + F
        p = a * (1 - e**2)
        h = np.sqrt(p*self.mu)

        # compute PQW coords (these are necessary no matter the final coordinate system)
        r_p = np.array([[
            ((p*dcos(F)) / (1+e*dcos(F))),
            ((p*dsin(F)) / (1+e*dcos(F))),
            0
        ]])

        v_p = self.mu/h * np.array([[
            - dsin(F),
            (e + dcos(F)),
            0
        ]])

        if type == "PQW":
            return np.hstack([r_p, v_p])
        elif type == "ECI": 
            r_trans = np.matrix([
                [dcos(O)*dcos(w)-dsin(O)*dsin(w)*dcos(i), -dcos(O)*dsin(w)-dsin(O)*dcos(w)*dcos(i), dsin(O)*dsin(i)],
                [dsin(O)*dcos(w)+dcos(O)*dsin(w)*dcos(i), -dsin(O)*dsin(w)+dcos(O)*dcos(w)*dcos(i), -dcos(O)*dsin(i)],
                [dsin(w)*dsin(i), dcos(w)*dsin(i), dcos(i)]
            ])
            
            r = r_trans * r_p.T
            v = r_trans * v_p.T
            
            return np.hstack([r.T.tolist()[0], v.T.tolist()[0]])
    
    def ECI_to_ECF(self):
        # rotation matrices from ECI to ECEF
        pass
  
    def print_elements(self):
        print("semi major axis", self.elements[0], "\neccentricity", self.elements[1], "\ninclination", self.elements[2], "\nright ascension", self.elements[3], "\nargument of periapsis", self.elements[4], "\ntrue anomaly", self.elements[5])

    def print_state(self):
        print("position", self.state[:3], "velocity", self.state[3:])

    def evolve_state(self, t):
        # evolve the state of the satellite to time t (where t is in seconds)
        t_points = np.linspace(0, t, 10000)
        sol = solve_ivp(two_body_eom, [0, t], self.state, t_eval=t_points, method="DOP853", args=[self.mu])
        y = sol.y.T
        R = y[:, :3]
        V = y[:, 3:]
        return (R, V)
    
    def get_period(self):
        '''
        Compute orbit period. This can then be used in other computations, such as in evolve_state for generating arrays
        of positions and velocities that perfectly loop with the orbital period. 
        '''
        # print(self.elements[0], pow(self.elements[0], 3/2))
        period = (2*np.pi * pow(self.elements[0], 3/2)) / np.sqrt(self.mu)
        return period
        
    def ground_track(self):
        '''
        Find the ground track of the satellite with respect to the reference ellipsoid.

        Steps:
            (0) express state in Perifocal coordinates
            (1) convert Perifocal to ECI (Earth Centered Inertial)
            (2) convert ECI to ECEF (Earth Centered Earth Fixed)
            (3) convert ECEF to Geodetic
                - find geodetic latitude and longitude
                - find height of satellite above reference ellipsoid
        '''
        track = []
        return track
    
def inverse_kepler(E_0, e, iter):
    E_n = E_0
    for i in range(0, iter):
        E_n = E_n - ((E_n - E_0 - e * np.sin(E_n)) / (1 - e * np.cos(E_n)))
    return E_n

def two_body_eom(t, y, mu):
    # initial conditions
    r = y[:3]
    v = y[3:]
    
    # result vector
    res = np.zeros_like(y)
    res[:3] = v

    r_mag = np.sqrt(np.sum(np.square(r)))
    a = (- mu * r) / r_mag**3 
    res[3:] = a
    return res

def handle_units(type, quantity, arr):
    # convert units to and from canonical and metric 
    # note: to avoid hardcoded frontend scaling, should work in canonical (TU/DU) units.
    if type == "canonical":
        if quantity == "r":
            return arr / 6378.136
        if quantity == "v":
            return arr / 7.90536828
    if type == "metric":
        if quantity == "r":
            return arr * 6378.136
        if quantity == "v":
            return arr * 7.90536828

def dcos(angle):
    # takes in degrees and converts to rad to perform cosine
    return np.cos(np.deg2rad(angle))

def dsin(angle):
    # takes in degrees and converts to rad to perform sine
    return np.sin(np.deg2rad(angle))