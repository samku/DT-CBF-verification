import numpy as np
import jax.numpy as jnp
import control as ctrl
from controlAffine import controlAffine

class VanderpolOscl_trigonometric(controlAffine):
    def __init__(self, nx=2, nu=1):
        super().__init__()

        self.factor_trig = 1.
        self.nx = nx
        self.nu = nu
        self.mu = 1.
        self.dt = 0.1
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = np.array([1])
        self.x_lim = np.array([[-3., -3.], [3., 3.]])
        self.u_lim = np.array([[-2.], [2.]])    

    def f_x(self, x):
        mu = self.mu
        factor_trig = self.factor_trig
        return jnp.array([x[1]+factor_trig*jnp.cos(x[0]),    
                        mu*(1-x[0]**2)*x[1]-x[0]+factor_trig*jnp.sin(x[1])])
    
    def g_x(self, x):
        return jnp.array([[0.], [1.]])
    
    def K_x(self, x):
        #LQR at each step
        mu = self.mu
        A = np.array([[0, 1], [-2*mu*x[0]*x[1]-1, mu*(1-x[0]**2)]])
        B = np.array([[0.], [1.]])
        K, _, _ = ctrl.lqr(A, B, self.Q, self.R)
        control_input = jnp.array(-K@x)
        return control_input

    def BF_indicator(self, x):
        if np.dot(x, x) <= 2.**2 and np.dot(x, x) >= 1.2**2:
            safe = 1
        else:
            safe = 0
        if np.dot(x, x) > 2.8**2 or np.dot(x, x) < 0.4**2:
            unsafe = 1
        else:
            unsafe = 0
        if np.dot(x, x) <= 3**2:
            BF = 1
        else:
            BF = 0
        return [safe, unsafe, BF]