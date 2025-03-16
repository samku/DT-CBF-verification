import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import linen as nn
import pickle
import time
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from pathlib import Path
from itertools import combinations
import qpax
current_directory = Path(__file__).parent

