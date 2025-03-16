from imports import *
from jax import grad, jit, value_and_grad
from jax.scipy.linalg import block_diag
import jaxopt
from vector_to_params import extract_params
from jax.scipy.optimize import minimize
from jax import grad, jit, jacobian
from jax.tree_util import tree_flatten, tree_unflatten
from jax_sysid.utils import lbfgs_options, vec_reshape


def CBF_Step1(dataset, BF_params, K_params, NN_params, train_params, activation):
    
    #Extract data
    system = dataset['system']
    X_train = dataset['X_train']
    T_train = dataset['T_train']
    N_train = X_train.shape[0]
    nx = system.nx
    nu = system.nu
    id_safe = np.where(T_train[:, 0] == 1)
    id_unsafe = np.where(T_train[:, 1] == 1)
    X_safe = X_train[id_safe]
    X_unsafe = X_train[id_unsafe]

    #BF params
    alpha = BF_params['alpha']
    lu = BF_params['lu']
    delta = BF_params['delta']
    
    #NN params
    nH = NN_params['nH']
    nth = NN_params['nth']

    #K params
    nL_K = K_params['nL_K']
    nH_K = K_params['nH_K']
    nth_K = K_params['nth_K']

    #Training params
    adam_iters = train_params['adam_iters']
    BFGS_iters = train_params['BFGS_iters']
    regularization = train_params['regularization']

    #Initialize
    total_NN = (nth * nx + nth + (nH - 1) * nth * nth + (nH - 1) * nth + nth + 1)
    total_K = (nth_K * nx + nth_K + (nH_K - 1) * nth_K * nth_K + (nH_K - 1) * nth_K + 
               (nu * nL_K + nu) * nth_K + (nu * nL_K + nu))
    num_vars = total_NN + total_K
    seed = 21
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num=2)
    param_vec = 0.1*jax.random.normal(keys[1], (num_vars,))

    params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)

    jax.config.update("jax_enable_x64", True)
    #Make network
    @jax.jit
    def h_x(x,params):
        Win, bin, Whid, bhid, Wout, bout = params
        post_linear = Win@x+bin
        post_activation = activation(post_linear)
        for j in range(nH-1): 
            post_linear = Whid[j]@post_activation+bhid[j]
            post_activation = activation(post_linear)
        post_linear = Wout@post_activation+bout
        return post_linear
    
    @jax.jit
    def compute_h(x, params):
        return jax.vmap(lambda xi: h_x(xi, params))(x).reshape(1,-1)[0]
    
    #Formulate log objective
    @jax.jit
    def objective(param_vec):
        #Maximize entropy
        params, _ = extract_params(param_vec, nx, nu, NN_params, K_params)
        h_safe = compute_h(X_safe, params)
        h_unsafe = compute_h(X_unsafe, params)
        loss_safe = jnp.sum(jnp.maximum(0, h_safe))/X_safe.shape[0]
        loss_unsafe = jnp.sum(jnp.maximum(0, lu-h_unsafe))/X_unsafe.shape[0]
        r_cost = 0.
        for i in range(len(params)):    
            r_cost += jnp.sum(params[i]**2)
        return loss_safe+loss_unsafe+regularization*r_cost, loss_safe, loss_unsafe
    
    @jax.jit
    def objective_single(x):
        return objective(x)[0]
    
    grad_fn = jit(grad(objective_single))
    gradients = grad_fn(param_vec)

    #Adam params
    m = jnp.zeros(num_vars)  # Initialize first moment vector
    v = jnp.zeros(num_vars)  # Initialize second moment vector
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    print_interval = 100
    cost_history = []  
    for i in range(adam_iters):
        [violation, safe, unsafe] = objective(param_vec)
        cost_history.append(np.array([violation, safe, unsafe]))

        #Update params
        gradients = grad_fn(param_vec)
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        param_vec = param_vec - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)

        #Printing
        
        if np.remainder(i, print_interval) == 0:
            print('Iteration:', i, ', Cost:', violation, ', Safe:', safe, ', Unsafe:', unsafe)

    params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params) 
    h_eval = compute_h(X_train, params)
    xId_safe = np.where(h_eval<=0)  
    xId_unsafe = np.where(h_eval>lu)
    param_vec_adam = param_vec.copy()
    
    options_BFGS = lbfgs_options(iprint=5, iters=BFGS_iters, lbfgs_tol=1.e-10, memory=100)
    solver = jaxopt.ScipyBoundedMinimize(
        fun=objective_single, tol=1.e-10, method="L-BFGS-B", maxiter=BFGS_iters, options=options_BFGS)
    param_vec_warmstart, state = solver.run(np.array(param_vec_adam),bounds=(-10*np.ones(num_vars), 10*np.ones(num_vars)))

    params, params_K = extract_params(param_vec_warmstart, nx, nu, NN_params, K_params) 
    h_eval = compute_h(X_train, params)
    xId_safe = np.where(h_eval<=0)  
    xId_unsafe = np.where(h_eval>lu)
    X_safe_recon = X_train[xId_safe]
    X_unsafe_recon = X_train[xId_unsafe]

    plt.scatter(X_safe_recon[:, 0], X_safe_recon[:, 1], c='g', label='Safe')
    plt.scatter(X_unsafe_recon[:, 0], X_unsafe_recon[:, 1], c='r', label='Unsafe')
    plt.legend()
    plt.show()

 
    return np.array(param_vec_warmstart).copy()