from imports import *
from jax import grad, jit, value_and_grad
from jax.scipy.linalg import block_diag
import jaxopt
from vector_to_params import extract_params
from jax.scipy.optimize import minimize
from jaxopt import BoxCDQP
from jax import grad, jit, jacobian
from jax.tree_util import tree_flatten, tree_unflatten
from jax_sysid.utils import lbfgs_options, vec_reshape


def CBF_Step2(dataset, BF_params, K_params, NN_params, train_params, activation, activation_K, param_vec_warmstart):
    
    #Extract data
    system = dataset['system']
    X_train = dataset['X_train']
    T_train = dataset['T_train']
    N_train = X_train.shape[0]
    nx = system.nx
    nu = system.nu
    dt = system.dt
    id_safe = np.where(T_train[:, 0] == 1)
    id_unsafe = np.where(T_train[:, 1] == 1)
    id_barrier = np.where(T_train[:, 2] == 1)
    X_safe = X_train[id_safe]
    X_unsafe = X_train[id_unsafe]
    X_barrier = X_train[id_barrier]
    
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
    penalty_safe = train_params['penalty_safe']
    penalty_unsafe = train_params['penalty_unsafe']
    penalty_barrier = train_params['penalty_barrier']
    regularization = train_params['regularization']
    regularization_grad = train_params['regularization_grad']
    sim_length = train_params['sim_length']

    #Initialize
    param_vec = jnp.array(param_vec_warmstart.copy())
    num_vars = len(param_vec)
    params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)

    jax.config.update("jax_enable_x64", True)
    #Make network
    @jax.jit
    def h_x(x,param_vec):
        params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)
        Win, bin, Whid, bhid, Wout, bout = params
        post_linear = Win@x+bin
        post_activation = activation(post_linear)
        for j in range(nH-1): 
            post_linear = Whid[j]@post_activation+bhid[j]
            post_activation = activation(post_linear)
        post_linear = Wout@post_activation+bout
        return post_linear
    
    @jax.jit
    def compute_h(x, param_vec):
        return jax.vmap(lambda xi: h_x(xi, param_vec))(x).reshape(1,-1)[0]
    
    @jax.jit
    def Lc_x(x,params_K):
        Win_K, bin_K, Whid_K, bhid_K, Wout_K, bout_K = params_K
        post_linear = Win_K@x+bin_K
        post_activation = activation_K(post_linear)
        for j in range(nH_K-1): 
            post_linear = Whid_K[j]@post_activation+bhid_K[j]
            post_activation = activation_K(post_linear)
        post_linear = Wout_K@post_activation+bout_K
        return post_linear

    @jax.jit
    def integrator(x,u):
        k1 = system.f_x(x) + system.g_x(x)@u
        k2 = system.f_x(x + 0.5*dt*k1) + system.g_x(x + 0.5*dt*k1)@u
        k3 = system.f_x(x + 0.5*dt*k2) + system.g_x(x + 0.5*dt*k2)@u
        k4 = system.f_x(x + dt*k3) + system.g_x(x + dt*k3)@u
        x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    def create_qp_layer():
        def qp_layer(Q,c):
            qp_solver = BoxCDQP()
            params_obj = (Q, c)
            params_ineq = (system.u_lim[0], system.u_lim[1])
            result = qp_solver.run(init_params=jnp.zeros(nu),    
                                params_obj=params_obj, 
                                params_ineq = params_ineq).params
            return result
        return qp_layer
    qp_layer = create_qp_layer()

    A_eq = np.zeros((0,nu))
    b_eq = np.zeros(0)
    A_ineq = np.vstack((np.eye(nu),-np.eye(nu)))
    b_ineq = np.hstack((system.u_lim[1],-system.u_lim[0]))
    @jax.jit
    def CBF_x(x, param_vec):
        params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)
        #Compute control input
        Lc = Lc_x(x,params_K)
        L = Lc[:nu*nL_K].reshape(nu,nL_K)
        Q = 0.5*(L @ L.T + (L @ L.T).T) + 1*jnp.eye(nu)
        c = Lc[nu*nL_K:]
        u = qp_layer(Q,c)
        #Compute CBF loss
        h_curr = h_x(x, param_vec)
        x_next = integrator(x,u)
        h_next = h_x(x_next, param_vec)
        CBF_loss = h_next-(1.-alpha)*h_curr+delta
        return CBF_loss, u, x_next, h_curr, h_next
        
    @jax.jit
    def compute_CBF(x, param_vec):
        CBF_loss, u, x_next, h_curr, h_next = jax.vmap(lambda xi: CBF_x(xi, param_vec))(x)
        return CBF_loss.reshape(1,-1)[0], u, x_next, h_curr.reshape(1,-1)[0], h_next.reshape(1,-1)[0]
    
    @jax.jit
    def CBF_closedLoop(x, index, param_vec):
        params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)
        #Compute control input
        Lc = Lc_x(x,params_K)
        L = Lc[:nu*nL_K].reshape(nu,nL_K)
        Q = 0.5*(L @ L.T + (L @ L.T).T) + 1.*jnp.eye(nu)
        c = Lc[nu*nL_K:]
        u = qp_layer(Q,c)
        x_next = integrator(x,u)
        delta_h = h_x(x_next, param_vec)-(1.-alpha)*h_x(x, param_vec)+delta
        return x_next, delta_h
    
    @jax.jit
    def CBF_closedLoop_single(x, param_vec):
        par_CBF_closedLoop = partial(CBF_closedLoop, param_vec=param_vec)
        return jax.lax.scan(par_CBF_closedLoop, x, np.arange(sim_length))[1]
    
    @jax.jit
    def compute_CBF_CL(param_vec):
        return jnp.vstack(jax.vmap(lambda xi: CBF_closedLoop_single(xi, param_vec))(X_barrier))[:,0]
    
    @jax.jit
    def CBF_closedLoop_noQP(x, index, param_vec):
        params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)
        #Compute control input
        Lc = Lc_x(x,params_K)
        c = Lc[nu*nL_K:]
        u = jnp.minimum(jnp.maximum(-c, system.u_lim[0]), system.u_lim[1])
        x_next = integrator(x,u)
        delta_h = h_x(x_next, param_vec)-(1.-alpha)*h_x(x, param_vec)+delta
        return x_next, delta_h
    
    @jax.jit
    def CBF_closedLoop_single_noQP(x, param_vec):
        par_CBF_closedLoop = partial(CBF_closedLoop_noQP, param_vec=param_vec)
        return jax.lax.scan(par_CBF_closedLoop, x, np.arange(sim_length))[1]
    
    @jax.jit
    def compute_CBF_CL_noQP(param_vec):
        return jnp.vstack(jax.vmap(lambda xi: CBF_closedLoop_single_noQP(xi, param_vec))(X_barrier))[:,0]
    
    
    #Solve with Adam for some iterations
    @jax.jit
    def objective(param_vec):
        params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)
        h_safe = compute_h(X_safe, param_vec)
        h_unsafe = compute_h(X_unsafe, param_vec)
        CBF_loss = compute_CBF_CL_noQP(param_vec)

        safe_loss = jnp.sum(jnp.maximum(h_safe,0))/X_safe.shape[0]
        unsafe_loss = jnp.sum(jnp.maximum(lu-h_unsafe,0))/X_unsafe.shape[0]
        cbf_loss = jnp.sum(jnp.maximum(0, CBF_loss))/len(CBF_loss)

        normalized_loss = penalty_safe*safe_loss+penalty_unsafe*unsafe_loss+penalty_barrier*cbf_loss

        total_loss = normalized_loss+regularization*jnp.sum(param_vec**2)
        return total_loss, safe_loss, unsafe_loss, cbf_loss

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
    best_params = param_vec.copy()
    best_value = objective(param_vec)[0]
    for i in range(adam_iters):
        value, safe, unsafe, barrier = objective(param_vec)
        cost_history.append(np.array([value, safe, unsafe, barrier]))

        if i==0:
            best_params = param_vec.copy()
            best_value = value
        else:
            if value < best_value:
                best_params = param_vec.copy()
                best_value = value

        #Update params
        gradients = grad_fn(param_vec)
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        param_vec = param_vec - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)

        #Printing
        if np.remainder(i, print_interval) == 0:
            print('Iteration:', i, ', Cost:', value, ', Safe:', safe, ', Unsafe:', unsafe, ', Barrier:', barrier, 'best:', best_value)

    param_CBF = best_params.copy()
    params, params_K = extract_params(param_CBF, nx, nu, NN_params, K_params)
    h_safe = compute_h(X_safe, param_CBF)
    h_unsafe = compute_h(X_unsafe, param_CBF)
    CBF_loss, u, x_next, h_curr, h_next = compute_CBF(X_barrier, param_CBF)

    safe_loss = jnp.max(jnp.maximum(h_safe,0))
    unsafe_loss = jnp.max(jnp.maximum(lu-h_unsafe,0))
    cbf_loss = jnp.max(jnp.maximum(0, CBF_loss))
    violation_vector = jnp.hstack((safe_loss, unsafe_loss, cbf_loss))  
    print('Violation vector:', violation_vector) 

    h_all = compute_h(X_train, param_CBF)
    h_neg = np.where(h_all<0)
    h_pos = np.where(h_all>=0)

    plt.scatter(X_train[h_neg, 0], X_train[h_neg, 1], c='r', label='Negative')
    plt.scatter(X_train[h_pos, 0], X_train[h_pos, 1], c='g', label='Positive')
    plt.legend()
    plt.show()

    params, params_K = extract_params(best_params, nx, nu, NN_params, K_params)
    param_vec_adam = param_vec.copy()

    options_BFGS = lbfgs_options(iprint=2, iters=BFGS_iters, lbfgs_tol=1.e-10, memory=100)
    solver2 = jaxopt.ScipyBoundedMinimize(
        fun=objective_single, tol=1.e-10, method="L-BFGS-B", maxiter=BFGS_iters, options=options_BFGS)
    param_CBF, state = solver2.run(param_vec_adam, 
                                    bounds=(-100*np.ones(num_vars), 100*np.ones(num_vars)))
    
    value_adam, safe_adam, unsafe_adam, barrier_adam = objective(param_vec_adam)
    value_BFGS, safe_BFGS, unsafe_BFGS, barrier_BFGS = objective(param_CBF)
    print('Adam:', value_adam, 'BFGS:', value_BFGS)
    print('Safe:', safe_adam, 'Safe:', safe_BFGS)
    print('Unsafe:', unsafe_adam, 'Unsafe:', unsafe_BFGS)
    print('Barrier:', barrier_adam, 'Barrier:', barrier_BFGS)

    param_CBF = param_CBF.copy()
    params, params_K = extract_params(param_CBF, nx, nu, NN_params, K_params)
    h_safe = compute_h(X_safe, param_CBF)
    h_unsafe = compute_h(X_unsafe, param_CBF)
    CBF_loss, u, x_next, h_curr, h_next = compute_CBF(X_barrier, param_CBF)

    safe_loss = jnp.max(jnp.maximum(h_safe,0))
    unsafe_loss = jnp.max(jnp.maximum(lu-h_unsafe,0))
    cbf_loss = jnp.max(jnp.maximum(0, CBF_loss))
    violation_vector = jnp.hstack((safe_loss, unsafe_loss, cbf_loss))  
    print('Violation vector:', violation_vector) 

    return np.array(param_CBF).copy()
