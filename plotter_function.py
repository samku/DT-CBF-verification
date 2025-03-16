from imports import *
from vector_to_params import extract_params
from jaxopt import BoxCDQP


def plotter_function(dataset, BF_params, K_params, NN_params, activation, activation_K, param_vec):
    
    #Extract data
    system = dataset['system']
    X_train = dataset['X_train']
    T_train = dataset['T_train']
    nx = system.nx
    nu = system.nu
    dt = system.dt

    #BF params
    alpha = BF_params['alpha']
    lu = BF_params['lu']
    delta = BF_params['delta']

    alpha = 0.7
    delta = 0.
    
    #NN params
    nH = NN_params['nH']
    nth = NN_params['nth']

    #K params
    nL_K = K_params['nL_K']
    nH_K = K_params['nH_K']
    nth_K = K_params['nth_K']

    #Initialize
    num_vars = len(param_vec)
    params, params_K = extract_params(param_vec, nx, nu, NN_params, K_params)

    jax.config.update("jax_enable_x64", True)
    #Make network
    @jax.jit
    def h_x(x):
        Win, bin, Whid, bhid, Wout, bout = params
        post_linear = Win@x+bin
        post_activation = activation(post_linear)
        for j in range(nH-1): 
            post_linear = Whid[j]@post_activation+bhid[j]
            post_activation = activation(post_linear)
        post_linear = Wout@post_activation+bout
        return post_linear
    
    @jax.jit
    def compute_h(x):
        return jax.vmap(lambda xi: h_x(xi))(x).reshape(1,-1)[0]   
    
    @jax.jit
    def Lc_x(x):
        Win_K, bin_K, Whid_K, bhid_K, Wout_K, bout_K = params_K
        post_linear = Win_K@x+bin_K
        post_activation = activation_K(post_linear)
        for j in range(nH_K-1): 
            post_linear = Whid_K[j]@post_activation+bhid_K[j]
            post_activation = activation_K(post_linear)
        post_linear = Wout_K@post_activation+bout_K
        return post_linear

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

    @jax.jit
    def integrator(x,u):
        k1 = system.f_x(x) + system.g_x(x)@u
        k2 = system.f_x(x + 0.5*dt*k1) + system.g_x(x + 0.5*dt*k1)@u
        k3 = system.f_x(x + 0.5*dt*k2) + system.g_x(x + 0.5*dt*k2)@u
        k4 = system.f_x(x + dt*k3) + system.g_x(x + dt*k3)@u
        x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    @jax.jit
    def CBF_x(x):
        #Compute control input
        Lc = Lc_x(x)
        L = Lc[:nu*nL_K].reshape(nu,nL_K)
        Q = 0.5*(L @ L.T + (L @ L.T).T) + 1*jnp.eye(nu)
        c = Lc[nu*nL_K:]
        u = qp_layer(Q,c)
        #u = qpax.solve_qp_primal(Q, c, A_eq,b_eq, A_ineq, b_ineq, solver_tol=1e-8, target_kappa=1e-8)
        #Compute CBF loss
        h_curr = h_x(x)
        x_next = integrator(x,u)
        h_next = h_x(x_next)
        CBF_loss = h_next-(1.-alpha)*h_curr+delta
        return CBF_loss, u, x_next, h_curr, h_next
    
    @jax.jit
    def compute_CBF(x):
        CBF_loss, u, x_next, h_curr, h_next = jax.vmap(lambda xi: CBF_x(xi))(x)
        return CBF_loss.reshape(1,-1)[0], u, x_next, h_curr.reshape(1,-1)[0], h_next.reshape(1,-1)[0]
    
    h_values = compute_h(X_train)
    id_safe_marked = np.where(h_values<=0)
    id_safe_ideal = np.where(T_train[:, 0] == 1)   
    id_unsafe_marked = np.where(h_values>lu)
    id_unsafe_ideal = np.where(T_train[:, 1] == 1)

    plt.scatter(X_train[id_safe_ideal, 0], X_train[id_safe_ideal, 1], s=30, c='b', label='Safe Ideal', alpha=0.2)
    plt.scatter(X_train[id_safe_marked, 0], X_train[id_safe_marked, 1], s=10, c='g', label='Safe Marked', alpha=0.2)
    #plt.scatter(X_train[id_unsafe_ideal, 0], X_train[id_unsafe_ideal, 1], s=30, c='r', label='Unsafe Ideal')
    #plt.scatter(X_train[id_unsafe_marked, 0], X_train[id_unsafe_marked, 1], s=10, c='y', label='Unsafe')
    plt.show()

    #Test
    x_lim = system.x_lim
    num_pt = np.array([300,300])
    axes = [np.linspace(x_lim[0, i], x_lim[1, i], num_pt[i]) for i in range(len(num_pt))]
    grid = np.meshgrid(*axes, indexing='ij')
    X_test = np.stack(grid, axis=-1).reshape(-1, len(num_pt))
    h_values = compute_h(X_test)
    id_safe_marked = np.where(h_values<=0)
    id_unsafe_marked = np.where(h_values>lu)

    from matplotlib import rc

    rc('text', usetex=True)
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.scatter(X_test[id_safe_marked, 0], X_test[id_safe_marked, 1], s=10, c='g', label='Safe Marked', alpha=0.2)
    N_sim = 100
    n_pt_sim = 100
    id_random = np.random.randint(0, len(id_safe_marked[0]), n_pt_sim)
    x0_random = X_test[id_safe_marked[0][id_random], :]
    ax1.scatter(x0_random[:, 0], x0_random[:, 1], s=30, c='r', label='Initial Condition')
    X_traj_all = []
    h_vals_all = []
    max_h_violation = []
    for idx in range(n_pt_sim):
        X_traj = np.zeros((N_sim, nx))
        U_traj = np.zeros((N_sim-1, nu))
        X_traj[0] = x0_random[idx]
        for t in range(N_sim-1):
            _, u, _, _, _ = CBF_x(np.array(X_traj[t]))
            U_traj[t] = u
            X_traj[t+1] = integrator(X_traj[t], U_traj[t])
        ax1.plot(X_traj[:, 0], X_traj[:, 1], 'b-')
        h_vals = compute_h(X_traj)
        max_h_violation.append(np.max(h_vals[1:]-(1-alpha)*h_vals[:-1]+delta))
        if idx <= 3:
            h_values_traj = compute_h(X_traj)
            color_rand = np.random.rand(3,)
            ax2.plot(h_values_traj[1:], color=color_rand)
            ax2.plot((1-alpha)*h_values_traj[:-1]+delta, color=color_rand, linestyle='--')
            ax3.plot(U_traj, color=color_rand)
        #plt.pause(1)
        X_traj_all.append(X_traj)   
        h_vals_all.append(h_values_traj)

    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax2.set_xlabel(r'$t$')
    ax3.set_xlabel(r'$t$')
    ax2.set_xlim(0, N_sim)
    ax3.set_xlim(0, N_sim)

    plt.show()

    max_h_violation = np.array(max_h_violation)
    plt.plot(max_h_violation)
    plt.show()
    return X_traj_all, h_vals_all