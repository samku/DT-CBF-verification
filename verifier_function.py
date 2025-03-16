from imports import *
import jaxopt
from vector_to_params import extract_params
from jax_sysid.utils import lbfgs_options, vec_reshape


def verifier_function(dataset, BF_params, K_params, NN_params, activation, activation_K, param_vec, X_traj_all):
    
    #Extract data
    system = dataset['system']
    X_train = dataset['X_train']
    nx = system.nx
    nu = system.nu
    dt = system.dt

    #BF params
    alpha = BF_params['alpha']
    alpha = 0.2
    alpha_bar = 0.4

    lu = BF_params['lu']
    delta = BF_params['delta']
    delta = 0.01
    
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

    @jax.jit
    def integrator(x,u):
        k1 = system.f_x(x) + system.g_x(x)@u
        k2 = system.f_x(x + 0.5*dt*k1) + system.g_x(x + 0.5*dt*k1)@u
        k3 = system.f_x(x + 0.5*dt*k2) + system.g_x(x + 0.5*dt*k2)@u
        k4 = system.f_x(x + dt*k3) + system.g_x(x + dt*k3)@u
        x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    @jax.jit
    def CBF_x(x, alpha_ver, delta_ver):
        #Compute control input
        Lc = Lc_x(x)
        L = Lc[:nu*nL_K].reshape(nu,nL_K)
        Q = 0.5*(L @ L.T + (L @ L.T).T) + 1*jnp.eye(nu)
        c = Lc[nu*nL_K:]
        u = jnp.minimum(jnp.maximum(-c, system.u_lim[0]), system.u_lim[1])
        #Compute CBF loss
        h_curr = h_x(x)
        x_next = integrator(x,u)
        h_next = h_x(x_next)
        CBF_loss = h_next-(1.-alpha_ver)*h_curr+delta_ver
        return CBF_loss, u, x_next, h_curr, h_next
    
    @jax.jit
    def compute_CBF(x, alpha_ver=alpha, delta_ver=delta):
        CBF_loss, u, x_next, h_curr, h_next = jax.vmap(lambda xi: CBF_x(xi, alpha_ver, delta_ver))(x)
        return CBF_loss.reshape(1,-1)[0], u, x_next, h_curr.reshape(1,-1)[0], h_next.reshape(1,-1)[0]
    
    #Test
    x_lim = system.x_lim
    num_pt = np.array([2000,2000])
    axes = [np.linspace(x_lim[0, i], x_lim[1, i], num_pt[i]) for i in range(len(num_pt))]
    grid = np.meshgrid(*axes, indexing='ij')
    X_test = np.stack(grid, axis=-1).reshape(-1, len(num_pt))
    T_test = np.zeros((len(X_test), 3))
    for i in range(len(X_test)):
        T_test[i] = system.BF_indicator(X_test[i])
    
    X_unsafe_con = X_test[np.where(T_test[:, 1] == 1)]

    h_values = compute_h(X_test)
    id_safe_marked = np.where(h_values<=0)
    id_unsafe_marked = np.where(h_values>lu)
    X_test_safe = X_test[id_safe_marked]
    X_test_unsafe = X_test[id_unsafe_marked]

    if 0:
        #Forward reachable sets
        plt.scatter(X_unsafe_con[:, 0], X_unsafe_con[:, 1], s=10, c='r', label='Unsafe', alpha=0.2)
        plt.scatter(X_test_safe[:, 0], X_test_safe[:, 1], s=10, c='g', label='Safe Marked', alpha=0.2)
        X_next = X_test_safe
        for t in range(10):
            _, _, X_next, _, _ = compute_CBF(X_next)
            rand_color = np.random.rand(3,)
            plt.scatter(X_next[:, 0], X_next[:, 1], s=10, c=rand_color, alpha=0.2)
            plt.pause(0.001)
        plt.show()

    #Step 1: Compute gamma_0
    @jax.jit
    def h_gamma0(x):
        return h_x(x)[0]

    options_BFGS = lbfgs_options(iprint=5,iters=10000, lbfgs_tol=1.e-10, memory=100)
    solver = jaxopt.ScipyBoundedMinimize(
        fun=h_gamma0, tol=1.e-10, method="L-BFGS-B", maxiter=10000, options=options_BFGS)
    x_gamma0, state = solver.run(jnp.array([0.,0.]), bounds=(system.x_lim[0], system.x_lim[1]))  
    gamma_0 = h_gamma0(x_gamma0)  
    print('gamma_0:', gamma_0, 'Empirical:', jnp.min(h_values))
    gamma_0 = np.minimum(gamma_0, jnp.min(h_values))

    #Step 2: Estimate Lf and Lh
    x_lim = np.array([[-2.4,-2.8],[2.8,2.8]])
    num_pt = np.array([100,100])
    axes = [np.linspace(x_lim[0, i], x_lim[1, i], num_pt[i]) for i in range(len(num_pt))]
    grid = np.meshgrid(*axes, indexing='ij')
    X_Lip_F = np.array(np.stack(grid, axis=-1).reshape(-1, len(num_pt)))
    h_Lip_F = np.array(compute_h(X_Lip_F)).reshape(-1,1) 
    zero_h = np.where(h_Lip_F<=0)
    X_Lip = X_Lip_F[zero_h[0]]
    h_Lip = h_Lip_F[zero_h[0]]  
    plt.scatter(X_Lip[:,0], X_Lip[:,1],s=20, c='g') 
    _, _, Xp_Lip, _, _ = compute_CBF(X_Lip)

    N_Lip = X_Lip.shape[0]  
    Xp_diff = Xp_Lip[:, None, :] - Xp_Lip[None, :, :]
    X_diff  = X_Lip[:, None, :]  - X_Lip[None, :, :]
    h_diff  = h_Lip[:, None, :]  - h_Lip[None, :, :]
    Xp_norms = np.linalg.norm(Xp_diff, axis=2) 
    X_norms  = np.linalg.norm(X_diff, axis=2)
    h_norms  = np.linalg.norm(h_diff, axis=2)
    triu_idx = np.triu_indices(N_Lip, k=1)
    ratios_f = Xp_norms[triu_idx] / X_norms[triu_idx]
    ratios_h = h_norms[triu_idx] / X_norms[triu_idx]

    Lf = 1.1 * np.max(ratios_f)
    Lh = 1.1 * np.max(ratios_h)
    Lr = Lh * Lf + Lh

    print('Lf:', Lf, 'Lh:', Lh, 'Lr:', Lr)

    #For gamma and epsilon sequences
    gamma_hat = -(1-alpha_bar)*delta/(alpha*(1-alpha_bar)+alpha_bar*Lf)
    print('gamma_hat:', gamma_hat)

    id_ghat = np.where(h_Lip<=gamma_hat)[0]
    X_ghat = X_Lip[id_ghat]
    X_LB = 1.01*jnp.min(X_ghat, axis=0)
    X_UB = 1.01*jnp.max(X_ghat, axis=0)
    print('X_LB:', X_LB, 'X_UB:', X_UB)
    plt.scatter(X_ghat[:,0], X_ghat[:,1],s=20, c='r')
    plt.show()

    if 1:
        #Enable for gamma strategy using linear system
        a_seq = (1-alpha_bar)*(Lh*Lf+(1-alpha)*Lh)/(Lh*Lf+(1-alpha_bar)*Lh)
        b_seq = (alpha_bar-1)*Lh/(Lh*Lf+(1-alpha_bar)*Lh)*delta
        gamma_array = np.array([gamma_0])
        for i in range(10000):
            gamma_next = a_seq*gamma_array[-1]+b_seq
            gamma_array = np.append(gamma_array, gamma_next)
            if gamma_next>1.0001*gamma_hat:
                break
        print('gamma_array:', gamma_array)  
        plt.plot(gamma_array)
        plt.show()
        gamma_seq = gamma_array.copy()

    N_gamma = len(gamma_seq)
    epsilon_seq = (delta+(alpha_bar-alpha)*jnp.abs(gamma_seq[1:]))/(Lh*Lf+(1-alpha_bar)*Lh)
    print('epsilon_seq:', epsilon_seq)

    #Verification
    figure, [ax1, ax2] = plt.subplots(1, 2)
    tic = time.time()
    X_samples = []
    dh_violation_max = []
    num_points_total = []
    resolution = np.zeros(N_gamma-1)
    for i in range(N_gamma-1):
        gamma_minus = gamma_seq[i]
        gamma_plus = gamma_seq[i+1]
        epsilon_curr = epsilon_seq[i]
        resolution[i] = epsilon_curr/np.sqrt(nx)
        num_pt = np.zeros(nx, dtype=int)   
        for j in range(nx):
            num_pt[j] = int(np.ceil((X_UB[j]-X_LB[j])/resolution[i]))
        axes = [np.linspace(X_LB[j], X_UB[j], num_pt[j]) for j in range(len(num_pt))]
        grid = np.meshgrid(*axes, indexing='ij')
        X_samp_nR = np.array(np.stack(grid, axis=-1).reshape(-1, len(num_pt)))
        h_loc = compute_h(X_samp_nR)
        ids_curr = np.where(np.logical_and(h_loc <= gamma_plus, h_loc > gamma_minus))[0] 
        if len(ids_curr) > 0:
            X_samples.append(X_samp_nR[ids_curr])
        else:
            for j in range(nx):
                num_pt[j] = int(np.ceil((X_UB[j]-X_LB[j])/(0.5*resolution[i])))
            print('i:', i, 'num_pt:', num_pt,'resolution:', 0.5*resolution[i])#Half the size
            axes = [np.linspace(X_LB[j], X_UB[j], num_pt[j]) for j in range(len(num_pt))]
            grid = np.meshgrid(*axes, indexing='ij')
            X_samp_nR = np.array(np.stack(grid, axis=-1).reshape(-1, len(num_pt)))
            h_loc = compute_h(X_samp_nR)
            ids_curr = np.where(np.logical_and(h_loc <= gamma_plus, h_loc > gamma_minus))[0] 
            X_samples.append(X_samp_nR[ids_curr])

        dh_violation, _, _, _, _ = compute_CBF(X_samp_nR[ids_curr], alpha_ver = alpha, delta_ver = delta)
        dh_violation_max.append(np.max(dh_violation))
        h_loc = h_loc[ids_curr] 
        num_points_total.append(len(ids_curr))
        print('i:', i, 'num_pt_tot:', num_pt,'resolution:', resolution[i], 'num_points:', len(ids_curr), 'max_dh_violation:', np.max(dh_violation))
    toc = time.time()
    print('Time:', toc-tic)
    #plt.show()

    dh_violation_max = np.array(dh_violation_max)
    print('Points:', num_points_total)
    print('Sum:', np.sum(np.array(num_points_total)))
    print('dh_violation_max:', dh_violation_max)
    print('Max violation:', np.max(dh_violation_max))

    cmap = plt.cm.get_cmap('tab10', 11)  
    colors = [cmap(i) for i in range(len(X_samples))]


    if 1:
        from matplotlib import rc
        rc('text', usetex=True)
        figure, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 13.333),
        gridspec_kw={'height_ratios': [1, 3]}
        )
        plt.tight_layout()
        ax1.plot(gamma_seq, 'o-',linewidth=2)
        ax1.set_xlabel(r'$i$')
        ax1.set_ylabel(r'$\gamma_i$')
        ax1.set_xlim(0, N_gamma-1)
        ax1.grid(True)
        for i in range(len(X_samples)):
            color_rand = np.random.rand(3,) 
            plt.scatter(X_samples[i][:,0], X_samples[i][:,1], s=0.1, c=[colors[i]])
            plt.pause(0.001)

        for i in range(len(X_traj_all)):
            plt.plot(X_traj_all[i][:,0], X_traj_all[i][:,1], 'k-', alpha=0.2)
            #plt.pause(1)

        if 1:
            x_lim = system.x_lim
            num_pt = np.array([800,800])
            axes = [np.linspace(x_lim[0, i], x_lim[1, i], num_pt[i]) for i in range(len(num_pt))]
            grid = np.meshgrid(*axes, indexing='ij')
            X_test = np.stack(grid, axis=-1).reshape(-1, len(num_pt))
            T_test = np.zeros((X_test.shape[0], 3)) 
            for i in range(len(X_test)):
                T_test[i] = system.BF_indicator(X_test[i])
            id_safe = np.where(T_test[:, 0] == 1)
            id_unsafe = np.where(T_test[:, 1] == 1)
            ax2.scatter(X_test[id_unsafe, 0], X_test[id_unsafe, 1], s=1, c='k', label='Unsafe', alpha = 0.1)
        ax2.set_xlim(-3,3)
        ax2.set_ylim(-3,3)
        ax2.set_xlabel(r'$x_1$')
        ax2.set_ylabel(r'$x_2$')
        ax2.grid(True)


        plt.gca().set_rasterized(True)  # Rasterize the entire axes
        plt.savefig('fully_rasterized_plot.pdf', format='pdf', dpi=300)

        plt.show()
    
    


