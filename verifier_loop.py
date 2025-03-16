from imports import *
from vector_to_params import extract_params
import jaxopt
from jax_sysid.utils import lbfgs_options, vec_reshape
from generate_file_path import generate_file_path

def verifier_loop(dataset, BF_params, K_params, NN_params, activation, activation_K, param_vec, alpha_bar, base_filename, current_directory, regenerate_data=False):
    
    #Extract data
    system = dataset['system']
    nx = system.nx
    nu = system.nu
    dt = system.dt

    #BF params
    alpha = 0.2
    delta = 0.01
    file_name = 'results_alpha_bar_' + str(alpha_bar)
    file_path = generate_file_path(file_name, base_filename, current_directory)
    if not file_path.exists() or regenerate_data:

        lu = BF_params['lu']
        delta = BF_params['delta']
        
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
        num_pt = np.array([200,200])
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

        N_gammas = np.arange(35)+2

        num_points_total_all = np.zeros((len(N_gammas),))
        time_all = np.zeros((len(N_gammas),))
        for qqq in range(len(N_gammas)):
            print('N_gamma:', N_gammas[qqq], qqq)
            N_gamma = N_gammas[qqq]
            gamma_seq = np.linspace(gamma_0, gamma_hat, N_gamma)
            epsilon_seq = (delta+(alpha_bar-alpha)*jnp.abs(gamma_seq[1:]))/(Lh*Lf+(1-alpha_bar)*Lh)
            print('epsilon_seq:', epsilon_seq)

            #Verification
            start_time = time.time()
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
            end_time = time.time()
            time_all[qqq] = end_time-start_time
            print('Time:', end_time-start_time)
            
            dh_violation_max = np.array(dh_violation_max)
            print('Points:', num_points_total)
            print('Sum:', np.sum(np.array(num_points_total)))
            print('dh_violation_max:', dh_violation_max)
            print('Max violation:', np.max(dh_violation_max))
            num_points_total_all[qqq] = np.sum(num_points_total)
        
        print(time_all)
        print(num_points_total_all)
        dict_results = {'N_gammas': N_gammas, 'num_points_total': num_points_total_all, 'time_all': time_all, 'alpha_bar': alpha_bar}
        with open(file_path, 'wb') as f:
            pickle.dump(dict_results, f)

    else:
        with open(file_path, 'rb') as f:
            CBF_dict = pickle.load(f)
            N_gammas = CBF_dict['N_gammas']
            num_points_total = CBF_dict['num_points_total']
            time_all = CBF_dict['time_all']
            alpha_bar = CBF_dict['alpha_bar']
            figure, [ax1, ax2] = plt.subplots(1, 2) 
            ax1.plot(num_points_total)
            ax2.plot(time_all)
            plt.show()
            print('N_gammas:', N_gammas)
            print('num_points_total:', num_points_total)
            print('time_all:', time_all)
            print('alpha_bar:', alpha_bar)


    
    


