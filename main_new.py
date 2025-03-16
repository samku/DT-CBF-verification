from imports import *
from generate_file_path import generate_file_path
from dynamics.VanderpolOscl_trigonometric import VanderpolOscl_trigonometric 
from CBF_Step1 import CBF_Step1
from CBF_Step2 import CBF_Step2
from plotter_function import plotter_function   
from verifier_function import verifier_function


base_filename = 'VanderpolOscl_trigonometric'
#################################
#Training data
regenerate_data = False
file_path = generate_file_path('Step1', base_filename, current_directory)
if not file_path.exists() or regenerate_data:
    system = VanderpolOscl_trigonometric()
    x_lim = system.x_lim
    nx = system.nx
    nu = system.nu
    dt = system.dt
    num_pt = np.array([100,100])
    axes = [np.linspace(x_lim[0, i], x_lim[1, i], num_pt[i]) for i in range(len(num_pt))]
    grid = np.meshgrid(*axes, indexing='ij')
    X_train = np.stack(grid, axis=-1).reshape(-1, len(num_pt))
    N_train = X_train.shape[0]
    T_train = np.zeros((N_train, 3))
    for i in range(N_train):
        T_train[i] = system.BF_indicator(X_train[i])
    
    #Add refined grid near constraint boundaries
    x_lim_refined = np.array([[-1.3,-1.3],[1.3,1.3]])
    num_pt_refined = np.array([100,100])
    axes_refined = [np.linspace(x_lim_refined[0, i], x_lim_refined[1, i], num_pt_refined[i]) for i in range(nx)]
    grid_refined = np.meshgrid(*axes_refined, indexing='ij')
    X_train_refined = np.stack(grid_refined, axis=-1).reshape(-1, nx)
    N_train_refined = X_train_refined.shape[0]
    T_train_refined = np.zeros((N_train_refined, 3))
    for i in range(N_train_refined):
        T_train_refined[i] = system.BF_indicator(X_train_refined[i])
    
    N_train = N_train+N_train_refined
    X_train = np.vstack((X_train, X_train_refined))
    T_train = np.vstack((T_train, T_train_refined))

    safe_pt = np.where(T_train[:, 0] == 1)
    unsafe_pt = np.where(T_train[:, 1] == 1)
    BF_pt = np.where(T_train[:, 2] == 1) 

    print('N_train:', N_train)  
    N_train = X_train.shape[0]
    print('N_train:', N_train)
    dataset = {'system': system, 'X_train': X_train, 'T_train': T_train}
    ID_dict = {'dataset': dataset}
    with open(file_path, 'wb') as f:
        pickle.dump(ID_dict, f)
else:
    with open(file_path, 'rb') as f:
        ID_dict = pickle.load(f)
        dataset = ID_dict['dataset']
        system = dataset['system']
        X_train = dataset['X_train']
        T_train = dataset['T_train']
        nx = system.nx
        nu = system.nu
        dt = system.dt

print('X shape:', X_train.shape)
print('T shape:', T_train.shape)
#################################

if 1:
    id_safe = np.where(T_train[:, 0] == 1)
    id_unsafe = np.where(T_train[:, 1] == 1)
    id_barrier = np.where(T_train[:, 2] == 1)
    plt.scatter(X_train[id_barrier, 0], X_train[id_barrier, 1], s=20, c='b', label='Barrier')
    plt.scatter(X_train[id_safe, 0], X_train[id_safe, 1], s=10, c='g', label='Safe')
    plt.scatter(X_train[id_unsafe, 0], X_train[id_unsafe, 1], s=30, c='r', label='Unsafe')
    plt.show()

#################################
#CBF_identification
regenerate_data = False
activation = nn.tanh
activation_K = nn.tanh
file_path = generate_file_path('Step2', base_filename, current_directory)
if not file_path.exists() or regenerate_data:

    #BF params
    alpha = 0.01
    lu = 0.1
    delta = 0.01
    BF_params = {'alpha': alpha, 'lu': lu, 'delta': delta}

    #Controller params
    nL_K = 0 #Q(x) = L(x)L(x)^T + eye(nu), L(x) \in R^{nu \times nL_K}
    nH_K = 1 #Number of layers
    nth_K = 10 #Number of neurons per layer
    K_params = {'nL_K': nL_K, 'nH_K': nH_K, 'nth_K': nth_K}

    #NN params
    nH = 2 #Number of layers
    nth = 10 #Number of neurons per layer
    NN_params = {'nH': nH, 'nth': nth}

    #Training params
    adam_iters = 8000
    BFGS_iters = 2000
    penalty_safe = 1.
    penalty_unsafe = 2.
    penalty_barrier = 2.
    regularization = 0.0001 #ell_1 regularization

    train_params = {'BFGS_iters':BFGS_iters, 'adam_iters':adam_iters,
                    'penalty_safe': penalty_safe, 'penalty_unsafe': penalty_unsafe, 'penalty_barrier': penalty_barrier,
                    'regularization': regularization}

    #Training
    param_vec_warmstart = CBF_Step1(dataset, BF_params, K_params, NN_params, train_params, activation)
    CBF_dict = {'BF_params': BF_params, 'K_params': K_params, 'NN_params': NN_params, 'train_params': train_params, 
                'param_vec_warmstart': param_vec_warmstart}
    
    with open(file_path, 'wb') as f:
        pickle.dump(CBF_dict, f)
else:
    with open(file_path, 'rb') as f:
        CBF_dict = pickle.load(f)
        BF_params = CBF_dict['BF_params']
        K_params = CBF_dict['K_params']
        NN_params = CBF_dict['NN_params']
        train_params = CBF_dict['train_params']
        param_vec_warmstart = CBF_dict['param_vec_warmstart']

regenerate_data = False
file_path = generate_file_path('Step3', base_filename, current_directory)
if not file_path.exists() or regenerate_data:
    train_params['adam_iters'] = 10000
    train_params['BFGS_iters'] = 10000
    train_params['regularization'] = 0.00025
    train_params['regularization_grad'] = 0
    train_params['penalty_safe'] = 1.
    train_params['penalty_unsafe'] = 2.
    train_params['penalty_barrier'] = 10.
    train_params['sim_length'] = 1
    param_CBF = CBF_Step2(dataset, BF_params, K_params, NN_params, train_params, activation, activation_K, param_vec_warmstart)
    CBF_dict['param_CBF'] = param_CBF   
    with open(file_path, 'wb') as f:
        pickle.dump(CBF_dict, f)
else:
    with open(file_path, 'rb') as f:
        CBF_dict = pickle.load(f)
        BF_params = CBF_dict['BF_params']
        K_params = CBF_dict['K_params']
        NN_params = CBF_dict['NN_params']
        train_params = CBF_dict['train_params']
        param_vec_warmstart = CBF_dict['param_vec_warmstart']
        param_CBF = CBF_dict['param_CBF']


#################################
X_traj_all, h_vals_all = plotter_function(dataset, BF_params, K_params, NN_params, activation, activation_K, param_CBF)
#X_traj_all = []
verifier_function(dataset, BF_params, K_params, NN_params, activation, activation_K, param_CBF, X_traj_all)

#from verifier_loop import verifier_loop
#verifier_loop(dataset, BF_params, K_params, NN_params, activation, activation_K, param_CBF, 0.8, base_filename, current_directory,regenerate_data=True)


