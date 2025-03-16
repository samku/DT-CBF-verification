def extract_params(total_vec, nx, nu, NN_params, K_params):
    nH = NN_params['nH']
    nth = NN_params['nth']
    nL_K = K_params['nL_K']
    nH_K = K_params['nH_K']
    nth_K = K_params['nth_K']
    
    # Calculate sizes for NN
    size_Win = nth * nx
    size_bin = nth
    size_Whid = (nH - 1) * nth * nth
    size_bhid = (nH - 1) * nth
    size_Wout = nth
    size_bout = 1

    # Calculate sizes for K
    size_Win_K = nth_K * nx
    size_bin_K = nth_K
    size_Whid_K = (nH_K - 1) * nth_K * nth_K
    size_bhid_K = (nH_K - 1) * nth_K
    size_Wout_K = (nu * nL_K + nu) * nth_K
    size_bout_K = (nu * nL_K + nu)

    idx = 0

    # Extract and reshape NN parameters
    Win = total_vec[idx:idx + size_Win].reshape(nth, nx)
    idx += size_Win
    
    bin = total_vec[idx:idx + size_bin].reshape(nth,)
    idx += size_bin

    Whid = total_vec[idx:idx + size_Whid].reshape(nH - 1, nth, nth)
    idx += size_Whid

    bhid = total_vec[idx:idx + size_bhid].reshape(nH - 1, nth)
    idx += size_bhid

    Wout = total_vec[idx:idx + size_Wout].reshape(1, nth)
    idx += size_Wout

    bout = total_vec[idx:idx + size_bout].reshape(1,)
    idx += size_bout

    # Extract and reshape K parameters
    Win_K = total_vec[idx:idx + size_Win_K].reshape(nth_K, nx)
    idx += size_Win_K
    
    bin_K = total_vec[idx:idx + size_bin_K].reshape(nth_K,)
    idx += size_bin_K

    Whid_K = total_vec[idx:idx + size_Whid_K].reshape(nH_K - 1, nth_K, nth_K)
    idx += size_Whid_K

    bhid_K = total_vec[idx:idx + size_bhid_K].reshape(nH_K - 1, nth_K)
    idx += size_bhid_K

    Wout_K = total_vec[idx:idx + size_Wout_K].reshape(nu * nL_K + nu, nth_K)
    idx += size_Wout_K

    bout_K = total_vec[idx:idx + size_bout_K].reshape(nu * nL_K + nu,)
    idx += size_bout_K

    return [Win, bin, Whid, bhid, Wout, bout], [Win_K, bin_K, Whid_K, bhid_K, Wout_K, bout_K]