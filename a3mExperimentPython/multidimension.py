import argparse
import numpy as np
import random
import pandas as pd

from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage

from utils import histogram_RR, denoise_histogram_RR, histogram_to_freq 
from utils import compute_gaussian_sigma, duchi_algo, piecewise_algo, hybrid_algo
from a3m import opt_variance, a3m_perturb
from utils import sw

def read_data(acs_data, task, beta):
    if task == 0:
        inc_features, inc_labels, _ = ACSIncome.df_to_numpy(acs_data)
        inc_data = np.vstack([inc_features.T,inc_labels.T]).T
        inc_data = (inc_data - inc_data.min(0)) / inc_data.ptp(0)
        inc_data = 2 * beta * inc_data - beta
        return inc_data
    elif task == 1:
        emp_features, emp_labels, _ = ACSEmployment.df_to_numpy(acs_data)
        emp_data = np.vstack([emp_features.T,emp_labels.T]).T
        emp_data = (emp_data - emp_data.min(0)) / emp_data.ptp(0)
        emp_data = 2 * beta * emp_data - beta
        return emp_data
    else:
        pc_features, pc_labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
        pc_data = np.vstack([pc_features.T,pc_labels.T]).T
        pc_data = (pc_data - pc_data.min(0)) / pc_data.ptp(0)
        pc_data = 2 * beta * pc_data - beta
        return pc_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    # range for DATA
    parser.add_argument("--beta", help="range for data", type=float, default=1)
    # Privacy
    parser.add_argument("--delta", help="privacy constraint", type=float, default=0.00001)
    # independent runs
    parser.add_argument("--runs", help="independent runs", type=int, default=1000) 
    # gaussian
    parser.add_argument("--step_start", type=int, default=0)
    parser.add_argument("--step_end", type=int, default=300000)
    parser.add_argument("--step_chi", type=float, default=0.00001)
    parser.add_argument("--prec", help="relative prec", type=float, default=0.0001)
    # a3m
    parser.add_argument("--bin_size", help="bin length", type=float, default=0.5)
    parser.add_argument("--axRatio", help="ratio between amax/xmax", type=float, default=4)
    parser.add_argument("--s", help="split ratio", type=float, default=0.1)
    # task
    parser.add_argument("--task", type=int, default=0) # 0 is income, 1 is employment, 2 is public coverage
    # number of sampled dimension
    parser.add_argument("--sample_dim", type=int, default=1) # number of sampled dimensions for each user

    
    args = parser.parse_args()
    print(args)
    
    # fix seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    epsilon_array = np.linspace(1, 8, 8)

    error_laplace = np.zeros(epsilon_array.size)
    error_gaussian = np.zeros(epsilon_array.size)
    error_duchi = np.zeros(epsilon_array.size)
    error_piecewise = np.zeros(epsilon_array.size)
    error_hybrid = np.zeros(epsilon_array.size)
    error_a3m_pure = np.zeros(epsilon_array.size)
    error_a3m_app = np.zeros(epsilon_array.size)
    # error_sw = np.zeros(epsilon_array.size)

    
    """ 
        read data 
    """
    print('Reading data =====>')
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir="data")
    acs_data = data_source.get_data(states=state_list, download=True)
    data_all = read_data(acs_data, args.task, args.beta)
    print('Finish reading.')
    
    total_samples =  data_all.shape[0] # number of samples
    K = data_all.shape[1] # number of dimension

    split_ratio = args.s # proportion of data for frequency estimation for 3am
    for k in range(K):
        print(f'{k}-th dimension')
        data_k = data_all[:,k] # k-th dimension
        true_mean = np.sum(data_k) / total_samples

        # subsample sample_dim/K fraction for estimation
        n = args.sample_dim * int(total_samples/K)
        data = data_k[np.random.choice(total_samples, n, replace=False)]

        # split for 3am
        data_1 = data[0:int(split_ratio*n)]
        data_2 = data[int(split_ratio*n):n]
        
        
        for i in range(epsilon_array.size):
            epsilon = epsilon_array[i] * 1. / args.sample_dim
            # setup Gaussian noise
            sigma, found = compute_gaussian_sigma(args.beta, epsilon, args.delta, args.step_start, args.step_end,
                                        args.step_chi, args.prec)
            if found == 0:
                print('sigma not found for Gaussian')
            for run in range(args.runs):
                """ 
                    laplace 
                """
                laplace_scale = 2 * args.beta / epsilon
                laplace_noise = np.random.laplace(loc=np.zeros(n),scale=laplace_scale)
                laplace_data = data + laplace_noise
                laplace_mean = np.sum(laplace_data) / n
                error_laplace[i] += (true_mean - laplace_mean) ** 2 / args.runs
                """ 
                    gaussian
                """
                if found == 1:
                    gaussian_noise = np.random.normal(loc=np.zeros(n),scale=sigma)
                    gaussian_data = data + gaussian_noise
                    gaussian_mean = np.sum(gaussian_data) / n
                    error_gaussian[i] += (true_mean - gaussian_mean) ** 2 / args.runs    
                """ 
                    duchi takes input from [-1,1]
                """
                duchi_output = duchi_algo(data/args.beta, epsilon)
                duchi_data = args.beta * duchi_output 
                duchi_mean = np.sum(duchi_data) / n
                error_duchi[i] += (true_mean - duchi_mean) ** 2 / args.runs
                """ 
                    piecewise takes input from [-1,1]
                """
                piecewise_output = piecewise_algo(data/args.beta, epsilon)
                piecewise_data = args.beta * piecewise_output 
                piecewise_mean = np.sum(piecewise_data) / n
                error_piecewise[i] += (true_mean - piecewise_mean) ** 2 / args.runs
                """ 
                    hybrid takes input in [-1,1]
                """
                hybrid_outcome = hybrid_algo(data/args.beta, epsilon)
                hybrid_data = args.beta * hybrid_outcome
                hybrid_mean = np.sum(hybrid_data) / n
                error_hybrid[i] += (true_mean - hybrid_mean) ** 2 / args.runs
                # """
                #     sw
                # """
                # sw_bins = 1024
                # sw_bin_size = 2*args.beta/sw_bins
                # sw_outcome = sw(data, -args.beta, args.beta, epsilon, sw_bins, sw_bins)
                # sw_centers = np.linspace(-args.beta+sw_bin_size/2, args.beta-sw_bin_size/2, sw_bins)
                # sw_mean = np.inner(np.array(sw_outcome), sw_centers) / n
                # error_sw[i] += (true_mean - sw_mean) ** 2 / args.runs
                """ 
                    a3m pure and app
                """
                # compute noisy histogram with randomize response
                true_histogram_1, noisy_histogram_1 = histogram_RR(data_1, -args.beta, args.beta, args.bin_size, epsilon)    
                true_histogram_2, noisy_histogram_2 = histogram_RR(data_2, -args.beta, args.beta, args.bin_size, epsilon)    
                # convert to frequency
                true_freq = histogram_to_freq(true_histogram_2, -args.beta, args.beta, args.bin_size)
                noisy_freq = histogram_to_freq(noisy_histogram_1, -args.beta, args.beta, args.bin_size)
                # denoise the histogram and convert to frequency
                estimated_freq = denoise_histogram_RR(noisy_histogram_1, -args.beta, args.beta, args.bin_size, epsilon)
                """ pure """
                # use estimated freq to generate a3m noise
                noise_values, opt_distribution_pure = opt_variance(estimated_freq, args.beta, args.bin_size, args.axRatio, epsilon, 0)
                # perturb with a3m
                a3m_noise_pure = a3m_perturb(true_histogram_2, args.beta, args.bin_size, noise_values, opt_distribution_pure)
                error_a3m_pure[i] += np.power(np.sum(a3m_noise_pure) / (n-int(split_ratio*n)), 2) / args.runs
                """ app """
                # use estimated freq to generate a3m noise
                noise_values, opt_distribution_app = opt_variance(estimated_freq, args.beta, args.bin_size, args.axRatio, epsilon, args.delta)
                # perturb with a3m
                a3m_noise_app = a3m_perturb(true_histogram_2, args.beta, args.bin_size, noise_values, opt_distribution_app)
                error_a3m_app[i] += np.power(np.sum(a3m_noise_app) / (n-int(split_ratio*n)), 2) / args.runs
            print(f'Epsilon: {epsilon_array[i]} finished')
            print(f'Laplace scale:{laplace_scale}, error: {error_laplace[i]}')
            print(f'Gaussian sigma:{sigma}, error: {error_gaussian[i]}')
            print(f'Duchi\'s error: {error_duchi[i]}')
            print(f'Piecewise error: {error_piecewise[i]}')
            print(f'Hybrid error: {error_hybrid[i]}')
            # print(f'SW error: {error_sw[i]}')
            print(f'Pure-a3m error: {error_a3m_pure[i]}')
            print(f'App-a3m error: {error_a3m_app[i]}\n')
    
    print(f'On Green Taxi Data with beta={args.beta} and bin_size={args.bin_size},averaged over {args.runs} runs')
    print(f'Laplace error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_laplace[i])
    print('\n')
    print(f'Gaussian error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_gaussian[i])
    print('\n')
    print(f'Duchi error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_duchi[i])
    print('\n')
    print(f'Piecewise error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_piecewise[i])
    print('\n')
    print(f'Hybrid error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_hybrid[i])
    print('\n')
    # print(f'SW error:')
    # for i in range(epsilon_array.size):
    #     print(epsilon_array[i], error_sw[i])
    # print('\n')
    print(f'Pure-A3M error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_a3m_pure[i])
    print('\n')
    print(f'App-A3M error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_a3m_app[i])
    print('\n')