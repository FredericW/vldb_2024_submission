import argparse
import numpy as np
import random
import time

from synthetic_generate import generate_synthetic_data
from utils import histogram_RR, denoise_histogram_RR, histogram_to_freq 
from utils import compute_gaussian_sigma, duchi_algo, piecewise_algo, hybrid_algo
from a3m import opt_variance, a3m_perturb

"""
Example:
    python synthetic_exp.py --data_type=GAUSSIAN --n=10000
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    # Synthetic DATA
    parser.add_argument("--data_type", help="which data to use", type=str, default="GAUSSIAN")
    parser.add_argument("--n", help="overall number of data points", type=int, default=10000)
    parser.add_argument("--low", help="lower limit for clipping", type=float, default=-5)
    parser.add_argument("--high", help="upper limit for clipping", type=float, default=5)
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
    
    for i in range(epsilon_array.size):
        epsilon = epsilon_array[i]
        # setup Gaussian noise
        sigma, found = compute_gaussian_sigma(args.beta, epsilon, args.delta, args.step_start, args.step_end,
                                    args.step_chi, args.prec)
        if found == 0:
            print('sigma not found for Gaussian')
        for run in range(args.runs):
            """ 
                generate data in [-beta,beta] 
            """
            split_ratio = args.s # proportion of data for frequency estimation for 3am
            data = generate_synthetic_data(args.data_type, args.n, args.low, args.high, args.beta)
            data_1 = data[0:int(split_ratio*args.n)]
            data_2 = data[int(split_ratio*args.n):args.n]
            true_mean = np.sum(data) / args.n
            """ 
                laplace 
            """
            laplace_scale = 2 * args.beta / epsilon
            laplace_noise = np.random.laplace(loc=np.zeros(args.n),scale=laplace_scale)
            laplace_data = data + laplace_noise
            laplace_mean = np.sum(laplace_data) / args.n
            error_laplace[i] += (true_mean - laplace_mean) ** 2 / args.runs
            """ 
                gaussian
            """
            if found == 1:
                gaussian_noise = np.random.normal(loc=np.zeros(args.n),scale=sigma)
                gaussian_data = data + gaussian_noise
                gaussian_mean = np.sum(gaussian_data) / args.n
                error_gaussian[i] += (true_mean - gaussian_mean) ** 2 / args.runs    
            """ 
                duchi takes input from [-1,1]
            """
            duchi_output = duchi_algo(data/args.beta, epsilon)
            duchi_data = args.beta * duchi_output 
            duchi_mean = np.sum(duchi_data) / args.n
            error_duchi[i] += (true_mean - duchi_mean) ** 2 / args.runs
            """ 
                piecewise takes input from [-1,1]
            """
            piecewise_output = piecewise_algo(data/args.beta, epsilon)
            piecewise_data = args.beta * piecewise_output 
            piecewise_mean = np.sum(piecewise_data) / args.n
            error_piecewise[i] += (true_mean - piecewise_mean) ** 2 / args.runs
            """ 
                hybrid takes input in [-1,1]
            """
            hybrid_outcome = hybrid_algo(data/args.beta, epsilon)
            hybrid_data = args.beta * hybrid_outcome
            hybrid_mean = np.sum(hybrid_data) / args.n
            error_hybrid[i] += (true_mean - hybrid_mean) ** 2 / args.runs
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
            error_a3m_pure[i] += np.power(np.sum(a3m_noise_pure) / (args.n-int(split_ratio*args.n)), 2) / args.runs
            """ app """
            # use estimated freq to generate a3m noise
            noise_values, opt_distribution_app = opt_variance(estimated_freq, args.beta, args.bin_size, args.axRatio, epsilon, args.delta)
            # perturb with a3m
            a3m_noise_app = a3m_perturb(true_histogram_2, args.beta, args.bin_size, noise_values, opt_distribution_app)
            error_a3m_app[i] += np.power(np.sum(a3m_noise_app) / (args.n-int(split_ratio*args.n)), 2) / args.runs
        print(f'Epsilon: {epsilon} finished')
        print(f'Laplace scale:{laplace_scale}, error: {error_laplace[i]}')
        print(f'Gaussian sigma:{sigma}, error: {error_gaussian[i]}')
        print(f'Duchi\'s error: {error_duchi[i]}')
        print(f'Piecewise error: {error_piecewise[i]}')
        print(f'Hybrid error: {error_hybrid[i]}')
        print(f'Pure-a3m error: {error_a3m_pure[i]}')
        print(f'App-a3m error: {error_a3m_app[i]}\n')
    
    print(f'On {args.data_type} with {args.n} data points of low={args.low} and high={args.high} and beta={args.beta} and bin_size={args.bin_size},averaged over {args.runs} runs')
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
    print(f'Pure-A3M error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_a3m_pure[i])
    print('\n')
    print(f'App-A3M error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_a3m_app[i])
    print('\n')
   

            
    
    