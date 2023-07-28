import argparse
import numpy as np
import random
import time

from synthetic_generate import generate_synthetic_data
from utils import histogram_RR, denoise_histogram_RR, histogram_to_freq
from a3m import opt_variance, a3m_perturb

"""
Example:
    python vary_split_synthetic.py --data_type=GAUSSIAN --n=10000 --seed=2 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument("--seed", help="random seed", type=int, default=2)
    # Synthetic DATA
    parser.add_argument("--data_type", help="which data to use", type=str, default="GAUSSIAN")
    parser.add_argument("--n", help="overall number of data points", type=int, default=10000)
    parser.add_argument("--low", help="lower limit for clipping", type=float, default=-5)
    parser.add_argument("--high", help="upper limit for clipping", type=float, default=5)
    # range for DATA
    parser.add_argument("--beta", help="range for data", type=float, default=1)
    # bin size
    parser.add_argument("--bin_size", help="bin_size for input discretization", type=float, default=0.5)
    # number of bins is 2*beta/bin_size
    # Privacy
    parser.add_argument("--delta", help="privacy constraint", type=float, default=0.00001)
    # independent runs
    parser.add_argument("--runs", help="independent runs", type=int, default=2000) 
    
    args = parser.parse_args()
    print(args)
    
    # fix seed
    np.random.seed(args.seed)
    random.seed(args.seed)

  
    # epsilon_array = np.linspace(1, 8, 8)
    epsilon_array = np.array([1,2,4,8])
    # splits = np.array([0.4,0.8])
    splits = np.array([0.001,0.01,0.1,0.2,0.4])
    bin_size = args.bin_size
    axRatio = 4
    error_a3m_pure = np.zeros((splits.size,epsilon_array.size))
    error_a3m_app = np.zeros((splits.size,epsilon_array.size))

    for j in range(splits.size):
        split_ratio = splits[j]   
        for run in range(args.runs):
            for i in range(epsilon_array.size):
                epsilon = epsilon_array[i]  
                """ 
                generate data in [-beta,beta] 
                """
                data = generate_synthetic_data(args.data_type, args.n, args.low, args.high, args.beta)
                data_1 = data[0:int(split_ratio*args.n)]
                data_2 = data[int(split_ratio*args.n):args.n]
                true_mean = np.sum(data) / args.n
                """ 
                a3m pure 
                """
                # compute noisy histogram with randomize response
                true_histogram_1, noisy_histogram_1 = histogram_RR(data_1, -args.beta, args.beta, bin_size, epsilon)    
                true_histogram_2, noisy_histogram_2 = histogram_RR(data_2, -args.beta, args.beta, bin_size, epsilon)    
                # convert to frequency
                true_freq = histogram_to_freq(true_histogram_2, -args.beta, args.beta, bin_size)
                noisy_freq = histogram_to_freq(noisy_histogram_1, -args.beta, args.beta, bin_size)
                # denoise the histogram and convert to frequency
                estimated_freq = denoise_histogram_RR(noisy_histogram_1, -args.beta, args.beta, bin_size, epsilon)
                # use estimated freq to generate a3m noise
                noise_values, opt_distribution_pure = opt_variance(estimated_freq, args.beta, bin_size, axRatio, epsilon, 0)
                # perturb with a3m
                a3m_noise_pure = a3m_perturb(true_histogram_2, args.beta, bin_size, noise_values, opt_distribution_pure)
                error_a3m_pure[j][i] += np.power(np.sum(a3m_noise_pure) / (args.n-int(split_ratio*args.n)), 2) / args.runs
                # perturb with a3m app
                # use estimated freq to generate a3m noise
                noise_values, opt_distribution_app = opt_variance(estimated_freq, args.beta, bin_size, axRatio, epsilon, args.delta)
                # perturb with a3m
                a3m_noise_app = a3m_perturb(true_histogram_2, args.beta, bin_size, noise_values, opt_distribution_app)
                error_a3m_app[j][i] += np.power(np.sum(a3m_noise_app) / (args.n-int(split_ratio*args.n)), 2) / args.runs
        print(f'Split: {split_ratio} finished')
        print(f'Pure-a3m error: {error_a3m_pure[j]}')
        print(f'App-a3m error: {error_a3m_app[j]}')

            
    
    