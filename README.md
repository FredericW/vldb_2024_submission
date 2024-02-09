# Implementation of A3M Mechanism

## Dependencies 
1. A3M needs the optimization module of [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).
2. To run real-world data experiments (including multidimensional data), additional packages of [pandas](https://pandas.pydata.org/) and [folktable](https://github.com/socialfoundations/folktables) are needed.
3. Code for preprocessing the real-world data is in **real_dataset.ipynb** (you can also try **multidimension.ipynb** to play with the ACSData).

## Synthetic data experiments

> python synthetic_exp_dis.py

This will give you the performance of A3M, in comparison with several other baselines (run the script *_sw.py for SW) over a range of choices of epsilon. The data type and various hyperparameters (e.g., bin size, split ratio) can be specified in the input arguments. For example, for 10000 Gaussian data points, run

> python synthetic_exp_dis.py --data_type=GAUSSIAN --n=10000

## Real-world data experiments 

> python real_income.py

> python real_retirement.py

> python real_green_taxi.py

> python multidimension.py

This will give you the performance of A3M on multi-dimensional data and real-world datasets, in comparison with several other baselines (run the script *_sw.py for SW). 

## Hyperparameter studies of A3M 

> python vary_N_synthetic.py

> python vary_ax_synthetic.py

> python vary_split_synthetic.py


To further facilitate reproducibility, we also provide the Matlab implementation for solving the optimal noise distribution, which is numerically, more stable.
