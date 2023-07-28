# Implementation of A3M Mechanism

## Dependencies 
1. A3M needs the optimization module of [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).
2. To run real-world data experiments (including multidimensional data), additional packages of [pandas](https://pandas.pydata.org/) and [folktable](https://github.com/socialfoundations/folktables) are needed.
3. Code for preprocessing the real-world data is in **real_dataset.ipynb** (you can also try **multidimension.ipynb** to play with the ACSData).

## Synthetic data experiments (use *_sw.py for SW-EMS)

> python synthetic_exp.py

## Hyperparameter studies of A3M on synthetic data

> python vary_M_synthetic.py

> python vary_ax_synthetic.py

> python vary_split_synthetic.py

## Real-world data experiments (use *_sw.py for SW-EMS)

> python real_income.py

> python real_retirement.py

> python real_green_taxi.py

> python multidimension.py

## To further facilitate reproducibility, we also provide the matlab implementation

