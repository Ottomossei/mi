import os, io, sys, re, math, itertools
import pandas as pd
from functools import reduce
from pymatgen.core import Composition
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern, ExpSineSquared
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import numpy as np

from compname import ChemFormula, TriChemFormula

BASIC_KERNEL = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + DotProduct()

class BayesOpt():
    def __init__(self, X_all, xi = 0.01, kernel = None):
        self.X_all = X_all
        self.xi = xi
        self.kernel = kernel if kernel is not None else BASIC_KERNEL

    def fit(self, X_exp, y_exp):
        y_exp = y_exp.reshape(-1, 1)
        scaler_y = StandardScaler().fit(y_exp)
        # Fitting of gaussian process
        gpr = GaussianProcessRegressor(kernel = self.kernel)
        y_trans = scaler_y.transform(y_exp)
        gpr.fit(X_exp, y_trans)
        # Prediction of fitting model
        mu, sigma = gpr.predict(self.X_all, return_std = True)
        mu = scaler_y.inverse_transform(mu)
        mu_exp, mu_exp_std = gpr.predict(X_exp, return_std = True)
        mu_exp = scaler_y.inverse_transform(mu_exp)
        # Calculate the ei
        sigma = sigma.reshape(-1, 1)
        mu_exp_opt = np.max(mu_exp)
        with np.errstate(divide = 'warn'):
            imp = mu - mu_exp_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        self.mu = mu
        self.sigma = sigma
        self.ei = ei.ravel()
    
    def get_info(self):
        return self.mu, self.sigma, self.ei
    
    def get_next(self, num, cutoff=0.1):
        def remove_obj():
            print("a")
        ei = self.ei.copy()
        X_obj = self.X_all.copy()
        X_next = list()
        # print(X_obj)
        for n in range(num):
            idx = ei.argmax()
            X_next.append(X_obj[idx])
            # remove_idx = 
        return np.array(X_next)

        

    
    # def bays_optimize(self, xi, next_point, y_gp, y_gp_std, y_gp_exp, d_cut):
    #     ei = self.EI(xi, y_gp, y_gp_std, y_gp_exp)
    #     output_ei = ei
    #     comps, space = COMP, X_all
    #     target_ratioes, target_names = [], []
    #     for n in range(next_point):
    #         idx = ei.argmax()
    #         target_ratioes.append(space[idx])
    #         target_names.append(comps[idx])
    #         Log(str(4 * int(N) + n + 1) + "st_Compositon : " + str(comps[idx]) + "  EI : " + str(ei[idx]))
    #         comps, space, ei = self.Remove_Comp(idx, comps, space, ei, d_cut)
    #     return target_ratioes, target_names, output_ei
    
    



if __name__ == "__main__":
    # creating chemical formula
    atoms = ["Pb", "Sn", "La"]
    tcf = TriChemFormula()
    # creating comp name
    comp_all = tcf.get_tri_name(atoms, delta = 0.05)
    X_all = tcf.get_only_pseudo_ratio(comp_all,  atoms)
    # Experimental composition and results
    comp_exp = ["PbF2", "SnF2", "LaF3", "PbSnF4"]
    X_exp = tcf.get_only_pseudo_ratio(comp_exp,  atoms)
    y_exp = np.array([-5.5, -8.3, -8.9, -3.4])
    # Bayesian Optimization
    bo = BayesOpt(X_all)
    bo.fit(X_exp, y_exp)
    mu, sigma, ei = bo.get_info()
    print(ei)
    X_predict = bo.get_next(1)
    print(X_predict)


    
"""
アルゴリズム検討
"""
