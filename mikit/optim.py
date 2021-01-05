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
    def __init__(self, xi = 0.01, kernel = None):
        self.kernel = kernel if kernel is not None else BASIC_KERNEL
        self.xi = xi


    def gaussian_process(self, X_exp, y_exp, kernel):
        # スケール調整
        scaler_y = StandardScaler().fit(y_exp)
        gpr = GaussianProcessRegressor(kernel = kernel)
        y_trans = scaler_y.transform(y_exp)
        gpr.fit(X_exp, y_trans)
        # GPによる平均および標準偏差の予測
        y_gp, y_gp_std = gpr.predict(X_all, return_std = True)
        y_gp = scaler_y.inverse_transform(y_gp)
        y_gp_exp, y_gp_exp_std = gpr.predict(X_exp, return_std = True)
        y_gp_exp = scaler_y.inverse_transform(y_gp_exp)
        return y_gp, y_gp_std, y_gp_exp
    
    @staticmethod
    def get_ei(xi, y_gp, y_gp_std, y_gp_exp):
        # 獲得関数
        mu, sigma = y_gp, y_gp_std
        mu_sample = y_gp_exp
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)
        with np.errstate(divide = 'warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        ei = ei.ravel()
        return ei
    
    def bays_optimize(self, xi, next_point, y_gp, y_gp_std, y_gp_exp, d_cut):
        ei = self.EI(xi, y_gp, y_gp_std, y_gp_exp)
        output_ei = ei
        comps, space = COMP, X_all
        target_ratioes, target_names = [], []
        for n in range(next_point):
            idx = ei.argmax()
            target_ratioes.append(space[idx])
            target_names.append(comps[idx])
            Log(str(4 * int(N) + n + 1) + "st_Compositon : " + str(comps[idx]) + "  EI : " + str(ei[idx]))
            comps, space, ei = self.Remove_Comp(idx, comps, space, ei, d_cut)
        return target_ratioes, target_names, output_ei
    
    


class MultiBayes(BayesOpt):
    def __init__(self):
        d_cut = 0
        next_points = 0
        d_type = "euclid"
        pass

if __name__ == "__main__":
    # creating chemical formula
    atoms = ["Pb", "Sn", "La"]
    tcf = TriChemFormula()
    # creating comp name
    comp = tcf.get_tri_name(atoms, delta = 0.05)
    pseudo_ratio = tcf.get_only_pseudo_ratio(comp,  atoms)
    # Experimental composition and results
    comp_ex = ["PbF2", "SnF2", "LaF3", "PbSnF4"]
    pseudo_ratio_ex = tcf.get_only_pseudo_ratio(comp_ex,  atoms)
    y_ex = np.array([-5.5, -8.3, -8.9, -3.4])
    # Bayesian Optimization
    bo = BayesOpt()
    # bo.opt(kernel = kernel)


    
"""
アルゴリズム検討
"""
