import numpy as np
import pandas as pd
from compname import ChemFormula

class CreateFeature:
    def __init__(self, path="./data/atom.csv", excluded_col=["Element"]):
        df_atom_features = pd.read_csv(path).drop(excluded_col, axis=1)
        self.atom_feature_values = df_atom_features.values.T
        self.atom_feature_colnames = df_atom_features.columns.values
    
    def get_features(self, molratioes)
        print(self.atom_feature_values[0])
        print(self.atom_feature_colnames)

if __name__ == "__main__":
    path = "./mi/data/atom.csv"
    cf = CreateFeature(path)