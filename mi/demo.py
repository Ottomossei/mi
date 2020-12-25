from compname import ChemFormula
import pandas as pd

# load csv
path = "./mi/data/learning.csv"
df = pd.read_csv(path)
comp = df["X"].values

# output matrix of moler ratio
cn = ChemFormula()
comp_ratio = cn.get_molratio(comp)
print(comp_ratio)