import numpy as np
import pandas as pd
import re, os

"""
This module converts chemical formula information into numerical values.
"""

class ChemFormula:
    def __init__(self, path="./data/atom.csv", index="Element"):
        """
        Args:
            path(str) : Path of the csv file containing the name of the target element
            index(str) : Index with the target element name
        """
        self.path = path
        self.atoms = pd.read_csv(self.path)[index].values
        patterns = str()
        for i, a in enumerate(self.atoms):
            if len(re.findall('[A-Z]', a)) == 1:
                patterns += str(a) + "|"
            else:
                patterns += "\(" + str(a) + "\)|"
                self.atoms[i] = "(" + str(a) + ")"
        else:
            patterns = patterns[:-1]
        self.regex = re.compile(patterns)
        self.list_atoms = list(self.atoms)
        # Definition of abbreviated names
        # self.Ln 

    @staticmethod
    def _num_judge(num):
        """
        Methods for identifying subscripts
        """
        if re.match(r"\d",num):
            output = float(re.match(r"\d+\.\d+|\d",num).group(0))
        else:
            output = 1.0
        return output

    def get_mol(self, names):
        """
        Args:
            names(list, numpy): List of chemical formulas
        Returns:
            mol numpy[names, obj_atoms]
        """
        output = np.zeros((len(names), self.atoms.shape[0]))
        for i in range(len(names)):
            comp = names[i]
            check = self.regex.sub('', comp)
            num_first_indexes = [m.end() for m in self.regex.finditer(comp)]
            obj_atom_indexes = [self.list_atoms.index(m.group(0)) for m in self.regex.finditer(comp)]
            if len(re.findall('[A-Z]', check)):
                print("Elements (" + str(re.findall('[A-Z]', check)[0]) + ") not in the database are included.")
                output = None
                break
            else:
                for j in range(len(num_first_indexes[:-1])):
                    num = comp[num_first_indexes[j]:num_first_indexes[j+1]]
                    output[i,obj_atom_indexes[j]] = self._num_judge(num)
                else:
                    # check monoatomic molecule
                    if (len(num_first_indexes[:-1])==0):
                        # example F, O2
                        num = comp[num_first_indexes[0]:]
                        output[i,obj_atom_indexes[0]] = self._num_judge(num)
                    else:
                        # example LiO2
                        num = comp[num_first_indexes[j+1]:]
                        output[i,obj_atom_indexes[j+1]] = self._num_judge(num)
        return output
    
    def get_molratio(self, names, exc_atoms=None, obj_atoms=None):
        """
        Args:
            names(list, numpy) : List of chemical formulas
            exc_atoms(list, numpy) : Elements to be excluded from the molar ratio calculation
            obj_atoms(list, numpy) : Elements to be objected from the molar ratio calculation
        Returns:
            mole ratio numpy[names, obj_atoms]
        """
        output = self.get_mol(names)
        # Creating a matrix that excludes the target element
        if exc_atoms:
            exc_atom_indexes = [self.list_atoms.index(e) for e in exc_atoms]
            output[:, exc_atom_indexes] = 0.0
        elif obj_atoms:
            obj_atom_indexes = [self.list_atoms.index(e) for e in obj_atoms]
            exc_atom_indexes = [i for i in np.arange(len(self.list_atoms)) if i not in obj_atom_indexes]
            output[:, exc_atom_indexes] = 0.0
        for l in range(len(output)):
            output[l] /= output.sum(axis = 1)[l]
        return output
    
    def get_obj_name(self, names, obj_atoms, conductor_atom = "F"):
        p = "[A-Z][a-z]|[A-Z]"
        obj_atoms.append(conductor_atom)
        output = [n for n in names if len(set(re.findall(p, n)) & set(obj_atoms)) == len(set(re.findall(p, n)))]
        return output



class TriChemFormula(ChemFormula):
    def __init__(self, path="./data/atom.csv", index="Element"):
        super().__init__(path, index)
    
    def get_tri_name(self, obj_atoms, delta, conductor_atom = "F", index = "valence"):
        valence = pd.read_csv(self.path)[index].values
        all_atoms = list(self.atoms)
        obj_atoms.append(conductor_atom)
        idx = [all_atoms.index(atom) for atom in obj_atoms]
        comp_names = []
        x, y, z = obj_atoms[0:3]
        nums = np.array([x for x in range(int(1/delta) + 1)]) / int(1/delta)
        li_nums = list()
        for x in nums:
            for y in nums:
                z = round(1 - x - y, 5)
                if z > 0:
                    c = round(x * valence[idx[0]] + y * valence[idx[1]] + z * valence[idx[2]], 5)
                    comp_names.append(obj_atoms[0] + str(x) + obj_atoms[1] + str(y) + obj_atoms[2] + str(z) + conductor_atom + str(c))
                elif z == 0:
                    c = round(x * valence[idx[0]] + y * valence[idx[1]] + 0 * valence[idx[2]], 5)
                    comp_names.append(obj_atoms[0] + str(x) + obj_atoms[1] + str(y) + obj_atoms[2] + str(0) + conductor_atom + str(c))
        return comp_names

    def get_all_ratio(self, atoms, delta = 0.01, conductor_atom = "F"):
        comp_names = self.get_tri_name(atoms, delta, conductor_atom)
        return self.get_molratio(comp_names)

    def get_only_pseudo_ratio(self, comp_names, obj_atoms):
        output = self.get_molratio(comp_names, obj_atoms = obj_atoms)
        obj_atom_indexes = [self.list_atoms.index(e) for e in obj_atoms]
        output = output[:,obj_atom_indexes]
        for l in range(len(output)):
            output[l] /= output.sum(axis = 1)[l]
        return output
    
if __name__ == "__main__":
    names = ["LiPO4", "BaSnF4", "Pb2SnF2", "BaF", "LaBaF"]
    obj_atoms = ["Ba", "Sn", "La"]
    cn = ChemFormula()
    test = cn.get_obj_name(names, obj_atoms)
    # print(comps)
