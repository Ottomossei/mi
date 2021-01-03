import numpy as np
import pandas as pd
import re, os

"""
This module converts chemical formula information into numerical values.
"""

class ChemFormula:
    def __init__(self, path="./mi/data/atom.csv", index="Element"):
        """
        Args:
            path(str) : Path of the csv file containing the name of the target element
            index(str) : Index with the target element name
        """
        self.atoms = pd.read_csv(path)[index].values
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

