from rdkit import Chem
from rdkit.Chem import AllChem

params = AllChem.ETKDGv3()
print(", ".join(dir(params)))
