import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self.fingerprints = [self.get_fingerprint(mol) for mol in self.mols]

    def get_fingerprint(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        arr = torch.zeros((1, 1024))
        for bit in fp.GetOnBits():
            arr[0, bit] = 1
        return arr

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.fingerprints[idx]
