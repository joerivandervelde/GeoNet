import pickle
import warnings
import Bio
import pandas as pd
import numpy as np
import os
import sys
import shutil

from Bio.PDB import PDBParser
from sklearn.neighbors import KDTree
from torch_cluster import radius_graph
from tqdm import tqdm
import random
import torch
from torch_geometric.data import InMemoryDataset, Data
import prettytable as pt
import math
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", default='DNA',
                        help="A ligand type. It can be chosen from DNA,RNA,P.")
    parser.add_argument("--psepos", dest="psepos", default='SC',
                        help="Pseudo position of residues. SC, CA, C stand for centroid of side chain, alpha-C atom and centroid of residue, respectively.")
    parser.add_argument("--features", dest="features", default='PSSM,HMM,SS,AF',
                        help="Feature groups. Multiple features should be separated by commas. You can combine features from PSSM, HMM, SS(secondary structure) and AF(atom features).")
    parser.add_argument("--context_radius", dest="context_radius", default=20, type=int,
                        help="Radius of structure context.")
    parser.add_argument("--tvseed", dest='tvseed', type=int, default=1995,
                        help='The random seed used to separate the validation set from training set.')
    return parser.parse_args()


def checkargs(args):
    if args.ligand not in ['DNA', 'RNA', 'P']:
        print('ERROR: ligand "{}" is not supported by GraphBind!'.format(args.ligand))
        raise ValueError
    if args.psepos not in ['SC', 'CA', 'C']:
        print('ERROR: pseudo position of a residue "{}" is not supported by GraphBind!'.format(args.psepos))
        raise ValueError
    features = args.features.strip().split(',')
    for feature in features:
        if feature not in ['PSSM', 'HMM', 'SS', 'RSA', 'PR', 'OH', 'AF', 'B']:
            print('ERROR: feature "{}" is not supported by GraphBind!'.format(feature))
            raise ValueError
    if args.context_radius <= 0:
        print('ERROR: radius of structure context should be positive!')
        raise ValueError

    return


class NeighResidue3DPoint(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None):
        super(NeighResidue3DPoint, self).__init__(root, transform, pre_transform)

        if dataset == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif dataset == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif dataset == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        splits = ['train', 'valid', 'test']
        return ['{}_data.pkl'.format(s) for s in splits]

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def _download(self):
        pass

    def process(self):
        pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
        cossim = torch.nn.CosineSimilarity(dim=1)
        seq_data_dict = {}
        for s, dataset in enumerate(['train', 'valid', 'test']):
            data_list = []
            with open(self.raw_dir + '/{}_data.pkl'.format(dataset), 'rb') as f:
                [data_dict, seqlist] = pickle.load(f)
            for seq in tqdm(seqlist):
                seq_data_list = []
                seq_data = data_dict[seq]
                for res_data in seq_data:
                    node_feas = res_data['node_feas']
                    node_feas = torch.tensor(node_feas, dtype=torch.float32)
                    # pos_raw = torch.tensor(res_data['pos_raw'], dtype=torch.float32)
                    pos = torch.tensor(res_data['pos'], dtype=torch.float32)
                    # local_frame = torch.tensor(res_data['local_frame'], dtype=torch.float32).view(1, 9)
                    Pij = torch.tensor(res_data['Pij'], dtype=torch.float32)
                    label = torch.tensor([res_data['label']], dtype=torch.float32)
                    # eigen_vals = torch.tensor([res_data['eigen_vals']], dtype=torch.float32)
                    u_0 = torch.tensor([res_data['u_0']], dtype=torch.float32)

                    edge_index = radius_graph(pos, r=10, loop=True, max_num_neighbors=40)
                    edge_attr = torch.cat([pdist(pos[edge_index[0]], pos[edge_index[1]]) / 10,
                                           (cossim(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(-1) + 1) / 2],
                                          dim=1)
                    # if np.sum(np.isnan(edge_attr.cpu().numpy())):
                    #     print(f'{seq} compute edge attribution has nan data.')
                    #     sys.exit()
                    dij = torch.exp(-1 * torch.square(pdist(pos[edge_index[0]], pos[edge_index[1]])) / (2 * 10 ** 2))

                    global_index = torch.zeros([2, pos.shape[0]], dtype=torch.long)
                    global_index[1] = torch.arange(pos.shape[0], dtype=torch.long)
                    wij = torch.exp(
                        -1 * torch.square(pdist(pos[global_index[0]], pos[global_index[1]])) / (2 * 10 ** 2))
                    data = Data(x=node_feas, pos=pos, y=label, u_0=u_0, dij=dij, edge_index=edge_index,
                                edge_attr=edge_attr, global_index=global_index, wij=wij, Pij=Pij, name=seq)
                    # data = Data(x=node_feas, pos=pos, y=label, u_0=u_0, dij=dij, edge_index=edge_index,
                    #             edge_attr=edge_attr, global_index=global_index, wij=wij, name=seq)
                    seq_data_list.append(data)
                data_list.extend(seq_data_list)
                seq_data_dict[seq] = seq_data_list
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[s])
        # torch.save(seq_data_dict, root_dir + '/processed/seq_data_dict.pt')


def Create_NeighResidue3DPoint(psepos, dist, feature_dir, raw_dir, seqanno, feature_combine, train_list, valid_list,
                               test_list):
    with open(feature_dir + '/' + ligand + '_psepos_{}.pkl'.format(psepos), 'rb') as f:
        residue_psepos = pickle.load(f)
    with open(feature_dir + '/' + ligand + '_residue_feas_{}.pkl'.format(feature_combine), 'rb') as f:
        residue_feas = pickle.load(f)
    with open(f'{Dataset_dir}/res_eg_local_coordinate_dict.pkl', 'rb') as f:
        res_local_frame_dict = pickle.load(f)
    for s, (dataset, seqlist) in enumerate(zip(['train', 'valid', 'test'], [train_list, valid_list, test_list])):
        data_dict = {}
        for seq in tqdm(seqlist):
            seq_data = []
            feas = residue_feas[seq]
            pos = residue_psepos[seq]
            res_local_frame = res_local_frame_dict[seq]
            label = np.array(list(map(int, list(seqanno[seq]['label']))))
            try:
                assert len(label) == feas.shape[0]
            except:
                print(seq, len(label), feas.shape[0])
            kdt = KDTree(pos)
            ind, dis = kdt.query_radius(pos, r=dist, return_distance=True, sort_results=True)
            for i in range(len(label)):
                res_psepos = pos[i]
                local_frame = res_local_frame[i]
                neigh_index, dis_to_i = ind[i], dis[i]
                raw_pos = pos[neigh_index]
                res_pos = raw_pos - res_psepos
                res_feas = feas[neigh_index]
                res_label = label[i]

                rij = res_pos
                Pij = np.concatenate(
                    [np.sum(local_frame[k:k + 1] * rij, axis=-1)[:, None] / np.linalg.norm(local_frame[k:k + 1]) for k
                     in range(3)], axis=-1)
                Pij = Pij * np.dot(res_psepos, res_psepos)
                # Pij = Pij * (np.linalg.norm(res_psepos) ** 3)
                # Pij = rij   # without local

                eigen_vals, eigen_vecs = np.linalg.eig(np.cov(res_pos, rowvar=False))
                eigen_vals = np.sort(eigen_vals)[::-1]
                u_0 = eigen_vals

                res_data = {
                    'node_feas': res_feas.astype('float32'),
                    'pos': res_pos.astype('float32'),
                    'u_0': u_0.astype('float32'),
                    'label': res_label.astype('float32'),
                    'Pij': Pij.astype('float32'),
                }
                seq_data.append(res_data)
            data_dict[seq] = seq_data
        with open(raw_dir + '/{}_data.pkl'.format(dataset), 'wb') as f:
            pickle.dump([data_dict, seqlist], f)
    return


def get_pdb_data(file_path):
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C',
                'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
                'LYS': 'K', 'ARG': 'R'}
    atom_count = -1
    pdb_file = open(file_path, 'r')
    Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19, 'CO': 59,
                            'V': 51, 'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3, 'NA': 23,
                            'HG': 200.6, 'MN': 55, 'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9, 'SE': 79,
                            'NI': 58.7}
    xyz_cord = []
    lines = pdb_file.readlines()
    for line in lines:
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count += 1
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            xyz_cord.append(xyz)
    xyz_cord = np.array(xyz_cord)
    center = xyz_cord.mean(axis=0).reshape([1, 3])
    xyz_cord_modified = (xyz_cord - center).round(3)

    add_spaces = lambda s, n: ' ' * n + s
    atom_count = -1
    line_modified_list = []
    for line in lines:
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count += 1
            x = add_spaces(str(xyz_cord_modified[atom_count][0]), 8 - len(str(xyz_cord_modified[atom_count][0])))
            y = add_spaces(str(xyz_cord_modified[atom_count][1]), 8 - len(str(xyz_cord_modified[atom_count][1])))
            z = add_spaces(str(xyz_cord_modified[atom_count][2]), 8 - len(str(xyz_cord_modified[atom_count][2])))
            line_modified = line[:30] + x + y + z + line[54:]
            line_modified_list.append(line_modified)
    text = ''.join(line_modified_list)
    return text


def cal_PDB_Modify(seqlist, PDB_chain_dir, PDB_raw_dir):
    if not os.path.exists(PDB_chain_dir):
        os.mkdir(PDB_chain_dir)

    for seq_id in tqdm(seqlist):
        # seq_id = '8glp_SX'
        pdbid, chains = seq_id.split('_')[0], seq_id.split('_')[1]
        file_path = PDB_raw_dir + '/{}.pdb'.format(seq_id)
        # if chains.islower():
        #     file_path = f'{PDB_raw_dir}/{pdbid}_{chains}{chains}.pdb'
        # else:
        #     file_path = f'{PDB_raw_dir}/{pdbid}_{chains}.pdb'
        with open(file_path, 'r') as f:
            text = f.readlines()
        if len(text) == 1:
            print('ERROR: PDB {} is empty.'.format(seq_id))
        if not os.path.exists(f'{PDB_chain_dir}/{seq_id}.pdb'):
            try:
                pdb_data = get_pdb_data(file_path)
                with open(f'{PDB_chain_dir}/{seq_id}.pdb', 'w') as f:
                    f.write(pdb_data)
            except KeyError:
                print('ERROR: Modify PDB file in ', seq_id)
                raise KeyError
    return


def def_atom_features():
    A = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 3, 0]}
    V = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 3, 0],
         'CG2': [0, 3, 0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 1, 1]}
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 1], 'CG': [0, 2, 1],
         'CD': [0, 2, 1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 1, 0],
         'CD1': [0, 3, 0], 'CD2': [0, 3, 0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 2, 0],
         'CG2': [0, 3, 0], 'CD1': [0, 3, 0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 2, 0], 'CD': [0, 2, 0], 'NE': [0, 1, 0], 'CZ': [1, 0, 0], 'NH1': [0, 2, 0], 'NH2': [0, 2, 0]}
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [-1, 0, 0],
         'OD1': [-1, 0, 0], 'OD2': [-1, 0, 0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [-1, 0, 0], 'OE1': [-1, 0, 0], 'OE2': [-1, 0, 0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'OG': [0, 1, 0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'OG1': [0, 1, 0],
         'CG2': [0, 3, 0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'SG': [-1, 1, 0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 0, 0],
         'OD1': [0, 0, 0], 'ND2': [0, 2, 0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 0, 0], 'OE1': [0, 0, 0], 'NE2': [0, 2, 0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'ND1': [-1, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'NE2': [-1, 1, 1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 2, 0], 'CE': [0, 2, 0], 'NZ': [0, 3, 1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 0, 1],
         'OH': [-1, 1, 0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'SD': [0, 0, 0], 'CE': [0, 3, 0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 0, 1], 'NE1': [0, 1, 1], 'CE2': [0, 0, 1], 'CE3': [0, 1, 1],
         'CZ2': [0, 1, 1], 'CZ3': [0, 1, 1], 'CH2': [0, 1, 1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                     'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    return atom_features


def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C',
                'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
                'LYS': 'K', 'ARG': 'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path, 'r')
    pdb_res = pd.DataFrame(columns=['ID', 'atom', 'res', 'res_id', 'xyz', 'B_factor'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19, 'CO': 59,
                            'V': 51, 'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3, 'NA': 23,
                            'HG': 200.6, 'MN': 55, 'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9, 'SE': 79,
                            'NI': 58.7}

    # while True:
    lines = pdb_file.readlines()
    for line in lines:
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count += 1
            # res_pdb_id = int(line[22:26])
            res_pdb_id = f'{line[21]}{line[22:27].strip()}'
            if res_pdb_id != before_res_pdb_id:
                res_count += 1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N', 'CA', 'C', 'O', 'H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5, 0.5, 0.5]
            tmps = pd.Series(
                {'ID': atom_count, 'atom': line[12:16].strip(), 'atom_type': atom_type, 'res': res,
                 # 'res_id': int(line[22:26]),
                 'res_id': res_pdb_id,
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                 'occupancy': float(line[54:60]),
                 'B_factor': float(line[60:66]), 'mass': Relative_atomic_mass[atom_type], 'is_sidechain': is_sidechain,
                 'charge': atom_fea[0], 'num_H': atom_fea[1], 'ring': atom_fea[2]})
            tmps = tmps.to_frame().transpose()
            # if len(res_id_list) == 0:
            #     res_id_list.append(int(line[22:26]))
            # elif res_id_list[-1] != int(line[22:26]):
            #     res_id_list.append(int(line[22:26]))
            if len(res_id_list) == 0:
                res_id_list.append(res_pdb_id)
            elif res_id_list[-1] != res_pdb_id:
                res_id_list.append(res_pdb_id)
            # pdb_res = pdb_res.append(tmps, ignore_index=True)
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            #     pdb_res = pdb_res.append(tmps, ignore_index=True)
            pdb_res = pd.concat([pdb_res, tmps])
        # if line.startswith('TER'):
        #     break

    return pdb_res, res_id_list


def cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir, seqanno):
    if not os.path.exists(PDB_DF_dir):
        os.mkdir(PDB_DF_dir)

    for seq_id in tqdm(seqlist):
        # seq_id = '8a22_z'
        pdbid, chains = seq_id.split('_')[0], seq_id.split('_')[1]
        # print(seq_id)
        file_path = PDB_chain_dir + '/{}.pdb'.format(seq_id)
        # file_path = PDB_chain_dir + '/pdb/{}.pdb'.format(seq_id)
        with open(file_path, 'r') as f:
            text = f.readlines()
        if len(text) == 1:
            print('ERROR: PDB {} is empty.'.format(seq_id))
        if not os.path.exists(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id)):
            try:
                pdb_DF, res_id_list = get_pdb_DF(file_path)
                assert len(res_id_list) == len(seqanno[seq_id]['seq']), f'{seq_id} has not equal sequence.'
                with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'wb') as f:
                    pickle.dump({'pdb_DF': pdb_DF, 'res_id_list': res_id_list}, f)
            except KeyError:
                print('ERROR: UNK in ', seq_id)
                raise KeyError

    return


def cal_Psepos(seqlist, PDB_DF_dir, Dataset_dir, psepos, ligand, seqanno):
    seq_CA_pos = {}
    seq_centroid = {}
    seq_sidechain_centroid = {}

    for seq_id in tqdm(seqlist):
        # seq_id = '3qlp_H'
        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)
        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        # center = np.mean(np.array(pdb_res_i['xyz'].tolist()), axis=0)  # add

        res_CA_pos = []
        res_centroid = []
        res_sidechain_centroid = []
        res_types = []
        for res_id in res_id_list:
            res_type = pdb_res_i[pdb_res_i['res_id'] == res_id]['res'].values[0]
            res_types.append(res_type)

            res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
            xyz = np.array(res_atom_df['xyz'].tolist())
            masses = np.array(res_atom_df['mass'].tolist()).reshape(-1, 1)
            centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_atom_df = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['is_sidechain'] == 1)]

            try:
                CA = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['atom'] == 'CA')]['xyz'].values[0]
            except IndexError:
                print('IndexError: no CA in seq:{} res_id:{}'.format(seq_id, res_id))
                CA = centroid

            res_CA_pos.append(CA)
            res_centroid.append(centroid)

            if len(res_sidechain_atom_df) == 0:
                res_sidechain_centroid.append(centroid)
            else:
                xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
                masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
                sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
                res_sidechain_centroid.append(sidechain_centroid)

        if ''.join(res_types) != seqanno[seq_id]['seq']:
            print(f'{seq_id} has not eequal sequence.')
            print(''.join(res_types))
            print(seqanno[seq_id]['seq'])
            return

        res_CA_pos = np.array(res_CA_pos)
        res_centroid = np.array(res_centroid)
        res_sidechain_centroid = np.array(res_sidechain_centroid)
        seq_CA_pos[seq_id] = res_CA_pos
        seq_centroid[seq_id] = res_centroid
        seq_sidechain_centroid[seq_id] = res_sidechain_centroid


    with open(Dataset_dir + '/' + ligand + '_psepos_' + 'SC' + '.pkl', 'wb') as f:
        pickle.dump(seq_sidechain_centroid, f)
    with open(Dataset_dir + '/' + ligand + '_psepos_' + 'CA' + '.pkl', 'wb') as f:
        pickle.dump(seq_CA_pos, f)
    with open(Dataset_dir + '/' + ligand + '_psepos_' + 'C' + '.pkl', 'wb') as f:
        pickle.dump(seq_centroid, f)

    return


def cal_eg_local_frame(seq_list, dist, Dataset_dir, seqanno, aa_frames='triplet_sidechain'):
    # from scipy.spatial.transform import Rotation as R
    # rot = R.random()
    with open(Dataset_dir + '/' + ligand + '_psepos_{}.pkl'.format(psepos), 'rb') as f:
        residue_psepos = pickle.load(f)
    with open(Dataset_dir + '/' + ligand + '_psepos_CA.pkl', 'rb') as f:
        residue_CA = pickle.load(f)

    res_local_frame_dict = {}
    for seq_id in tqdm(seq_list):
        # seq_id = '1brp_A'
        pdbid, chains = seq_id.split('_')[0], seq_id.split('_')[1]
        # for c in chains:
        #     seq = f'{pdbid}_{c}'
        seq = seq_id
        pos = residue_psepos[seq]
        pos_ca = residue_CA[seq]
        sc_to_ca = pos - pos_ca
        label = np.array(list(map(int, list(seqanno[seq]['label']))))
        kdt = KDTree(pos)
        ind, dis = kdt.query_radius(pos, r=dist, return_distance=True, sort_results=True)
        res_local_frame = []
        for i in range(len(label)):
            res_psepos = pos[i]
            sc_to_ca_i = sc_to_ca[i]
            neigh_index, dis_to_i = ind[i], dis[i]
            res_pos = pos[neigh_index]
            # rij = res_pos - res_psepos
            eigen_vals, eigen_vecs = np.linalg.eig(np.cov(res_pos, rowvar=False))
            # from scipy.spatial.transform import Rotation as R
            # rot = R.random()
            # rotated_pos = rot.apply(res_pos)
            # eigen_vals_rot, eigen_vecs_rot = np.linalg.eig(np.cov(rotated_pos, rowvar=False))
            # u0_st.append(eigen_vals.reshape([1, len(eigen_vals)]))
            eigen_idx = np.argsort(eigen_vals)[::-1]
            eigen_vals = eigen_vals[eigen_idx]
            eigen_vecs = eigen_vecs[:, eigen_idx]
            eigen_vecs = np.array([np.sign(np.dot(sc_to_ca_i, eigen_vecs[:, 0]) + 1e-8) * eigen_vecs[:, 0],
                                   np.sign(np.dot(sc_to_ca_i, eigen_vecs[:, 1]) + 1e-8) * eigen_vecs[:, 1],
                                   np.sign(np.dot(sc_to_ca_i, eigen_vecs[:, 2]) + 1e-8) * eigen_vecs[:, 2]])
            res_local_frame.append(eigen_vecs[None, :, :])
        res_local_frame_dict[seq] = np.concatenate(res_local_frame, axis=0)

    with open(f'{Dataset_dir}/res_eg_local_coordinate_dict.pkl', 'wb') as f:
        pickle.dump(res_local_frame_dict, f)
    return


def cal_PSSM(ligand, seq_list, pssm_dir, feature_dir, seqanno):
    nor_pssm_dict = {}
    if ligand != 'PP':
        for seqid in seq_list:
            file = f'{seqid}.pssm'
            try:
                with open(pssm_dir + '/' + file, 'r') as fin:
                    fin_data = fin.readlines()
                    pssm_begin_line = 3
                    pssm_end_line = 0
                    for i in range(1, len(fin_data)):
                        if fin_data[i] == '\n':
                            pssm_end_line = i
                            break
                    feature = np.zeros([(pssm_end_line - pssm_begin_line), 20])
                    axis_x = 0
                    for i in range(pssm_begin_line, pssm_end_line):
                        raw_pssm = fin_data[i].split()[2:22]
                        axis_y = 0
                        for j in raw_pssm:
                            feature[axis_x][axis_y] = (1 / (1 + math.exp(-float(j))))
                            axis_y += 1
                        axis_x += 1
                    nor_pssm_dict[file.split('.')[0]] = feature
            except:
                print(f'{pssm_dir}/{file} not find.')
                sequence = seqanno[seqid]['seq']
                feature = np.zeros([len(sequence), 20])
                nor_pssm_dict[file.split('.')[0]] = feature
    else:
        for seqid in seq_list:
            with open(f'{feature_dir}/label_single_dict.pkl', 'rb') as f:
                label_single_dict = pickle.load(f)
            if len(seqid.split('_')) == 2:
                pdbid, chains = seqid.split('_')[0], seqid.split('_')[1]
                # sequence = seqanno[seqid]['seq']
                features = []
                for i, c in enumerate(chains):
                    file = f'{pssm_dir}/{pdbid}_{c}.pssm'
                    sequence = label_single_dict[f'{pdbid}_{c}']['seq']
                    try:
                        with open(file, 'r') as fin:
                            fin_data = fin.readlines()
                            pssm_begin_line = 3
                            pssm_end_line = 0
                            for i in range(1, len(fin_data)):
                                if fin_data[i] == '\n':
                                    pssm_end_line = i
                                    break
                            feature = np.zeros([(pssm_end_line - pssm_begin_line), 20])
                            axis_x = 0
                            for i in range(pssm_begin_line, pssm_end_line):
                                raw_pssm = fin_data[i].split()[2:22]
                                axis_y = 0
                                for j in raw_pssm:
                                    feature[axis_x][axis_y] = (1 / (1 + math.exp(-float(j))))
                                    axis_y += 1
                                axis_x += 1
                            features.append(feature)
                    except:
                        feature = np.zeros([len(sequence), 20])
                        features.append(feature)
            elif len(seqid.split('_')) == 3:
                pdbid, chains, lr = seqid.split('_')[0], seqid.split('_')[1], seqid.split('_')[2]
                features = []
                for i, c in enumerate(chains):
                    file = f'{pssm_dir}/{pdbid}_{c}_{lr}.pssm'
                    sequence = label_single_dict[f'{pdbid}_{c}_{lr}']['seq']
                    try:
                        with open(file, 'r') as fin:
                            fin_data = fin.readlines()
                            pssm_begin_line = 3
                            pssm_end_line = 0
                            for i in range(1, len(fin_data)):
                                if fin_data[i] == '\n':
                                    pssm_end_line = i
                                    break
                            feature = np.zeros([(pssm_end_line - pssm_begin_line), 20])
                            axis_x = 0
                            for i in range(pssm_begin_line, pssm_end_line):
                                raw_pssm = fin_data[i].split()[2:22]
                                axis_y = 0
                                for j in raw_pssm:
                                    feature[axis_x][axis_y] = (1 / (1 + math.exp(-float(j))))
                                    axis_y += 1
                                axis_x += 1
                            features.append(feature)
                    except:
                        feature = np.zeros([len(sequence), 20])
                        features.append(feature)
            else:
                print(f'error in {seqid}')
                sys.exit()
            nor_pssm_dict[seqid] = np.concatenate(features, axis=0)
            assert len(seqanno[seqid]['seq']) == np.concatenate(features, axis=0).shape[0]
    with open(feature_dir + '/{}_PSSM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(nor_pssm_dict, f)
    return


def cal_HMM(ligand, seq_list, hmm_dir, feature_dir):
    hmm_dict = {}
    if ligand != 'PP':
        for seqid in seq_list:
            file = f'{seqid}.hhm'
            with open(hmm_dir + '/' + file, 'r') as fin:
                fin_data = fin.readlines()
                hhm_begin_line = 0
                hhm_end_line = 0
                for i in range(len(fin_data)):
                    if '#' in fin_data[i]:
                        hhm_begin_line = i + 5
                    elif '//' in fin_data[i]:
                        hhm_end_line = i
                feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
                axis_x = 0
                for i in range(hhm_begin_line, hhm_end_line, 3):
                    line1 = fin_data[i].split()[2:-1]
                    line2 = fin_data[i + 1].split()
                    axis_y = 0
                    for j in line1:
                        if j == '*':
                            feature[axis_x][axis_y] = 9999 / 10000.0
                        else:
                            feature[axis_x][axis_y] = float(j) / 10000.0
                        axis_y += 1
                    for j in line2:
                        if j == '*':
                            feature[axis_x][axis_y] = 9999 / 10000.0
                        else:
                            feature[axis_x][axis_y] = float(j) / 10000.0
                        axis_y += 1
                    axis_x += 1
                feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
                hmm_dict[file.split('.')[0]] = feature
    else:
        for seqid in seq_list:
            with open(f'{feature_dir}/label_single_dict.pkl', 'rb') as f:
                label_single_dict = pickle.load(f)
            if len(seqid.split('_')) == 2:
                pdbid, chains = seqid.split('_')[0], seqid.split('_')[1]
                # sequence = seqanno[seqid]['seq']
                features = []
                for i, c in enumerate(chains):
                    file = f'{hmm_dir}/{pdbid}_{c}.hhm'
                    sequence = label_single_dict[f'{pdbid}_{c}']['seq']
                    try:
                        with open(file, 'r') as fin:
                            fin_data = fin.readlines()
                            hhm_begin_line = 0
                            hhm_end_line = 0
                            for i in range(len(fin_data)):
                                if '#' in fin_data[i]:
                                    hhm_begin_line = i + 5
                                elif '//' in fin_data[i]:
                                    hhm_end_line = i
                            feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
                            axis_x = 0
                            for i in range(hhm_begin_line, hhm_end_line, 3):
                                line1 = fin_data[i].split()[2:-1]
                                line2 = fin_data[i + 1].split()
                                axis_y = 0
                                for j in line1:
                                    if j == '*':
                                        feature[axis_x][axis_y] = 9999 / 10000.0
                                    else:
                                        feature[axis_x][axis_y] = float(j) / 10000.0
                                    axis_y += 1
                                for j in line2:
                                    if j == '*':
                                        feature[axis_x][axis_y] = 9999 / 10000.0
                                    else:
                                        feature[axis_x][axis_y] = float(j) / 10000.0
                                    axis_y += 1
                                axis_x += 1
                            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
                            features.append(feature)
                    except:
                        feature = np.zeros([len(sequence), 30])
                        features.append(feature)
            elif len(seqid.split('_')) == 3:
                pdbid, chains, lr = seqid.split('_')[0], seqid.split('_')[1], seqid.split('_')[2]
                features = []
                for i, c in enumerate(chains):
                    file = f'{hmm_dir}/{pdbid}_{c}_{lr}.hhm'
                    sequence = label_single_dict[f'{pdbid}_{c}_{lr}']['seq']
                    try:
                        with open(file, 'r') as fin:
                            fin_data = fin.readlines()
                            hhm_begin_line = 0
                            hhm_end_line = 0
                            for i in range(len(fin_data)):
                                if '#' in fin_data[i]:
                                    hhm_begin_line = i + 5
                                elif '//' in fin_data[i]:
                                    hhm_end_line = i
                            feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
                            axis_x = 0
                            for i in range(hhm_begin_line, hhm_end_line, 3):
                                line1 = fin_data[i].split()[2:-1]
                                line2 = fin_data[i + 1].split()
                                axis_y = 0
                                for j in line1:
                                    if j == '*':
                                        feature[axis_x][axis_y] = 9999 / 10000.0
                                    else:
                                        feature[axis_x][axis_y] = float(j) / 10000.0
                                    axis_y += 1
                                for j in line2:
                                    if j == '*':
                                        feature[axis_x][axis_y] = 9999 / 10000.0
                                    else:
                                        feature[axis_x][axis_y] = float(j) / 10000.0
                                    axis_y += 1
                                axis_x += 1
                            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
                            features.append(feature)
                    except:
                        feature = np.zeros([len(sequence), 30])
                        features.append(feature)
            else:
                print(f'error in {seqid}')
                sys.exit()
            hmm_dict[seqid] = np.concatenate(features, axis=0)
            assert len(seqanno[seqid]['seq']) == np.concatenate(features, axis=0).shape[0]
    with open(feature_dir + '/{}_HMM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(hmm_dict, f)
    return


def cal_DSSP(ligand, seq_list, dssp_dir, feature_dir):
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319}
    map_ss_8 = {' ': [1, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 1, 0, 0, 0, 0, 0, 0], 'T': [0, 0, 1, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 1, 0, 0, 0, 0], 'G': [0, 0, 0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0, 0],
                'E': [0, 0, 0, 0, 0, 0, 1, 0], 'B': [0, 0, 0, 0, 0, 0, 0, 1]}
    dssp_dict = {}
    for seqid in seq_list:
        # seqid = '8i9z_CR'
        file = f'{dssp_dir}/{seqid}.dssp'
        with open(file, 'r') as fin:
            fin_data = fin.readlines()
        for i, line in enumerate(fin_data):
            if line.split()[0] == '#':
                line_start = i + 1
                break
        seq_feature = {}
        for i in range(line_start, len(fin_data)):
            line = fin_data[i]
            if line[13] not in maxASA.keys() or line[9] == ' ':
                continue
            # res_id = float(line[5:10])
            res_id = f'{line[11]}{line[5:10].strip()}'
            feature = np.zeros([14])
            feature[:8] = map_ss_8[line[16]]
            feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1)  # ASA / maxASA --> ACC
            feature[9] = (float(line[85:91]) + 1) / 2  # TCO
            feature[10] = min(1, float(line[91:97]) / 180)  # KAPPA
            feature[11] = min(1, (float(line[97:103]) + 180) / 360)  # ALPHA
            feature[12] = min(1, (float(line[103:109]) + 180) / 360)  # PHI
            feature[13] = min(1, (float(line[109:115]) + 180) / 360)  # PSI
            seq_feature[res_id] = feature.reshape((1, -1))

        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seqid), 'rb') as f:
            tmp = pickle.load(f)
        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        fea_dssp = []
        for res_id_i in res_id_list:
            if res_id_i in seq_feature.keys():
                fea_dssp.append(seq_feature[res_id_i])
            else:
                fea_dssp.append(np.zeros(list(seq_feature.values())[0].shape))
        fea_dssp = np.concatenate(fea_dssp, axis=0)
        dssp_dict[seqid] = fea_dssp
    with open(feature_dir + '/{}_SS.pkl'.format(ligand), 'wb') as f:
        pickle.dump(dssp_dict, f)
    return


def cal_ResProperty(ligand, seqlist, seqanno, feature_dir):
    with open(f'{feature_dir}/feature/res_properties.pkl', 'rb') as f:
        properties = pickle.load(f)
    property_dict = {}
    for seqid in seqlist:
        sequence = seqanno[seqid]['seq']
        protien_properties = np.zeros([len(sequence), 8])
        for i, res in enumerate(sequence):
            protien_properties[i] = properties[res]
        property_dict[seqid] = protien_properties

    with open(feature_dir + '/{}_PR.pkl'.format(ligand), 'wb') as f:
        pickle.dump(property_dict, f)
    return


def cal_OneHot(ligand, seqlist, seqanno, feature_dir):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    amino_acid_dict = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}
    one_hot_dict = {}
    for seqid in seqlist:
        sequence = seqanno[seqid]['seq']
        one_hot_feas = np.zeros([len(sequence), 20])
        for i, res in enumerate(sequence):
            index = amino_acid_dict[res]
            one_hot_feas[i, index] = 1
        one_hot_dict[seqid] = one_hot_feas
    with open(feature_dir + '/{}_OH.pkl'.format(ligand), 'wb') as f:
        pickle.dump(one_hot_dict, f)
    return


def cal_Blosum(ligand, seqlist, seqanno, feature_dir):
    Max_blosum = np.array([4, 5, 6, 6, 9, 5, 5, 6, 8, 4, 4, 5, 5, 6, 7, 4, 5, 11, 7, 4])
    Min_blosum = np.array([-3, -3, -4, -4, -4, -3, -4, -4, -3, -4, -4, -3, -3, -4, -4, -3, -2, -4, -3, -3])
    with open(f'{feature_dir}/blosum_dict.pkl', 'rb') as f:
        blosum_dict = pickle.load(f)
    blosum_feas_dict = {}
    for seqid in seqlist:
        sequence = seqanno[seqid]['seq']
        blosum_feas = []
        for i, res in enumerate(sequence):
            blosum_feas.append(blosum_dict[res])
        blosum_feas = (np.array(blosum_feas) - Min_blosum) / (Max_blosum - Min_blosum)
        blosum_feas_dict[seqid] = blosum_feas
    with open(feature_dir + '/{}_B.pkl'.format(ligand), 'wb') as f:
        pickle.dump(blosum_feas_dict, f)
    return


def cal_AA_property(ligand, seqlist, seqanno, feature_dir):
    Side_Chain_Atom_num = {'A': 5.0, 'C': 6.0, 'D': 8.0, 'E': 9.0, 'F': 11.0, 'G': 4.0, 'H': 10.0, 'I': 8.0, 'K': 9.0,
                           'L': 8.0, 'M': 8.0, 'N': 8.0, 'P': 7.0, 'Q': 9.0, 'R': 11.0, 'S': 6.0, 'T': 7.0, 'V': 7.0,
                           'W': 14.0, 'Y': 12.0}
    Side_Chain_Charge_num = {'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0, 'G': 0.0, 'H': 1.0, 'I': 0.0, 'K': 1.0,
                             'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0, 'S': 0.0, 'T': 0.0, 'V': 0.0,
                             'W': 0.0, 'Y': 0.0}
    Side_Chain_hydrogen_bond_num = {'A': 2.0, 'C': 2.0, 'D': 4.0, 'E': 4.0, 'F': 2.0, 'G': 2.0, 'H': 4.0, 'I': 2.0,
                                    'K': 2.0, 'L': 2.0, 'M': 2.0, 'N': 4.0, 'P': 2.0, 'Q': 4.0, 'R': 4.0, 'S': 4.0,
                                    'T': 4.0, 'V': 2.0, 'W': 3.0, 'Y': 3.0}
    Side_Chain_pKa = {'A': 7.0, 'C': 7.0, 'D': 3.65, 'E': 3.22, 'F': 7.0, 'G': 7.0, 'H': 6.0, 'I': 7.0, 'K': 10.53,
                      'L': 7.0, 'M': 7.0, 'N': 8.18, 'P': 7.0, 'Q': 7.0, 'R': 12.48, 'S': 7.0, 'T': 7.0, 'V': 7.0,
                      'W': 7.0, 'Y': 10.07}
    Hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9,
                      'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                      'W': -0.9, 'Y': -1.3}
    res_frequency = {'A': 0.04, 'C': 0.009, 'D': 0.034, 'E': 0.034, 'F': 0.034, 'G': 0.07, 'H': 0.039, 'I': 0.031,
                     'K': 0.133, 'L': 0.041, 'M': 0.02, 'N': 0.05, 'P': 0.041, 'Q': 0.045, 'R': 0.171, 'S': 0.061,
                     'T': 0.05, 'V': 0.039, 'W': 0.013, 'Y': 0.042}

    AA_property = {}
    for res in Hydrophobicity:
        property = []
        # property.append((Side_Chain_Atom_num[res] - min(Side_Chain_Atom_num.values())) / (
        #         max(Side_Chain_Atom_num.values()) - min(Side_Chain_Atom_num.values())))
        # property.append(Side_Chain_Charge_num[res])
        # property.append((Side_Chain_hydrogen_bond_num[res] - min(Side_Chain_hydrogen_bond_num.values())) / (
        #         max(Side_Chain_hydrogen_bond_num.values()) - min(Side_Chain_hydrogen_bond_num.values())))
        # property.append((Side_Chain_pKa[res] - min(Side_Chain_pKa.values())) / (
        #         max(Side_Chain_pKa.values()) - min(Side_Chain_pKa.values())))
        # property.append((Hydrophobicity[res] - min(Hydrophobicity.values())) / (
        #         max(Hydrophobicity.values()) - min(Hydrophobicity.values())))
        # property.append(res_frequency[res])
        property.append(Side_Chain_Atom_num[res])
        property.append(Side_Chain_Charge_num[res])
        property.append(Side_Chain_hydrogen_bond_num[res])
        property.append(Side_Chain_pKa[res])
        property.append(Hydrophobicity[res])
        AA_property[res] = np.array(property).reshape([1, len(property)])

    AA_property_dict = {}
    for seqid in seqlist:
        sequence = seqanno[seqid]['seq']
        AA_protein = []
        for res in sequence:
            AA_protein.append(AA_property[res])
        AA_property_dict[seqid] = np.concatenate(AA_protein, axis=0)
    with open(feature_dir + '/{}_PR.pkl'.format(ligand), 'wb') as f:
        pickle.dump(AA_property_dict, f)
    return


def cal_rsa(ligand, seqlist, seqanno, pdb_dir, feature_dir):
    import freesasa
    rsa_dict = {}
    for seq in seqlist:
        pdbid, chains = seq.split('_')[0], seq.split('_')[1]
        pdb_path = f'{pdb_dir}/{seq}.pdb'
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure, freesasa.Parameters(
            {'algorithm': freesasa.LeeRichards, 'n-slices': 100, 'probe-radius': 1.4}))
        residueAreas = result.residueAreas()
        RSA = {}
        for c in chains:
            for r in residueAreas[c].keys():
                name = f'{c}{r}'
                RSA_AA = []
                RSA_AA.append(min(1, residueAreas[c][r].relativeTotal))
                RSA_AA.append(min(1, residueAreas[c][r].relativePolar))
                RSA_AA.append(min(1, residueAreas[c][r].relativeApolar))
                RSA_AA.append(min(1, residueAreas[c][r].relativeMainChain))
                if math.isnan(residueAreas[c][r].relativeSideChain):
                    RSA_AA.append(0)
                else:
                    RSA_AA.append(min(1, residueAreas[c][r].relativeSideChain))
                RSA[name] = np.array(RSA_AA).reshape(1, len(RSA_AA))
        rsa_dict[seq] = RSA

    with open(feature_dir + '/{}_RSA.pkl'.format(ligand), 'wb') as f:
        pickle.dump(rsa_dict, f)
    return


def cal_atom_feas(ligand, seqlist, PDB_DF_dir, feature_dir):
    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85, 'H': 1.2, 'D': 1.2, 'SE': 1.9, 'P': 1.8, 'FE': 2.23,
                        'BR': 1.95, 'F': 1.47, 'CO': 2.23, 'V': 2.29, 'I': 1.98, 'CL': 1.75, 'CA': 2.81, 'B': 2.13,
                        'ZN': 2.29, 'MG': 1.73, 'NA': 2.27, 'HG': 1.7, 'MN': 2.24, 'K': 2.75, 'AC': 3.08, 'AL': 2.51,
                        'W': 2.39, 'NI': 2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)
    res_atom_feas_dict = {}
    for seq_id in tqdm(seqlist):
        # seq_id = '1VFB_A_l'
        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)

        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        pdb_res_i = pdb_res_i[pdb_res_i['atom_type'] != 'H']
        mass = np.array(pdb_res_i['mass'].tolist()).reshape(-1, 1)
        mass = mass / 32

        B_factor = np.array(pdb_res_i['B_factor'].tolist()).reshape(-1, 1)
        if (max(B_factor) - min(B_factor)) == 0:
            B_factor = np.zeros(B_factor.shape) + 0.5
        else:
            B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
        is_sidechain = np.array(pdb_res_i['is_sidechain'].tolist()).reshape(-1, 1)
        occupancy = np.array(pdb_res_i['occupancy'].tolist()).reshape(-1, 1)
        charge = np.array(pdb_res_i['charge'].tolist()).reshape(-1, 1)
        num_H = np.array(pdb_res_i['num_H'].tolist()).reshape(-1, 1)
        ring = np.array(pdb_res_i['ring'].tolist()).reshape(-1, 1)

        atom_type = pdb_res_i['atom_type'].tolist()
        atom_vander = np.zeros((len(atom_type), 1))
        for i, type in enumerate(atom_type):
            try:
                atom_vander[i] = atom_vander_dict[type]
            except:
                atom_vander[i] = atom_vander_dict['C']

        atom_feas = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]
        atom_feas = np.concatenate(atom_feas, axis=1)

        res_atom_feas = []
        atom_begin = 0
        for i, res_id in enumerate(res_id_list):
            res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
            atom_num = len(res_atom_df)
            res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
            res_atom_feas_i = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
            res_atom_feas.append(res_atom_feas_i)
            atom_begin += atom_num
        res_atom_feas = np.concatenate(res_atom_feas, axis=0)
        res_atom_feas_dict[seq_id] = res_atom_feas

    with open(feature_dir + '/{}_res_atom_feas.pkl'.format(ligand), 'wb') as f:
        pickle.dump(res_atom_feas_dict, f)
    return


def PDBResidueFeature(seqlist, PDB_chain_dir, PDB_DF_dir, feature_dir, ligand, residue_feature_list, feature_combine,
                      atomfea):
    for fea in residue_feature_list:
        with open(feature_dir + '/' + ligand + '_{}.pkl'.format(fea), 'rb') as f:
            locals()['residue_fea_dict_' + fea] = pickle.load(f)

    residue_feas_dict = {}
    for seq_id in tqdm(seqlist):
        # seq_id = '1VFB_A_l'
        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)
        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        residue_feas = []
        pdbid, chains = seq_id.split('_')[0], seq_id.split('_')[1]
        for fea in residue_feature_list:
            fea_i = locals()['residue_fea_dict_' + fea][seq_id]
            if isinstance(fea_i, np.ndarray):
                residue_feas.append(fea_i)
            elif isinstance(fea_i, dict):
                fea_ii = []
                for res_id_i in res_id_list:
                    if res_id_i in fea_i.keys():
                        fea_ii.append(fea_i[res_id_i])
                    else:
                        fea_ii.append(np.zeros(list(fea_i.values())[0].shape))
                fea_ii = np.concatenate(fea_ii, axis=0)
                residue_feas.append(fea_ii)
        try:
            residue_feas = np.concatenate(residue_feas, axis=1)
            assert np.sum(np.isnan(residue_feas)) == 0, f'{seq_id} residue feature has nan.'
        except ValueError:
            print('ERROR: Feature dimensions of {} are inconsistent!'.format(seq_id))
            raise ValueError
        if residue_feas.shape[0] != len(res_id_list):
            print(
                'ERROR: For {}, the number of residues with features is not consistent with the number of residues in the query!'.format(
                    seq_id))
            raise IndexError

        if atomfea:
            with open(f'{feature_dir}/{ligand}_res_atom_feas.pkl', 'rb') as f:
                res_atom_feas_dict = pickle.load(f)
            res_atom_feas = res_atom_feas_dict[seq_id]
            residue_feas = np.concatenate((res_atom_feas, residue_feas), axis=1)
            # residue_feas = np.concatenate((residue_feas, res_atom_feas), axis=1)

        if np.isnan(residue_feas).any():
            print(seq_id)
            sys.exit()
        residue_feas_dict[seq_id] = residue_feas

    with open(feature_dir + '/' + ligand + '_residue_feas_' + feature_combine + '.pkl', 'wb') as f:
        pickle.dump(residue_feas_dict, f)

    return


def tv_split(train_list, seed):
    random.seed(seed)
    random.shuffle(train_list)
    valid_list = train_list[:int(len(train_list) * 0.2)]
    train_list = train_list[int(len(train_list) * 0.2):]
    return train_list, valid_list


def StatisticsSampleNum(train_list, valid_list, test_list, seqanno):
    def sub(seqlist, seqanno):
        pos_num_all = 0
        res_num_all = 0
        for seqid in seqlist:
            anno = list(map(int, list(seqanno[seqid]['label'])))
            pos_num = sum(anno)
            res_num = len(anno)
            pos_num_all += pos_num
            res_num_all += res_num
        neg_num_all = res_num_all - pos_num_all
        pnratio = pos_num_all / float(neg_num_all)
        return len(seqlist), res_num_all, pos_num_all, neg_num_all, pnratio

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'NumSeq', 'NumRes', 'NumPos', 'NumNeg', 'PNratio']
    tb.float_format = '0.3'

    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list + valid_list, seqanno)
    tb.add_row(['train+valid', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list, seqanno)
    tb.add_row(['train', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(valid_list, seqanno)
    tb.add_row(['valid', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(test_list, seqanno)
    tb.add_row(['test', seq_num, res_num, pos_num, neg_num, pnratio])
    print(tb)
    return


if __name__ == '__main__':

    args = parse_args()
    checkargs(args)

    ligand = 'P' + args.ligand if args.ligand != 'HEME' else 'PHEM'
    psepos = args.psepos
    features = args.features.strip().split(',')
    dist = args.context_radius
    feature_list = []
    feature_combine = ''

    if 'EM' in features:
        feature_list.append('EM')
        feature_combine += 'E'
    if 'PR' in features:
        feature_list.append('PR')
        feature_combine += 'Pr'
    if 'PSSM' in features:
        feature_list.append('PSSM')
        feature_combine += 'P'
    if 'MSA' in features:
        feature_list.append('MSA')
        feature_combine += 'M'
    if 'HMM' in features:
        feature_list.append('HMM')
        feature_combine += 'H'
    if 'B' in features:
        feature_list.append('B')
        feature_combine += 'B'
    if 'SS' in features:
        feature_list.append('SS')
        feature_combine += 'S'
    if 'RSA' in features:
        feature_list.append('RSA')
        feature_combine += 'R'
    if 'AF' in features:
        feature_list.append('AF')
        feature_combine += 'A'
    if 'OH' in features:
        feature_list.append('OH')
        feature_combine += 'O'

    trainingset_dict = {
        'PDNA': 'PDNA-881_Train.txt',
        'PRNA': 'PRNA-1497_Train.txt',
        'PP': 'PP-1001_Train.txt',
    }

    testset_dict = {
        'PDNA': 'PDNA-220_Test.txt',
        'PRNA': 'PRNA-374_Test.txt',
        'PP': 'PP-250_Test.txt',
    }

    # Dataset_dir = os.path.abspath('..') + '/Datasets' + '/' + ligand + '/modified_data/'
    # Dataset_dir = os.path.abspath('..') + '/Datasets' + '/' + ligand + '/GeoBind_data/'
    # Dataset_dir = os.path.abspath('..') + '/Datasets' + '/' + ligand + '/GeoBind_data/modified_data'
    # Dataset_dir = os.path.abspath('..') + '/Datasets/customed_data' + '/' + ligand + '/modified_data/'
    Dataset_dir = os.path.abspath('..') + '/Datasets/customed_data' + '/' + ligand

    trainset_anno = Dataset_dir + '/{}'.format(trainingset_dict[ligand])
    testset_anno = Dataset_dir + '/{}'.format(testset_dict[ligand])

    seqanno = {}
    train_list = []
    valid_list = []
    test_list = []

    with open(trainset_anno, 'r') as f:
        train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        query_seq = train_text[i + 1].strip()
        query_anno = train_text[i + 2].strip()
        train_list.append(query_id)
        seqanno[query_id] = {'seq': query_seq, 'label': query_anno}

    with open(testset_anno, 'r') as f:
        test_text = f.readlines()
    for i in range(0, len(test_text), 3):
        query_id = test_text[i].strip()[1:]
        query_seq = test_text[i + 1].strip()
        query_anno = test_text[i + 2].strip()
        test_list.append(query_id)
        seqanno[query_id] = {'seq': query_seq, 'label': query_anno}

    train_list, valid_list = tv_split(train_list, args.tvseed)
    StatisticsSampleNum(train_list, valid_list, test_list, seqanno)
    seqlist = train_list + valid_list + test_list
    # with open(f'{Dataset_dir}/train_valid_test.pkl', 'wb') as f:
    #     pickle.dump([train_list, valid_list, test_list], f)
    # with open(f'{Dataset_dir}/seqanno.pkl', 'wb') as f:
    #     pickle.dump(seqanno, f)

    # train_list, valid_list, test_list = pickle.load(open(f'{Dataset_dir}/train_valid_test.pkl', 'rb'))
    # seqlist = train_list + valid_list + test_list
    # seqanno = pickle.load(open(f'{Dataset_dir}/seqanno.pkl', 'rb'))
    # seqlist = seqlist[::-1]

    PDB_raw_dir = Dataset_dir + '/PDB'
    Feature_dir = Dataset_dir + '/feature'
    Dataset_dir = Dataset_dir + '/modified_data'
    if not os.path.exists(Dataset_dir):
        os.mkdir(Dataset_dir)
    PDB_chain_dir = Dataset_dir + '/PDB'
    print('1.Modify the PDB information.')
    cal_PDB_Modify(seqlist, PDB_chain_dir, PDB_raw_dir)

    PDB_DF_dir = Dataset_dir + '/PDB_DF'
    print('2.Extract the PDB information.')
    cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir, seqanno)
    print('3.calculate the pseudo positions.')
    cal_Psepos(seqlist, PDB_DF_dir, Dataset_dir, psepos, ligand, seqanno)
    print('4.calculate the local coordinate.')
    cal_eg_local_frame(seqlist, dist, Dataset_dir, seqanno)

    print('5.calculate the residue features.')
    if 'AF' in feature_list:
        atomfea = True
        feature_list.remove('AF')
    else:
        atomfea = False

    cal_PSSM(ligand, seqlist, Feature_dir + '/PSSM', Dataset_dir, seqanno)
    cal_HMM(ligand, seqlist, Feature_dir + '/HMM', Dataset_dir)
    cal_DSSP(ligand, seqlist, Feature_dir + '/SS', Dataset_dir)
    cal_OneHot(ligand, seqlist, seqanno, Dataset_dir)
    cal_AA_property(ligand, seqlist, seqanno, Dataset_dir)  # final
    cal_atom_feas(ligand, seqlist, PDB_DF_dir, Dataset_dir)

    PDBResidueFeature(seqlist, PDB_chain_dir, PDB_DF_dir, Dataset_dir, ligand, feature_list, feature_combine, atomfea)

    root_dir = Dataset_dir + '/' + ligand + '_{}_dist{}_{}'.format(psepos, dist, feature_combine)
    raw_dir = root_dir + '/raw'
    if os.path.exists(raw_dir):
        shutil.rmtree(root_dir)
    os.makedirs(raw_dir)
    print('6.Calculate the neighborhood of residues. Save to {}.'.format(root_dir))
    Create_NeighResidue3DPoint(psepos, dist, Dataset_dir, raw_dir, seqanno, feature_combine, train_list,
                               valid_list, test_list)
    _ = NeighResidue3DPoint(root=root_dir, dataset='test')
