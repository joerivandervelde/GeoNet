import sys
import os
import warnings
import math

from Bio.PDB import PDBList

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(''))
GEONET = os.path.abspath('..')
sys.path.append(GEONET)

# Set the absolute paths of blast+, HHBlits and their databases in here.
PSIBLAST = '/mnt/data0/Hanjy/software/ncbi-blast-2.14.0+/bin/psiblast'
PSIBLAST_DB = '/mnt/data0/Hanjy/software/database/uniref90/uniref90'
HHblits = '/mnt/data0/Hanjy/software/hh-suite/build/bin/hhblits'
HHblits_DB = '/mnt/data0/Hanjy/software/database/uniclust30_2018_08/uniclust30_2018_08'

# DSSP is contained in "scripts/dssp", and it should be given executable permission by commend line "chmod +x scripts/dssp".
# DSSP = '/mnt/data0/Hanjy/.conda/envs/pytorch2/bin/mkdssp'
DSSP = GEONET + '/scripts/dssp'

import pickle
import numpy as np
import subprocess
from itertools import repeat, product
import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import shutil
import time
import argparse
from Bio.PDB.PDBParser import PDBParser
from sklearn.neighbors import KDTree
from torch_cluster import radius_graph

# python prediction_fast.py --querypath ../output/example --ligand DNA --pdbid 6ide --chainid B --filename 6ide_B.pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--querypath", dest="query_path", default='../output/example',
                        help="The path of query structure")
    parser.add_argument("--pdbid", dest="pdbid", default='1hhp',
                        help="The file name of the query structure which should be in PDB format.")
    parser.add_argument("--filename", dest="filename", default='1hhp_A.pdb',
                        help="The file name of the query structure which should be in PDB format.")
    parser.add_argument("--chainid", dest="chain_id", default='A',
                        help="The query chain id(case sensitive). If there is only one chain in your query structure, you can leave it blank.")
    parser.add_argument("--ligand", dest='ligand', default='DNA',
                        help='Ligand types. Multiple ligands should be separated by commas. You can choose from DNA,RNA,CA,MG,MN,ATP,HEME.')
    parser.add_argument("--cpu", dest="fea_num_threads", default='20',
                        help="The number of CPUs used for calculating PSSM and HMM profile.")
    return parser.parse_args()


def SaveChainPDB(chain_id, query_path, filename, query_id):
    pdb_file = "{}/{}".format(query_path, filename)
    chain_file = '{}/{}.pdb'.format(query_path, query_id)
    if not os.path.exists(chain_file):
        with open(pdb_file, 'r') as f:
            pdb_text = f.readlines()
        text = []

        chainid_list = set()
        for line in pdb_text:
            if line.startswith('ATOM'):
                chainid_list.add(line[21])
        chainid_list = list(chainid_list)
        for chian in chain_id:
            if not chian in chainid_list:
                print('ERROR: Your query structure dose not have the query chain, please check your chain ID!')
                raise ValueError
        chain_id = [v for v in chain_id]
        for line in pdb_text:
            if line.startswith('ATOM') and line[21] in chain_id:
                text.append(line)

        text.append('TER\n')
        text.append('END\n')

        with open(chain_file, 'w') as f:
            f.writelines(text)
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


def Modify_PDB(PDB_new_dir, PDB_raw_dir):
    with open(PDB_raw_dir, 'r') as f:
        text = f.readlines()
    if len(text) == 1:
        print('ERROR: PDB {} is empty.'.format(PDB_raw_dir))
    if not os.path.exists(PDB_new_dir):
        try:
            pdb_data = get_pdb_data(PDB_raw_dir)
            with open(PDB_new_dir, 'w') as f:
                f.write(pdb_data)
        except KeyError:
            print('ERROR: Modify PDB file in ', PDB_raw_dir)
            raise KeyError
    os.system(f'mv {PDB_raw_dir} {PDB_raw_dir}1')
    os.system(f'cp {PDB_new_dir} {PDB_raw_dir}')


def cal_eg_local_frame(query_path, query_id, dist, aa_frames='triplet_sidechain'):
    if not os.path.exists(f'{query_path}/{query_id}_eg_local_coordinate.pkl'):
        with open(f'{query_path}/{query_id}_psepos_SC.pkl', 'rb') as f:
            residue_psepos = pickle.load(f)
        with open(f'{query_path}/{query_id}_psepos_CA.pkl', 'rb') as f:
            residue_CA = pickle.load(f)
        pdbid, chain_id = query_id.split('_')[0], query_id.split('_')[1]
        sequence = ''
        for chain in chain_id:
            sequence += open(f'{query_path}/{pdbid}_{chain}.seq', 'r').readlines()[1].strip()
        pos = residue_psepos
        pos_ca = residue_CA
        sc_to_ca = pos - pos_ca
        kdt = KDTree(pos)
        ind, dis = kdt.query_radius(pos, r=dist, return_distance=True, sort_results=True)
        res_local_frame = []
        for i in range(len(sequence)):
            res_psepos = pos[i]
            sc_to_ca_i = sc_to_ca[i]
            neigh_index, dis_to_i = ind[i], dis[i]
            res_pos = pos[neigh_index]
            # rij = res_pos - res_psepos
            eigen_vals, eigen_vecs = np.linalg.eig(np.cov(res_pos, rowvar=False))

            eigen_idx = np.argsort(eigen_vals)[::-1]
            eigen_vals = eigen_vals[eigen_idx]
            eigen_vecs = eigen_vecs[:, eigen_idx]
            eigen_vecs = np.array([np.sign(np.dot(sc_to_ca_i, eigen_vecs[:, 0]) + 1e-8) * eigen_vecs[:, 0],
                                   np.sign(np.dot(sc_to_ca_i, eigen_vecs[:, 1]) + 1e-8) * eigen_vecs[:, 1],
                                   np.sign(np.dot(sc_to_ca_i, eigen_vecs[:, 2]) + 1e-8) * eigen_vecs[:, 2]])
            res_local_frame.append(eigen_vecs[None, :, :])
        res_local_frame = np.concatenate(res_local_frame, axis=0)
        with open(f'{query_path}/{query_id}_eg_local_coordinate.pkl', 'wb') as f:
            pickle.dump(res_local_frame, f)

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


def PDBFeature(query_id, PDB_chain_dir, results_dir):
    print('PDB_chain -> PDB_DF')
    pdb_path = PDB_chain_dir + '/{}.pdb'.format(query_id)

    if not os.path.exists(results_dir + '/{}.df'.format(query_id)):
        pdb_DF, res_id_list = get_pdb_DF(pdb_path)
        with open(results_dir + '/{}.df'.format(query_id), 'wb') as f:
            pickle.dump({'pdb_DF': pdb_DF, 'res_id_list': res_id_list}, f)
    else:
        with open(results_dir + '/{}.df'.format(query_id), 'rb') as f:
            pdb_df_Data = pickle.load(f)
            pdb_DF, res_id_list = pdb_df_Data['pdb_DF'], pdb_df_Data['res_id_list']

    # print('Extract PDB_feature')
    pdbid, chain_id = query_id.split('_')[0], query_id.split('_')[1]
    if not os.path.exists(results_dir + '/' + query_id + '_psepos_SC.pkl'):
        res_CA_pos = []
        res_centroid = []
        res_sidechain_centroid = []
        for chain in chain_id:
            res_types = []
            for res_id in res_id_list:
                if res_id[0] == chain:
                    res_type = pdb_DF[pdb_DF['res_id'] == res_id]['res'].values[0]
                    res_types.append(res_type)

                    res_atom_df = pdb_DF[pdb_DF['res_id'] == res_id]
                    xyz = np.array(res_atom_df['xyz'].tolist())
                    masses = np.array(res_atom_df['mass'].tolist()).reshape(-1, 1)
                    centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
                    res_sidechain_atom_df = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['is_sidechain'] == 1)]
                    try:
                        CA = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['atom'] == 'CA')]['xyz'].values[0]
                    except IndexError:
                        print('IndexError: no CA in seq:{} res_id:{}'.format(query_id, res_id))
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

                    sequence = ''.join(res_types)
                    with open(results_dir + '/' + pdbid + '_' + chain + '.seq', 'w') as f:
                        f.write('>{}\n'.format(pdbid + '_' + chain))
                        f.write(sequence)
        res_CA_pos = np.array(res_CA_pos)
        res_centroid = np.array(res_centroid)
        res_sidechain_centroid = np.array(res_sidechain_centroid)

        with open(results_dir + '/' + query_id + '_psepos_SC.pkl', 'wb') as f:
            pickle.dump(res_sidechain_centroid, f)
        with open(results_dir + '/' + query_id + '_psepos_CA.pkl', 'wb') as f:
            pickle.dump(res_CA_pos, f)
    return

def norm_DSSP(query_path, query_id):
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319}
    map_ss_8 = {' ': [1, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 1, 0, 0, 0, 0, 0, 0], 'T': [0, 0, 1, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 1, 0, 0, 0, 0],
                'G': [0, 0, 0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0, 0], 'E': [0, 0, 0, 0, 0, 0, 1, 0],
                'B': [0, 0, 0, 0, 0, 0, 0, 1]}

    with open('{}/{}.dssp'.format(query_path, query_id), 'r') as f:
        fin_data = f.readlines()

    for i, line in enumerate(fin_data):
        if line.split()[0] == '#':
            line_start = i + 1
            break
    seq_feature = {}
    for i in range(line_start, len(fin_data)):
        line = fin_data[i]
        if line[13] not in maxASA.keys() or line[9] == ' ':
            continue
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

    with open(query_path + '/{}.df'.format(query_id), 'rb') as f:
        tmp = pickle.load(f)
    pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
    fea_dssp = []
    for res_id_i in res_id_list:
        if res_id_i in seq_feature.keys():
            fea_dssp.append(seq_feature[res_id_i])
        else:
            fea_dssp.append(np.zeros(list(seq_feature.values())[0].shape))
    fea_dssp = np.concatenate(fea_dssp, axis=0)

    return fea_dssp

def norm_blosum(query_path, seqid):
    Max_blosum = np.array([4, 5, 6, 6, 9, 5, 5, 6, 8, 4, 4, 5, 5, 6, 7, 4, 5, 11, 7, 4])
    Min_blosum = np.array([-3, -3, -4, -4, -4, -3, -4, -4, -3, -4, -4, -3, -3, -4, -4, -3, -2, -4, -3, -3])
    blosum_dict = {'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
                   'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
                   'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
                   'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
                   'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
                   'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
                   'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
                   'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
                   'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
                   'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
                   'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
                   'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
                   'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
                   'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
                   'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
                   'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
                   'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
                   'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
                   'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
                   'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]}
    pdbid, chains = seqid.split('_')
    blosum_feas = []
    for c in chains:
        query_id = f'{pdbid}_{c}'
        with open(f'{query_path}/{query_id}.seq', 'r') as f:
            data = f.readlines()
        sequence = data[1].strip()
        for i, res in enumerate(sequence):
            blosum_feas.append(blosum_dict[res])
    blosum_feas = (np.array(blosum_feas) - Min_blosum) / (Max_blosum - Min_blosum)
    return blosum_feas

def norm_oh(query_path, seqid):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    amino_acid_dict = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}
    pdbid, chains = seqid.split('_')
    one_hot = []
    for c in chains:
        query_id = f'{pdbid}_{c}'
        with open(f'{query_path}/{query_id}.seq', 'r') as f:
            data = f.readlines()
        sequence = data[1].strip()
        one_hot_feas = np.zeros([len(sequence), 20])
        for i, res in enumerate(sequence):
            index = amino_acid_dict[res]
            one_hot_feas[i, index] = 1
        one_hot.append(one_hot_feas)
    one_hot = np.concatenate(one_hot, axis=0)
    return one_hot


def norm_atom(query_path, query_id):
    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85, 'H': 1.2, 'D': 1.2, 'SE': 1.9, 'P': 1.8, 'FE': 2.23,
                        'BR': 1.95,
                        'F': 1.47, 'CO': 2.23, 'V': 2.29, 'I': 1.98, 'CL': 1.75, 'CA': 2.81, 'B': 2.13, 'ZN': 2.29,
                        'MG': 1.73, 'NA': 2.27,
                        'HG': 1.7, 'MN': 2.24, 'K': 2.75, 'AC': 3.08, 'AL': 2.51, 'W': 2.39, 'NI': 2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    with open('{}/{}.df'.format(query_path, query_id), 'rb') as f:
        tmp = pickle.load(f)
    pdb_DF, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
    pdb_DF = pdb_DF[pdb_DF['atom_type'] != 'H']

    # atom features
    mass = np.array(pdb_DF['mass'].tolist()).reshape(-1, 1)
    mass = mass / 32
    B_factor = np.array(pdb_DF['B_factor'].tolist()).reshape(-1, 1)
    if (max(B_factor) - min(B_factor)) == 0:
        B_factor = np.zeros(B_factor.shape) + 0.5
    else:
        B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
    is_sidechain = np.array(pdb_DF['is_sidechain'].tolist()).reshape(-1, 1)
    charge = np.array(pdb_DF['charge'].tolist()).reshape(-1, 1)
    num_H = np.array(pdb_DF['num_H'].tolist()).reshape(-1, 1)
    ring = np.array(pdb_DF['ring'].tolist()).reshape(-1, 1)
    atom_type = pdb_DF['atom_type'].tolist()
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
        res_atom_df = pdb_DF[pdb_DF['res_id'] == res_id]
        atom_num = len(res_atom_df)
        res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
        res_atom_feas_i = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
        res_atom_feas.append(res_atom_feas_i)
        atom_begin += atom_num
    res_atom_feas = np.concatenate(res_atom_feas, axis=0)

    return res_atom_feas


def PDBResidueFeature_fast(query_path, query_id, blosum, dssp, atom, oh):
    if not (blosum.shape[0] == dssp.shape[0] == atom.shape[0] == oh.shape[0]):
        print('blosum shape: ', blosum.shape)
        print('DSSP shape: ', dssp.shape)
        print('ATOM shape: ', atom.shape)
        print('oh shape: ', oh.shape)
        raise ValueError

    residue_feas = [atom, blosum, dssp, oh]
    residue_feas = np.concatenate(residue_feas, axis=1)
    with open('{}/{}_fast.resfea'.format(query_path, query_id), 'wb') as f:
        pickle.dump(residue_feas, f)
    return

def PDBResidueFeature_fast_dna(query_path, query_id, blosum, dssp, atom):
    if not (blosum.shape[0] == dssp.shape[0] == atom.shape[0]):
        print('blosum shape: ', blosum.shape)
        print('DSSP shape: ', dssp.shape)
        print('ATOM shape: ', atom.shape)
        raise ValueError

    residue_feas = [atom, blosum, dssp]
    residue_feas = np.concatenate(residue_feas, axis=1)
    with open('{}/{}_fast.resfea'.format(query_path, query_id), 'wb') as f:
        pickle.dump(residue_feas, f)
    return

class seq_Dataset():
    def __init__(self, query_path, query_id, dist):

        seq_data = self.Create_seqData(query_path, query_id, dist)
        seq_data_list = []
        pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
        cossim = torch.nn.CosineSimilarity(dim=1)
        for res_data in seq_data:
            node_feas = torch.tensor(res_data['node_feas'], dtype=torch.float32)
            pos = torch.tensor(res_data['pos'], dtype=torch.float32)
            Pij = torch.tensor(res_data['Pij'], dtype=torch.float32)
            u_0 = torch.tensor([res_data['u_0']], dtype=torch.float32)
            edge_index = radius_graph(pos, r=10, loop=True, max_num_neighbors=40)
            edge_attr = torch.cat([pdist(pos[edge_index[0]], pos[edge_index[1]]) / 10,
                                   (cossim(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(-1) + 1) / 2],
                                  dim=1)
            dij = torch.exp(-1 * torch.square(pdist(pos[edge_index[0]], pos[edge_index[1]])) / (2 * 10 ** 2))
            global_index = torch.zeros([2, pos.shape[0]], dtype=torch.long)
            global_index[1] = torch.arange(pos.shape[0], dtype=torch.long)
            wij = torch.exp(
                -1 * torch.square(pdist(pos[global_index[0]], pos[global_index[1]])) / (2 * 10 ** 2))
            data = Data(x=node_feas, pos=pos, u_0=u_0, dij=dij, edge_index=edge_index, edge_attr=edge_attr,
                        global_index=global_index, wij=wij, Pij=Pij)
            seq_data_list.append(data)
        self.data, self.slices = self.collate(seq_data_list)
        with open('{}/{}.df'.format(query_path, query_id), 'rb') as f:
            self.res_id_list = pickle.load(f)['res_id_list']
        pdbid, chains = query_id.split('_')[0],query_id.split('_')[1]
        self.sequence = []
        for chian in chains:
            with open(f'{query_path}/{pdbid}_{chian}.seq', 'r') as f:
                self.sequence.extend([v for v in f.readlines()[1].strip()])

    def Create_seqData(self, query_path, query_id, dist):

        with open('{}/{}_psepos_SC.pkl'.format(query_path, query_id), 'rb') as f:
            pos = pickle.load(f)
        with open('{}/{}_fast.resfea'.format(query_path, query_id), 'rb') as f:
            feas = pickle.load(f)
        with open('{}/{}_eg_local_coordinate.pkl'.format(query_path, query_id), 'rb') as f:
            eg_local_frame = pickle.load(f)

        seq_data = []
        try:
            assert pos.shape[0] == feas.shape[0]
        except:
            print(query_id, pos.shape[0], feas.shape[0])
        kdt = KDTree(pos)
        ind, dis = kdt.query_radius(pos, r=dist, return_distance=True, sort_results=True)
        for i in range(len(pos)):
            res_psepos = pos[i]
            local_frame = eg_local_frame[i]
            neigh_index, dis_to_i = ind[i], dis[i]
            raw_pos = pos[neigh_index]
            res_pos = raw_pos - res_psepos
            res_feas = feas[neigh_index]
            rij = res_pos
            Pij = np.concatenate(
                [np.sum(local_frame[k:k + 1] * rij, axis=-1)[:, None] / np.linalg.norm(local_frame[k:k + 1]) for k
                 in range(3)], axis=-1)
            Pij = Pij * np.dot(res_psepos, res_psepos)
            eigen_vals, eigen_vecs = np.linalg.eig(np.cov(res_pos, rowvar=False))
            eigen_vals = np.sort(eigen_vals)[::-1]
            u_0 = eigen_vals

            res_data = {
                'node_feas': res_feas.astype('float32'),
                'pos': res_pos.astype('float32'),
                'u_0': u_0.astype('float32'),
                'Pij': Pij.astype('float32'),
            }
            seq_data.append(res_data)

        return seq_data

    def __len__(self):
        return self.slices[list(self.slices.keys())[0]].size(0) - 1

    def __getitem__(self, idx):

        if isinstance(idx, int):
            data = self.get(idx)
            return data
        elif isinstance(idx, slice):
            return self.__indexing__(range(*idx.indices(len(self))))
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            return self.__indexing__(idx)
        elif torch.is_tensor(idx) and idx.dtype == torch.uint8:
            return self.__indexing__(idx.nonzero())

        raise IndexError(
            'Only integers, slices (`:`) and long or byte tensors are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def shuffle(self, return_perm=False):
        perm = torch.randperm(len(self))
        dataset = self.__indexing__(perm)
        return (dataset, perm) if return_perm is True else dataset

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        # for key in self.data.keys():  # use this in aliyun server
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(
                slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    def __indexing__(self, index):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__ = self.__dict__.copy()
        copy.data, copy.slices = self.collate([self.get(i) for i in index])
        return copy

    def collate(self, data_list):
        keys = data_list[0].keys  # use blow that in aliyun server
        # keys = list(data_list[0].keys())
        data = data_list[0].__class__()
        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            elif isinstance(item[key], int) or isinstance(item[key], float):
                s = slices[key][-1] + 1
            else:
                raise ValueError('Unsupported attribute type.')
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            if torch.is_tensor(data_list[0][key]):
                data[key] = torch.cat(
                    data[key], dim=data.__cat_dim__(key, data_list[0][key]))
            else:
                data[key] = torch.tensor(data[key])
            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices


def predict(model, model_path, query_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # checkpoints = torch.load(model_path, map_location=torch.device('cpu'))
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    model.eval()
    threshold = checkpoints['val_th']
    if len(query_data) % 64 == 1:
        batch_size = 63
    else:
        batch_size = 64
    dataloader = DataLoader(query_data, batch_size=batch_size, shuffle=False)
    pred_score = []
    with torch.no_grad():
        for ii, data in enumerate(dataloader):
            data.to(device)
            score = model(data)
            pred_score += score.tolist()
    pred_score = np.array(pred_score)
    pred_binary = np.abs(np.ceil(pred_score - threshold)).astype('int')

    return threshold, pred_score, pred_binary


def main(query_path, filename, chain_id, ligand_list, fea_num_threads, localtime):
    pdbid = filename.split('.')[0][:4]
    seqid = f'{pdbid}_{chain_id}'
    print(f'1.extracting query chain {chain_id}...')
    SaveChainPDB(chain_id, query_path, filename, seqid)
    print('2.modify the query pdb file...')
    Modify_PDB(f'{query_path}/{seqid}_modified.pdb', f'{query_path}/{seqid}.pdb')
    print('3.extracting PDB_DF...')
    with open('{}/{}.pdb'.format(query_path, seqid), 'r') as f:
        text = f.readlines()
    residue_num = 0
    for line in text:
        if line.startswith('ATOM'):
            residue_type = line[17:20]
            if residue_type not in ["GLY", "ALA", "VAL", "ILE", "LEU", "PHE", "PRO", "MET", "TRP", "CYS",
                                    "SER", "THR", "ASN", "GLN", "TYR", "HIS", "ASP", "GLU", "LYS", "ARG"]:
                print("ERROR: There are mutant residues in your structure!")
                raise ValueError
            residue_num += 1
    if residue_num == 0:
        print('ERROR: Your query chain id "{}" is not in the uploaded structure, please check the chain ID!'.format(
            chain_id))
        raise ValueError
    PDBFeature(seqid, query_path, query_path)
    print('4.extracting local frame...')
    cal_eg_local_frame(query_path, seqid, 20)

    if not os.path.exists('{}/{}.dssp'.format(query_path, seqid)):
        DSSP_code = subprocess.call([DSSP, '-i', '{}/{}.pdb'.format(query_path, seqid),
                                     '-o', '{}/{}.dssp'.format(query_path, seqid)])
    if not os.path.exists('{}/{}.dssp'.format(query_path, seqid)):
        print("ERROR: The upload protein structure is not in correct PDB format, please check the structure!")
        raise ValueError

    # prediction
    csv_ligand = {'PDNA': 'DNA', 'PRNA': 'RNA', 'PP': 'P'}
    print('5.predicting...')
    nlayer_dict = {'PDNA': 6, 'PRNA': 4, 'PP': 4}
    probs_dict = {}
    for ligand in ligand_list:
        if not os.path.exists('{}/{}_fast.resfea'.format(query_path, seqid)):
            if ligand == 'PDNA':
                query_blosum = norm_blosum(query_path, seqid)
                query_dssp = norm_DSSP(query_path, seqid)
                query_atom = norm_atom(query_path, seqid)

                PDBResidueFeature_fast_dna(query_path, seqid, query_blosum, query_dssp, query_atom)
            else:
                query_blosum = norm_blosum(query_path, seqid)
                query_dssp = norm_DSSP(query_path, seqid)
                query_atom = norm_atom(query_path, seqid)
                query_oh = norm_oh(query_path, seqid)

                PDBResidueFeature_fast(query_path, seqid, query_blosum, query_dssp, query_atom, query_oh)

        dist = 20 if ligand in ['PDNA', 'PRNA', 'PP'] else 15
        query_data = seq_Dataset(query_path, seqid, dist)
        nlayer = nlayer_dict[ligand]
        import ModelCode.GN_model_gru_gat as GeoNet
        if ligand in ['PRNA', 'PP']:
            model = GeoNet.GeoNet(edge_aggr=['add'], node_aggr=['add'],
                                  nlayers=nlayer, heads=1, x_ind=62, edge_ind=2, x_hs=128, e_hs=128, u_hs=128,
                                  dropratio=0.5, bias=True, edge_method='radius', r_list=[10], dist=dist,
                                  max_nn=40, apply_edgeattr=True, apply_nodeposemb=True)
        else:
            model = GeoNet.GeoNet(edge_aggr=['add'], node_aggr=['add'],
                                  nlayers=nlayer, heads=1, x_ind=42, edge_ind=2, x_hs=128, e_hs=128, u_hs=128,
                                  dropratio=0.5, bias=True, edge_method='radius', r_list=[10], dist=dist,
                                  max_nn=40, apply_edgeattr=True, apply_nodeposemb=True)
        model_path = GEONET + '/Models/{}/model_fast.pth'.format(ligand)
        threshold, pred_score, pred_binary = predict(model, model_path, query_data)
        probs_dict[ligand] = {'threshold': threshold, 'pred_score': pred_score, 'pred_binary': pred_binary,
                              'residue_id': query_data.res_id_list, 'sequence': query_data.sequence}

        result_df = pd.DataFrame(data={'Residue_ID': query_data.res_id_list, 'Residue': query_data.sequence,
                                       'Probability': pred_score, 'Binary': pred_binary})
        result_df.to_csv('{}/{}_{}_{}-binding_result.csv'.format(query_path, seqid, localtime, csv_ligand[ligand]), float_format='%.3f',
                         columns=["Residue_ID", "Residue", "Probability", "Binary"])
    with open('{}/{}.result'.format(query_path, seqid), 'wb') as f:
        pickle.dump(probs_dict, f)

    text = '4.The results are saved in :\n'
    for ligand in ligand_list:
        text += '{}/{}_{}_{}-binding_result.csv\n'.format(query_path, seqid, localtime, csv_ligand[ligand])
    print(text)

    return


if __name__ == '__main__':

    args = parse_args()
    if args.query_path is None:
        print('ERROR: please --querypath!')
        raise ValueError
    query_path = args.query_path.rstrip('/')
    pdbid = args.pdbid
    fea_num_threads = args.fea_num_threads
    chain_id = args.chain_id
    if not os.path.exists(f'{query_path}/{args.filename}'):
        print(f'PDB file not find, download the {pdbid} file from RCSB...')
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdbid, pdir=f'{query_path}/', file_format='pdb')
        os.system(f'mv {query_path}/pdb{pdbid.lower()}.ent {query_path}/{pdbid}.pdb')
        filename = f'{pdbid}.pdb'
    else:
        filename = args.filename

    ligand_list_ = args.ligand.split(',')
    ligand_list = []

    if not os.path.exists('{}/{}'.format(query_path, filename)):
        print('ERROR: Your query structure "{}/{}" is not found!'.format(query_path, filename))
        raise ValueError

    for ligand_i in ligand_list_:
        if ligand_i not in ['DNA', 'RNA', 'P']:
            print('ERROR: ligand "{}" is not supported by GeoNet!'.format(ligand_i))
            raise ValueError
        else:
            if ligand_i == 'HEME':
                ligand_list.append('PHEM')
            else:
                ligand_list.append('P' + ligand_i)

    p1 = PDBParser(PERMISSIVE=1)
    try:
        structure = p1.get_structure('chain', '{}/{}'.format(query_path, filename))
        a = 0
    except:
        print('ERROR: The query structure "{}/{}" is not in correct PDB format, please check the structure!'.format(
            query_path, filename))
        raise ValueError
    localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    main(query_path, filename, chain_id, ligand_list, fea_num_threads, localtime)
