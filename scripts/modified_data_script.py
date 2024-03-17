import argparse
import os, shutil, sys
import pickle

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import pandas as pd
import numpy as np
from tqdm import tqdm
from scripts.data_io import def_atom_features, tv_split, StatisticsSampleNum


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", default='RNA',
                        help="A ligand type. It can be chosen from DNA,RNA,CA,MG,MN,ATP,HEME.")
    parser.add_argument("--tvseed", dest='tvseed', type=int, default=1995,
                        help='The random seed used to separate the validation set from training set.')
    return parser.parse_args()


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
        os.makedirs(PDB_chain_dir)

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
        # pdb_data = get_pdb_data(file_path)
        if not os.path.exists(f'{PDB_chain_dir}/{seq_id}.pdb'):
            try:
                pdb_data = get_pdb_data(file_path)
                with open(f'{PDB_chain_dir}/{seq_id}.pdb', 'w') as f:
                    f.write(pdb_data)
            except KeyError:
                print('ERROR: Modify PDB file in ', seq_id)
                raise KeyError

    return


if __name__ == '__main__':
    args = parse_args()
    ligand = 'P' + args.ligand if args.ligand != 'HEME' else 'PHEM'
    # Dataset_dir = os.path.abspath('..') + '/Datasets' + '/' + ligand + '/modified_data'
    # Dataset_dir = os.path.abspath('..') + '/Datasets/customed_data/' + '/' + ligand + '/modified_data'
    # Dataset_dir = os.path.abspath('..') + '/Datasets' + '/' + ligand + '/GeoBind_data/modified_data'
    Dataset_dir = os.path.abspath('..') + '/Datasets/customed_data' + '/' + ligand
    # Dataset_dir = os.path.abspath('..') + '/pair_data' + '/' + ligand + '/modified_data/'
    PDB_chain_dir = f'{Dataset_dir}/modified_data/PDB'

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

    trainset_anno = f'{Dataset_dir}/{trainingset_dict[ligand]}'
    testset_anno = f'{Dataset_dir}/{testset_dict[ligand]}'

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

    # train_list, test_list = tv_split(train_list, args.tvseed)
    # train_list, valid_list = tv_split(train_list, args.tvseed)
    # StatisticsSampleNum(train_list, valid_list, test_list, seqanno)
    # seqlist = train_list + valid_list + test_list
    #
    # with open(f'{Dataset_dir}/train_valid_test.pkl', 'wb') as f:
    #     pickle.dump([train_list, valid_list, test_list], f)

    # train_list, valid_list, test_list = pickle.load(open(f'{Dataset_dir}/train_valid_test.pkl', 'rb'))
    # seqlist = train_list + valid_list + test_list
    # seqanno = pickle.load(open(f'{Dataset_dir}/seqanno.pkl', 'rb'))
    # StatisticsSampleNum(train_list, valid_list, test_list, seqanno)

    # PDB_raw_dir = os.path.abspath('..') + '/Datasets' + '/' + ligand + '/GeoBind_data/PDB'
    PDB_raw_dir = os.path.abspath('..') + '/Datasets/customed_data' + '/' + ligand + '/PDB'
    # PDB_raw_dir = os.path.abspath('..') + '/pair_data' + '/' + ligand + '/PDB'
    print('1.Modify the PDB information.')
    cal_PDB_Modify(seqlist, PDB_chain_dir, PDB_raw_dir)

    print('Done')
