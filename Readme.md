```
                        Installation and implementation of GeoNet
                                (version 1.0 2024/03/17)

1 Description
    GeoNet is an accurate graph neural network-based predictor for identifying nucleic acid- and protein-binding residues on protein structures.
    GeoNet consists of two modules:
    (1) Constructing local sphere for the target residues from protein structures by integrating the local structural context topology. The residues are nodes and the spatial relationship of residues is employed to define edges. Node features, edge features, and geometric features are extracted to learn the high-level representations for the target residues.
    (2) Geometric Graph Encoder (GGE), which progressively updates the sphere feature vectors to learn effective latent rules for recognizing the binding residues.

2 Installation

2.1 system requirements
    For prediction process, you can predict functional binding residues from a protein structure within a few minutes with CPUs only. However, for training a new deep model from scratch, we recommend using a GPU for significantly faster training.
    To use GeoNet with GPUs, you will need: cuda >= 10.0, cuDNN.

2.2 Create an environment
    GeoNet is built on Python3.
    We highly recommend to use a virtual environment for the installation of GeoNet and its dependencies.

    A virtual environment can be created and (de)activated as follows by using conda(https://conda.io/docs/):
        # create
        $ conda create -n GeoNet_env python=3.6
        # activate
        $ source activate GeoNet_env
        # deactivate
        $ source deactivate
    OR the virtual environment can be created by using virtualenv(https://github.com/pypa/virtualenv/).
        # create
        $ virtualenv GeoNet_env --python=python3.6
        # activate
        $ source GeoNet_env/bin/activate
        # deactivate
        $ deactivate

2.3 Install GeoNet dependencies
    Note: If you are using a Python virtual environment, make sure it is activated before running each command in this guide.

2.3.1 Install requirements
	
	Note: Typical install requirements time on a "normal" desktop computer is 10 minutes. This is an install example, you can also install the latest versions of pytorch and torch-geometric, but please pay attention to compatibility issues between versions when installing.
	
    (1) Install pytorch 1.2.0 (For more details, please refer to https://pytorch.org/)
        For linux:
        # CUDA 10.0
        $ pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
        # CPU only
        $ pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    (2) Install torch_geometric 1.3.1 (For more details, please refer to https://pytorch-geometric.readthedocs.io/en/1.3.1/notes/installation.html)
        $ pip install --no-cache-dir torch-scatter==1.3.1
        $ pip install --no-cache-dir torch-sparse==0.4.0
        $ pip install --no-cache-dir torch-cluster==1.4.4
        $ pip install torch-geometric==1.3.1
    (3) Install other requirements
        $ pip install torchnet==0.0.4
        $ pip install tqdm
        $ pip install prettytable

2.3.2 Install the bioinformatics tools
    (1) Install blast+ for extracting PSSM (position-specific scoring matrix) profiles
    To install ncbi-blast-2.8.1+ and download NR database (ftp://ftp.ncbi.nlm.nih.gov/blast/db/) for psiblast, please refer to BLAST® Help (https://www.ncbi.nlm.nih.gov/books/NBK52640/).
    Set the absolute paths of blast+ and NR databases in the script "scripts/prediction.py".
    (2) Install HHblits for extracting HMM profiles
    To install HHblits and download uniclust30_2018_08 (http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz) for HHblits, please refer to https://github.com/soedinglab/hh-suite.
    Set the absolute paths of HHblits and uniclust30_2018_08 databases in the script "scripts/prediction.py".
    (3) Install DSSP for extracting SS (Secondary structure) profiles
    DSSP is contained in "scripts/dssp", and it should be given executable permission by:
        $ chmod +x scripts/dssp

    Note: Difference versions of blast+, HHblits and their databases may result in slightly different PSSM and HMM profiles, leading to slight different predictions.
    Typical download databases and install bioinformatics tools time on a "normal" desktop computer is 10 hours.



3 Usage

3.1 Predict functional binding residues from a protein structure(in PDB format) based on trained deep models

    Example:
        $ cd scripts
        $ python prediction.py --querypath ../output/example --filename 6ide_B.pdb --pdbid 6ide --chainid B --ligand DNA --cpu 10

    Output:
    The result named after "{ligand}-binding_result.csv" is saved in {querypath}. The five columns are represented residue index, residue sequence number in PDB, residue name, the probability of binding residue and the binary prediction category(1:binding residue, 0:non-bindind residue), respectively.
    The expected outputs for the demo are saved in ../output/example/.

    Note: Expected run time for the demo on a "normal" desktop computer is 10 minutes.

    The list of commands:
    --querypath         The path of query structure
    --filename          The file name of the query structure which should be in PDB format.
    --chainid           The query chain id (case sensitive). If there is only one chain in your query structure, you can leave it blank.(default='')
    --ligands           Ligand types. Multiple ligands should be separated by commas. You can choose from DNA,RNA,PP.(default=DNA)
    --cpu               The number of CPUs used for calculating PSSM and HMM profile.(default=1)


3.2 Train a new deep model from scratch

3.2.1 Download the datasets used in GeoNet.

	Donload the PDB files and the feature files (the PSSM profiles, HMM profiles, and the DSSP profiles) from http: and store the PDB files in the path of the corresponding data.
	Example:
		The PDB files of DNA data should be stored in ../Datasets/customed_data/PDNA/PDB, and the features file shuld be stored in ../Datasets/customed_data/PDNA/feature.

3.2.2 Modify the PDB files.

	Example:
		$ cd script
		$ python modified_data_script.py --ligand DNA
	
	Output:
	Tha modified PDB files are saved in ../Datasets/customed_data/P{ligand}/modified_data/PDB

3.2.3 Generate the training, validation and test data sets from original data sets

    Example:
        $ cd scripts
        # demo 1
        $ python data_io.py --ligand DNA --psepos SC --features PSSM,HMM,SS,AF --context_radius 20
        # demo 2
        $ python data_io.py --ligand RNA --psepos SC --features PSSM,HMM,SS,AF --context_radius 20

    Output:
    The data sets are saved in ../Datasets/customed_data/P{ligand}/modified_data/P{ligand}_{psepos}_dist{context_radius}_{featurecode}.

    Note: {featurecode} is the combination of the first letter of {features}.
    Expected run time for the demo 1 and demo 2 on a "normal" desktop computer are 30 and 40 minutes, respectively.

    The list of commands:
    --ligand            A ligand type. It can be chosen from DNA,RNA,P.
    --psepos            Pseudo position of residues. SC, CA, C stand for centroid of side chain, alpha-C atom and centroid of residue, respectively.(default=SC)
    --features          Feature groups. Multiple features should be separated by commas. You can combine features from PSSM, HMM, SS(secondary structure) and AF(atom features).(default=PSSM,HMM,SS,AF)
    --context_radius    Radius of structure context.
    --tvseed            The random seed used to separate the validation set from training set.(default=1995)


3.2.4 Train the deep model

    Example:
        $ cd scripts
        # demo 1
        $ python training_gat.py --ligand DNA --psepos SC --features PSSM,HMM,SS,AF --context_radius 20 --edge_radius 10 --apply_edgeattr True --apply_posemb True --aggr sum --nlayers 4
        # demo 2
        $ python training_gat.py --ligand RNA --psepos SC --features PSSM,HMM,SS,AF --context_radius 20 --edge_radius 10 --apply_edgeattr True --apply_posemb True --nlayers 4

    Output:
    The trained model is saved in ../Datasets/P{ligand}/checkpoints/{starttime}.
    The log file of training details is saved in ../Datasets/P{ligand}/checkpoints/{starttime}/training.log.

    Note: {starttime} is the time when training.py started be executed.
    Expected run time for demo 1 and demo 2 on a "normal" desktop computer with a GPU are 30 and 12 hours, respectively.

    The list of commands:
    --ligand            A ligand type. It can be chosen from DNA,RNA,CA,MG,MN,ATP,HEME.
    --psepos            Pseudo position of residues. SC, CA, C stand for centroid of side chain, alpha-C atom and centroid of residue, respectively.(default=SC)
    --features          Feature groups. Multiple features should be separated by commas. You can combine features from PSSM, HMM, SS(secondary structure), AF(atom features) and OH (one-hot encoding features).(default=PSSM,HMM,SS,AF)
    --context_radius    Radius of structure context.
    --edge_radius       Radius of the neighborhood of a node. It should be smaller than radius of structure context.(default=20)
    --apply_edgeattr    Use the edge feature vectors or not.(default=True)
    --apply_posemb      Use the relative distance from every node to the central node as position embedding of nodes or not.(default=True)
    --aggr              The aggregation operation in node update module and graph update module. You can choose from sum and max.(default=sum)
    --hidden_size       The dimension of encoded edge, node and graph feature vector.(default=64)
    --nlayers         	The number of Geometric Graph Encoder (GGE).(default=4)
    --lr                Learning rate for training the deep model.(default=0.00005)
    --batch_size        Batch size for training deep model.(default=64)
    --epoch             Training epochs.(default=30)


4 Frequently Asked Questions
(1) If the script is interrupted by "Segmentation fault (core dumped)" when torch of CUDA version is used, it may be raised because the version of gcc (our version of gcc is 5.5.0) and you can try to set CUDA_VISIBLE_DEVICES to CPU before execute the script to avoid it by:
        $ export CUDA_VISIBLE_DEVICES="-1"
(2) If your CUDA version is not 10.0, please refer to the homepages of Pytorch(https://pytorch.org/) and torch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/) to make sure that the installed dependencies match the CUDA version. Otherwise, the environment could be problematic due to the inconsistency.

5 How to cite GeoNet?

   If you are using the GeoNet program, you can cite:
   
```

