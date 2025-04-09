# making multiple single chains and energy min
def generate_single_chains(monomer_file, n_chains, dp_n, p_tact, work_dir, chain_end_control=True, name='single_chain', ter_smiles='*C', spring_strength=10.0, log='full_process.log', quiet=True):
    # load monomer
    monomer = Chem.SDMolSupplier(monomer_file, removeHs=False)[0]
    ter = utils.mol_from_smiles(ter_smiles)
    # load model chain charges
    with open("input/head_charges.json", "r") as f: head_charge = json.load(f)
    with open("input/body_charges.json", "r") as f: body_charge = json.load(f)
    with open("input/tail_charges.json", "r") as f: tail_charge = json.load(f)

    # make working directory and switch to it
    og_dir = os.getcwd()
    if not quiet: print("Current working directory:", og_dir)
    os.makedirs(work_dir, exist_ok=True)
    if not quiet: print('Changing working directory to:', work_dir)
    os.chdir(work_dir)

    # create a log file for the whole process
    process_log = open(log, 'w')

    try:
        for n in range(n_chains):
            # set up all file paths
            sdf_file = name + str(n) + '.sdf' # sdf mol file output
            data_file = name + str(n) + '.data' # lammps data file output
            template = '/home/ggolde/projects/polyheatline/dma_tacticity_clean/input/single_chain_template.lammps' # path to lammps input template
            data_file_relative = name + str(n) + '.data' # path to data file relative to script
            script_file = name + str(n) + '.input.lammps' # lammps script file output
            log_file = name + str(n) + '.log.lammps' # relative path of log file
            data_output_file = name + str(n) + '.output.data' # minimized output file

            # make n=dp_n polymer with tacticity and terminate it
            tac_arr = make_chain_end_tacticity_arr(dp_n, p_tact) if chain_end_control else make_enantiomorphic_site_tacticity_arr(dp_n, p_tact)
            polymer = poly.polymerize_mols(monomer, n=dp_n, tacticity='manual', tac_array=tac_arr)
            polymer = poly.terminate_mols(polymer, ter)
            
            # assign charges
            enforce_charges(polymer, head_charge, body_charge, tail_charge)

            # assign force feild parameters
            ff = GAFF2_mod()
            result = ff.ff_assign(polymer)

            # transform polymer coordinates to center and align chain on z-axis
            transform_polymer_coords_to_zaxis(polymer)

            # save RDkit object
            write_mol(polymer, sdf_file)

            # save lammps data file
            lammps.MolToLAMMPSdata(polymer, data_file)

            # create lammps input file
            make_lammps_script_single_chain(polymer, template, script_file, data_file_relative, data_output_file, log_file, spring_strength)

            # run the lammps script
            subprocess.run(['mpirun', '-np', '8', 'lmp', '-in', script_file], stdout=process_log)
        if not quiet: print(f'Successfuly generated {n_chains} of DPn {dp_n} with tacticity control {p_tact}!')
    
    finally:
        process_log.close()
        if not quiet: print('Changing working directory back to:', og_dir)
        os.chdir(og_dir)

    return 0

# !FLAG
def enforce_charges(polymer, head_charge, body_charge, tail_charge):
    """
    assign saved charges to polymer
    """
    n_atoms = len(polymer.GetAtoms())
    dp_n = 2+(n_atoms-len(head_charge)-len(tail_charge))/len(body_charge)
    if dp_n % 1 != 0:
        raise RuntimeError('Length of charges do not match length of polymer')
    dp_n = int(dp_n)
    body_extended = copy.deepcopy(body_charge)
    for i in range(1,(dp_n-2)):
        body_extended = body_extended + body_charge 
    charge_list = head_charge + body_extended + tail_charge
    for i, charge in enumerate(charge_list):
        polymer.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', charge)
    return

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lammps_logfile import File as read_lammps_log
from lammps_logfile import running_mean
from scipy.stats import ttest_ind, linregress
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from ..sim import lammps
import matplotlib.ticker as ticker

def stats_analysis(x, data):
    range = (data.min(), data.max())
    domain = (x.min(), x.max())
    mean = data.mean()
    std = data.std()
    slope, intercept, _, _, _ = linregress(x, data)
    return {'mean':mean, 'std':std, 'slope':slope, 'intercept':intercept, 'range': range, 'domain':domain}

def data_from_lammps_log(log_path, props, regional):
    log = read_lammps_log(log_path)
    time = log.get('Time')
    data = {}
    metadata = {}
    for prop in props:
        x = time if prop not in regional.keys() else time[regional[prop]:]
        y = log.get(prop) if prop not in regional.keys() else log.get(prop)[regional[prop]:]
        data[prop] = y
        metadata[prop] = stats_analysis(x, y)
    return pd.DataFrame(metadata).T, data

def calculate_msd_per_dihedral(molecule, backbone_indexes):
    # only calculates monomer to monomer bond dihedrals
    # Ensure there are enough atoms for dihedral angles
    if len(backbone_indexes) < 4:
        raise ValueError("Backbone must contain at least 4 atoms to define dihedral angles.")

    # Get the molecule's conformer
    conf = molecule.GetConformer()
    if not conf.Is3D():
        raise ValueError("Molecule must have 3D coordinates to calculate dihedral angles.")

    # Calculate all dihedral angles in the backbone
    dihedral_angles = []
    for i in range(len(backbone_indexes) - 3):
        idx1, idx2, idx3, idx4 = backbone_indexes[i:i+4]
        angle = rdMolTransforms.GetDihedralDeg(conf, idx1, idx2, idx3, idx4)
        dihedral_angles.append(angle)

    # take only odd indexes from dihedral angles (unit to unit angles)
    dihedral_angles = dihedral_angles[1::2]

    # Compute mean squared deviation from 180°
    dihedral_deviation = [(abs(angle) - 180.0) ** 2 for angle in dihedral_angles]
    mean_squared_deviation = np.mean(dihedral_deviation)

    return mean_squared_deviation / len(dihedral_angles)

def get_carbon_backbone_indexes_PDMA(mol):
    backbone_indexes = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6: 
            neigh_sig = [neigh.GetAtomicNum() for neigh in atom.GetNeighbors()]
            ter = (neigh_sig == [1,1,1,6])
            tail = (neigh_sig == [6,1,1,6])
            head = (neigh_sig == [6,6,1,6])
            if ter or tail or head:
                backbone_indexes.append(atom.GetIdx())
    return backbone_indexes

def calc_bb_indexs_PDMA(dp_n, n_chains):
    backbone_indexes = []
    n_atoms_per = dp_n*16 + 8
    for n in range(n_chains):
        bb_per = [x for i in np.arange(4, n_atoms_per-4, step=16) for x in (i, i+1)]
        bb_per.insert(0,0)
        bb_per.append(1604)
        bb_per = list(np.array(bb_per)+(n_atoms_per*n))
        bb_per = [int(x) for x in bb_per]
        backbone_indexes.append(bb_per)
    return backbone_indexes

def compute_relative_shape_anisotropy(positions):
    """
    Compute the relative shape anisotropy of a polymer chain.

    Parameters:
    positions (numpy.ndarray): Array of atomic positions with shape (N, 3),
                                where N is the number of backbone atoms.

    Returns:
    float: Relative shape anisotropy (0=sphere, 1=line).
    """
    # Center the positions by subtracting the center of mass
    com = np.mean(positions, axis=0)
    centered_positions = positions - com

    # Compute the gyration tensor
    gyration_tensor = np.zeros((3, 3))
    for pos in centered_positions:
        gyration_tensor += np.outer(pos, pos)
    gyration_tensor /= len(centered_positions)

    # Compute eigenvalues of the gyration tensor
    eigenvalues = np.linalg.eigvalsh(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort eigenvalues in descending order

    # Compute shape anisotropy
    lambda1, lambda2, lambda3 = eigenvalues
    top = lambda1**2 + lambda2**2 + lambda3**2
    bottom = (lambda1 + lambda2 + lambda3)**2
    anisotropy = (3/2)*(top/bottom)-(1/2)

    return anisotropy

def herman_orientation_factor(backbone_positions, z_ref=True):
    """
    Calculates the Herman orientation factor relative to the average chain bond vector.

    Parameters:
    -----------
    backbone_positions : np.ndarray
        A 2D array of shape (N, 3), where N is the number of backbone atoms.
        Each row represents the x, y, z coordinates of a backbone atom.

    Returns:
    --------
    float
        The Herman orientation factor.
    """
    # Compute bond vectors between consecutive backbone atoms
    bond_vectors = np.diff(backbone_positions, axis=0)

    # Normalize bond vectors to unit vectors
    bond_vectors_norm = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]

    # Calculate the reference vector
    avg_bond_vector = np.mean(bond_vectors_norm, axis=0)
    avg_bond_vector /= np.linalg.norm(avg_bond_vector)
    if z_ref:
        ref_vec = np.array([0,0,1])
    else:
        ref_vec = avg_bond_vector    
    
    # Ensure all bond vectors point in the same general direction as the average bond vector
    dot_products = np.dot(bond_vectors_norm, ref_vec)
    bond_vectors_norm[dot_products < 0] *= -1

    # Compute the cosine squared of the angle between each bond vector and the average bond vector
    cos_squared = np.dot(bond_vectors_norm, ref_vec) ** 2

    # Compute the Herman orientation factor
    herman_factor = 1.5 * np.mean(cos_squared) - 0.5

    return herman_factor

# function to write RDkit mol as sdf or XYZ
def write_mol(mol, path, ext='sdf'):
    if ext == 'sdf':
        with Chem.SDWriter(path) as writer:
            writer.write(mol)
    elif ext == 'xyz':
        Chem.MolToXYZFile(mol, path)
    else:
        raise RuntimeWarning('Invalid extension specified')
    return

def data_from_cell(cell, dp_n, n_chains):
    bbs_idxs = calc_bb_indexs_PDMA(dp_n, n_chains)
    bbs_idxs_flat = [x for xs in bbs_idxs for x in xs]
    bbs_coords_all = cell.GetConformer().GetPositions()[bbs_idxs_flat]

    single_chain_data = {}
    for i, bb_idxs in enumerate(bbs_idxs):
        bb_coords = cell.GetConformer().GetPositions()[bb_idxs]
        single_chain_data[i] = {'dihedral_msd':calculate_msd_per_dihedral(cell, bb_idxs),
                               'anisotropy': compute_relative_shape_anisotropy(bb_coords)}
    herman_z = herman_orientation_factor(bbs_coords_all, z_ref=True)
    herman_avg = herman_orientation_factor(bbs_coords_all, z_ref=False)

    data = pd.DataFrame(single_chain_data).T
    summary = pd.DataFrame([data.mean(), data.std()], index=['mean', 'std'])
    summary['herman_z'] = [herman_z, 0.0]
    summary['herman_avg'] = [herman_avg, 0.0]
    return summary, data

class Trial:
    def __init__(self, dir, dp_n=100, n_chains=9,
                 eq_props=['Step', 'TotEng', 'Temp', 'Press', 'Density'], eq_cutoff=500, 
                 tc_props=['Step', 'TotEng', 'Temp', 'Press', 'dTemp_step', 'kappa_inst'], kappa_cutoff=341, 
                 eq_log='md.log', tc_log='tc.log', final_cell='final.data'):
        if dir[-1] != '/':
            dir = dir + '/'
        self.dir=dir
        self.name = dir.split('/')[-2]
        self.kappa_cutoff = kappa_cutoff
        self.eq_summary, self.eq_data = data_from_lammps_log((dir+eq_log), eq_props, {prop:eq_cutoff for prop in eq_props})
        self.tc_summary, self.tc_data = data_from_lammps_log((dir+tc_log), tc_props, {'kappa_inst':kappa_cutoff})
        self.cell = lammps.MolFromLAMMPSdata((dir+final_cell))
        self.align_summary, self.align_data = data_from_cell(self.cell, dp_n, n_chains)

    def get(self, which, prop, col):
        match which:
            case 'eq':
                return self.eq_summary.loc[prop, col]
            case 'tc':
                return self.tc_summary.loc[prop, col]
            case 'align':
                return self.align_summary.loc[col, prop]
            case _:
                raise ValueError('which value invalid')
    
    def display(self, which, dir_tact_name=None):
        data = self.eq_data if which == 'eq' else self.tc_data
        stats = self.eq_summary if which == 'eq' else self.tc_summary
        props = [['Density', 'TotEng'], ['Temp', 'Press']] if which == 'eq' else [['kappa_inst', 'dTemp_step'], ['Temp', 'TotEng']]
        step = data['Step']

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(self.dir + ' ' + which)
        for i in range(2):
            for j in range(2):
                prop = props[i][j]
                mean = stats.loc[prop, 'mean']
                std = stats.loc[prop, 'std']
                slope = stats.loc[prop, 'slope']
                intercept = stats.loc[prop, 'intercept']
                step = data['Step'] if prop != 'kappa_inst' else data['Step'][self.kappa_cutoff:]

                axs[i,j].plot(step, data[prop], zorder=1)
                axs[i,j].plot(step, slope*step+intercept, zorder=2, linestyle='--', color='firebrick', label=f'Slope={slope:,.5f}')
                axs[i,j].axhline(mean, color="black", linestyle="--", zorder=3)
                axs[i,j].axhspan(mean-std, mean+std, color="green", alpha=0.3, linewidth=2, zorder=0)
                axs[i,j].set_title(f'{prop} ({mean:,.2f} +/- {std:,.2f})')
                axs[i,j].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
                axs[i,j].legend()
        
        if dir_tact_name is not None:
            plt.savefig(dir_tact_name + '_' + self.name + '_' + which + '.png')

class AvgTrial:
    def __init__(self, dir, tact=None, dp_n=100, n_chains=9,
                eq_props=['Step', 'TotEng', 'Temp', 'Press', 'Density'], eq_cutoff=500, 
                tc_props=['Step', 'TotEng', 'Temp', 'Press', 'dTemp_step', 'kappa_inst'], kappa_cutoff=341, 
                eq_log='md.log', tc_log='tc.log', final_cell='final.data'):
        
        if dir[-1] != '/':
            dir = dir + '/'
        self.dir = dir
        self.name = dir.split('/')[-2]
        self.tact = tact

        trial1 = Trial(dir+'trial1/', dp_n, n_chains,
                            eq_props=eq_props, eq_cutoff=eq_cutoff, 
                            tc_props=tc_props, kappa_cutoff=kappa_cutoff, 
                            eq_log=eq_log, tc_log=tc_log, final_cell=final_cell)
        trial2 = Trial(dir+'trial2/', dp_n, n_chains,
                            eq_props=eq_props, eq_cutoff=eq_cutoff, 
                            tc_props=tc_props, kappa_cutoff=kappa_cutoff, 
                            eq_log=eq_log, tc_log=tc_log, final_cell=final_cell)
        trial3 = Trial(dir+'trial3/', dp_n, n_chains,
                            eq_props=eq_props, eq_cutoff=eq_cutoff, 
                            tc_props=tc_props, kappa_cutoff=kappa_cutoff, 
                            eq_log=eq_log, tc_log=tc_log, final_cell=final_cell)
        
        self.trials = [trial1, trial2, trial3]

        self.compute_metrics()

    def compute_metrics(self):
        metric_names = ['Density', 'kappa_inst', 'dihedral_msd', 'anisotropy', 'herman_z', 'herman_avg']
        which = ['eq', 'tc', 'align', 'align', 'align', 'align']
        metrics = {}

        for which_data, prop in zip(which, metric_names):
            avgs = np.array([trial.get(which_data, prop, 'mean') for trial in self.trials])
            stds = np.array([trial.get(which_data, prop, 'std') for trial in self.trials])
            propagation_err = np.sqrt(np.power(stds,2).sum())
            sample_std = avgs.std()
            metrics[prop] = {'mean':avgs.mean(), 'std':sample_std, 'err':propagation_err,
                             'trial1':avgs[0], 'trial2':avgs[1], 'trial3':avgs[2],}

        self.metrics = pd.DataFrame(metrics).T
    
    def get_metrics(self):
        return self.metrics
    
    def save_md_summaries(self, save_dir):
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'

        for trial in self.trials:
            trial.display('eq', save_dir + self.name)
            trial.display('tc', save_dir + self.name)

class TactControl:
    def __init__(self, dir, tacts, dp_n=100, n_chains=9,
                eq_props=['Step', 'TotEng', 'Temp', 'Press', 'Density'], eq_cutoff=500, 
                tc_props=['Step', 'TotEng', 'Temp', 'Press', 'dTemp_step', 'kappa_inst'], kappa_cutoff=341, 
                eq_log='md.log', tc_log='tc.log', final_cell='final.data'):

        if dir[-1] != '/':
            dir = dir + '/'
        self.dir = dir
        self.name = dir.split('/')[-2]
        self.tacts = tacts
        self.tact_names = ['tact' + str(int(num*100)) for num in tacts]
        self.avgtrials = {}
        for tact_str, tact in zip(self.tact_names, tacts):
            self.avgtrials[tact_str] = AvgTrial(dir+tact_str, tact=tact, dp_n=dp_n, n_chains=n_chains, 
                                            eq_props=eq_props, eq_cutoff=eq_cutoff, 
                                            tc_props=tc_props, kappa_cutoff=kappa_cutoff, 
                                            eq_log=eq_log, tc_log=tc_log, final_cell=final_cell)
        self.build_data()
        
    def build_data(self):
        metric_names = ['Density', 'kappa_inst', 'dihedral_msd', 'anisotropy', 'herman_z', 'herman_avg']
        data = {}        
        for metric in metric_names:
            nested = {}
            for key, avgtrial in self.avgtrials.items():
                trial_data = avgtrial.get_metrics()
                nested[key] = {'tact':avgtrial.tact, 'mean':trial_data.loc[metric, 'mean'], 'std':trial_data.loc[metric, 'std']}
            nested = pd.DataFrame(nested).T
            data[metric] = nested
        self.data = data
    
    def export_md_summaries(self, save_dir):
        for key, avgtrial in self.avgtrials.items():
            avgtrial.save_md_summaries(save_dir)