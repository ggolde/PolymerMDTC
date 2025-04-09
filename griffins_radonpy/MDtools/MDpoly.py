import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from ..core import poly, utils, calc
from ..sim import qm, lammps
from ..ff.gaff2_mod import GAFF2_mod
import copy
import os
import subprocess
import matplotlib.pyplot as plt
import json

def calc_n_atoms(dp_n, monomer, ter, n_chains):
    """
    Calculate # of atoms in polymer

    Parameters:
    dp_n (int): degree of polymerization
    monomer (RDkit mol object): monomer molecule
    ter (RDkit mol object): terminator molecule
    n_chains (int): number of chains in simulation cell

    Returns:
    int: number of atoms
    """
    natom_mono = monomer.GetNumAtoms()-2
    natom_ter = (ter.GetNumAtoms()-1) * 2
    n_atoms = ((dp_n*natom_mono)+(natom_ter))*n_chains
    return n_atoms

def write_mol(mol, path, confId=0, ext='sdf'):
    """
    Writes RDkit mol object to .sdf or .xyz file

    Parameters:
    mol (RDkit mol object): mol to write
    path (string): path to save file
    ext (string): sdf or xyz, the format of desired file

    Returns:
    None
    """
    if ext == 'sdf':
        with Chem.SDWriter(path) as writer:
            writer.write(mol, confId=confId)
    elif ext == 'xyz':
        Chem.MolToXYZFile(mol, path, confId=confId)
    else:
        raise RuntimeWarning('Invalid extension specified')
    return

def read_mol(file_path):
    return Chem.SDMolSupplier(file_path, removeHs=False)[0]

def sample_schulz_zimm(n, Mn, PDI):
    """
    Sample n polymer chain lengths from the Schulz–Zimm distribution,
    parameterized to have an average degree of polymerization Mn and
    a specified polydispersity index (PDI = Mw/Mn).

    The Schulz–Zimm distribution can be written as:
      P(N) = ((z+1)**(z+1)/Gamma(z+1)) * (N**z)/(Mn**(z+1)) * exp[-(z+1)*N/Mn]
    with z = 1/(PDI - 1) - 1.

    This is equivalent to sampling from a Gamma distribution with:
      shape k = z + 1,
      scale theta = Mn / (z + 1).

    Parameters:
      n    : int
             Number of polymer chains to generate.
      Mn   : float
             Target average chain length (degree of polymerization).
      PDI  : float
             Target polydispersity index (Mw/Mn), must be > 1.
             
    Returns:
      chain_lengths : numpy array of ints
             Sampled chain lengths (each >= 1).
    """
    if PDI <= 1:
        raise ValueError("PDI must be greater than 1 for a polydisperse system.")
    
    # Compute z from PDI = (z+2)/(z+1)
    z = 1.0 / (PDI - 1.0) - 1.0
    k = z + 1.0          # Gamma shape parameter
    theta = Mn / (z + 1.0)  # Gamma scale parameter
    
    # Sample from gamma distribution
    lengths = np.random.gamma(shape=k, scale=theta, size=n)
    
    # Round to nearest integer and enforce a minimum chain length of 1.
    chain_Mns = np.maximum(lengths, 1)
    return chain_Mns

def sample_small_polydisperse_chains(n, PDI, DPn, MonoMw=99.13, n_iter=10000, DPn_tol=5, PDI_tol=0.025):
    """
    For a small number of chains (e.g., 10 < n < 20), randomly sample candidate sets
    from the Schulz–Zimm distribution and select the set for which the sample mean and 
    sample PDI are closest to the specified Mn and PDI.
    
    The sample PDI is computed as:
       PDI_sample = (sum(N^2) / sum(N)) / (mean of N)
       
    Parameters:
      n      : int
               Number of polymer chains to generate.
      Mn     : float
               Desired average chain length.
      PDI    : float
               Desired polydispersity index.
      n_iter : int, optional
               Number of candidate sets to generate.
      DPn_tol: float, optional
               Tolerance on the DPn error at which to stop early.
      PDI_tol: float, optional
               Tolerance on the DPI error at which to stop early.
               
    Returns:
      best_sample : numpy array of ints
               The candidate set of chain lengths (length n) with sample moments closest to target.
      (DPn_error, PDI_error)  
                  : float
               The errors of the best candidate.
    """
    Mn = DPn * MonoMw
    best_error = np.inf
    best_sample = None
    
    for i in range(n_iter):
        sample = sample_schulz_zimm(n, Mn, PDI)
        sample_mean = np.mean(sample)
        sample_DPn = np.round(sample_mean/MonoMw).astype(int)
        sample_Mw = np.sum(sample**2) / np.sum(sample)  # weight-average chain length
        sample_PDI = sample_Mw / sample_mean

        DPn_error = sample_DPn - DPn
        PDI_error = sample_PDI - PDI
        
        error = abs(DPn_error)/100 + abs(PDI_error)
        if error < best_error:
            best_error = error
            best_sample = sample.copy()
        if abs(DPn_error) < DPn_tol and abs(PDI_error) < PDI_tol:
            break

    best_sample = np.round(best_sample/MonoMw).astype(int)
            
    return list(best_sample), (DPn_error, PDI_error)

def align_z_axis_to_vec(points, new_z_vector):
    """
    change z-axis to chain vector
    """
    # Ensure the input is a numpy array for easy manipulation
    points = np.array(points)
    # Normalize the new z-axis vector
    new_z_vector = np.array(new_z_vector)
    new_z_vector /= np.linalg.norm(new_z_vector)
    # Create an arbitrary y-axis vector that is not collinear with the new z-axis
    if np.allclose(new_z_vector, [0, 0, 1.0]):
        arbitrary_y_vector = np.array([1.0, 0, 0])  # Choose the x-axis
    else:
        arbitrary_y_vector = np.cross(new_z_vector, [0, 0, 1])
    # Normalize the arbitrary y-axis vector
    arbitrary_y_vector /= np.linalg.norm(arbitrary_y_vector)
    # Calculate the new x-axis as the cross product of y and z
    new_x_vector = np.cross(arbitrary_y_vector, new_z_vector)
    # Create the rotation matrix
    rotation_matrix = np.array([new_x_vector, arbitrary_y_vector, new_z_vector]).T
    # Transform the points
    transformed_points = points @ rotation_matrix
    return transformed_points

def transform_polymer_coords_to_zaxis(polymer, term_idxs=None):
    """
    change polymer coordinates to align chain with z-axis (better ways see offset_xy_coords())
    """
    conformer = polymer.GetConformer()
    coords = np.array([np.array([conformer.GetAtomPosition(i).x,
                    conformer.GetAtomPosition(i).y,
                    conformer.GetAtomPosition(i).z])
                for i in range(polymer.GetNumAtoms())])

    coords = coords - coords[0]

    term_idx_1 = polymer.GetIntProp('terminal_idx1') if polymer.HasProp('terminal_idx1') else term_idxs[0]
    term_idx_2 = polymer.GetIntProp('terminal_idx2') if polymer.HasProp('terminal_idx2') else term_idxs[1]
    chain_vector = (coords[term_idx_2] - coords[term_idx_1])

    adj_coords = align_z_axis_to_vec(coords, chain_vector)

    for i, (x, y, z) in enumerate(adj_coords):
        conformer.SetAtomPosition(i, Point3D(x, y, z))

def offset_xyz_coords(mol, dx, dy, dz):
    """
    offsets the coords of a molecule by dx, dy, dz
    """
    conformer = mol.GetConformer()
    offset_arr = np.array([dx, dy, dz])
    offset_coords = (conformer.GetPositions() + offset_arr)
    conformer.SetPositions(offset_coords)

def infer_term_idxs(monomer, polymer, ter, n=1):
    """
    infer the terminal atoms of a system (currently only tested on methyl terminated)
    """
    natom_poly=polymer.GetNumAtoms()
    term_idx1 = 1
    term_idx2 = natom_poly - ter.GetNumAtoms() + 2
    idxs = [term_idx1, term_idx2]
    idxs_str = str(term_idx1) + ' ' + str(term_idx2)
    if n==1:
        return idxs
    elif n > 1:
        for i in range(n-1):
            new_term1 = term_idx1+((i+1)*natom_poly)
            new_term2 = term_idx2+((i+1)*natom_poly)
            idxs_str = idxs_str + ' ' + str(new_term1) + ' ' + str(new_term2)
        return idxs_str
    
def make_chain_end_tacticity_arr(n, p_meso):
    """
    generates boolean array of length n and models chain end control
    """
    tacticity = np.zeros(shape=(n), dtype=bool)
    tacticity[0] = np.random.choice([True, False])
    for i in range(1,n):
        if np.random.uniform(0, 1) <= p_meso:
            tacticity[i] = tacticity[i-1]
        else:
            tacticity[i] = not tacticity[i-1]
    return tacticity

def make_enantiomorphic_site_tacticity_arr(n, p_iso):
    """
    WIP!
    generates boolean array of length n and models enantiomorphic site control
    """
    tacticity = np.zeros(shape=(n), dtype=bool)
    tacticity[0] = False
    for i in range(1,n):
        if np.random.uniform(0, 1) <= p_iso:
            tacticity[i] = False
        else:
            tacticity[i] = True
    return tacticity

def enforce_monomer_charges(monomer, charge_json):
    """
    assign saved charges to monomer
    """
    n_atoms = len(monomer.GetAtoms())
    with open(charge_json, 'r') as f: charges = json.load(f)
    
    if n_atoms != len(charges):
        raise RuntimeError('Length of charge file do not match number of atoms in the monomer file.')
    
    for atom, charge in zip(monomer.GetAtoms(), charges):
        atom.SetDoubleProp('AtomicCharge', charge)

# depreciated
def make_lammps_script_single_chain(polymer, template_file, output_file, data_file, data_output_file='single_chain.min.data', log_file='default.log.lammps', spring_strength=10.0):
    """
    make a lammps script for single chain equilbration from template
    """
    # get terminal atom indicies
    termC_idx1 = polymer.GetIntProp('terminal_idx1')
    termC_idx2 = polymer.GetIntProp('terminal_idx2')
    # make strings from them
    head_text = ''
    for idx in range(termC_idx1, termC_idx1+4): head_text = head_text + str(idx) + ' '
    tail_text = ''
    for idx in range(termC_idx2, termC_idx2+4): tail_text = tail_text + str(idx) + ' '
    # read the template file
    with open(template_file, 'r') as file:
        template = file.read()
    # replace placeholders with specified values
    modified_script = template.replace('!FLAG<LOG_FILE>', log_file)
    modified_script = modified_script.replace("!FLAG<DATA_FILE>", data_file)
    modified_script = modified_script.replace("!FLAG<HEAD_ATOMS_IDX>", head_text)
    modified_script = modified_script.replace("!FLAG<TAIL_ATOMS_IDX>", tail_text)
    modified_script = modified_script.replace('!FLAG<SPRING_STRENGTH>', str(spring_strength))
    modified_script = modified_script.replace('!FLAG<DATA_OUTPUT>', data_output_file)
    # write the modified script to the output file
    with open(output_file, 'w') as file:
        file.write(modified_script)

def make_lammps_script_from_template(template_file, output_file, flag_dict):
    # read in template
    with open(template_file, 'r') as fin:
        modified_script = fin.read()
    # replace all flags with args
    for key, value in flag_dict.items():
        flag = '!FLAG<' + key + '>'
        modified_script = modified_script.replace(flag, value)
    # check for remaining flags
    if "!FLAG" in modified_script:
        raise RuntimeError('Not all !FLAGS were replaced in the output script.')
    # write out lammps script
    with open(output_file, 'w') as fout:
        fout.write(modified_script)

def make_test_chain(monomer_file, dp_n, charge_file, tact='atactic', tact_control=None, p_tact=None, ter_smiles='[H][3H]', debug=False):
    monomer = Chem.SDMolSupplier(monomer_file, removeHs=False)[0]
    ter = utils.mol_from_smiles(ter_smiles)
    enforce_monomer_charges(monomer, charge_file)

    tac_arr=None
    if tact_control == 'chain_end': tac_arr = make_chain_end_tacticity_arr(dp_n, p_tact)
    elif tact_control == 'site': tac_arr = make_enantiomorphic_site_tacticity_arr(dp_n, p_tact)
    polymer = poly.polymerize_mols(monomer, dp_n, tacticity=tact, tac_array=tac_arr)
    polymer = poly.terminate_mols(polymer, ter)
    if debug:
        return polymer, tac_arr
    return polymer


def generate_single_straight_chains(monomer_file, dp_n, lmp_template,  work_dir, charge_file, n_chains=1, opt=True, tact='atactic', p_tact=None, tact_control=None, ter_smiles='[H][3H]', base_name='chain', spring_strength='10.0', log_name='all.log', quiet=True):
    # clean dirs
    if work_dir[-1] != '/': work_dir = work_dir + '/'
    
    # load monomer
    monomer = Chem.SDMolSupplier(monomer_file, removeHs=False)[0]
    ter = utils.mol_from_smiles(ter_smiles)
    # load monomer charges
    enforce_monomer_charges(monomer, charge_file)
    if ter_smiles != '[H][3H]' or '[3H][H]':
        calc.assign_charges(ter, charge='gasteiger', opt=False)

    # make directory if it does not exist
    if not os.path.isdir(work_dir): 
        if not quiet: print('Made directory: ' + work_dir)
        os.makedirs(work_dir)

    log_file = work_dir + log_name
    f_log = open(log_file, 'w')
    og_dir = os.getcwd()
    # make n chains
    for n in range(n_chains):
        # set up all file paths
        chain_id = base_name + str(n)
        xyz_file = work_dir + chain_id + '.xyz' # sdf mol file output
        data_file = work_dir + chain_id + '.data' # lammps data file output
        script_file_name = chain_id + '.input.lammps' # lammps script file output
        script_file = work_dir + script_file_name # lammps script file output

        # make boolean tacticity array
        tac_arr=None
        if tact_control == 'chain_end': tac_arr = make_chain_end_tacticity_arr(dp_n, p_tact)
        elif tact_control == 'site': tac_arr = make_enantiomorphic_site_tacticity_arr(dp_n, p_tact)

        # build polymer chain
        polymer = poly.polymerize_mols(monomer, n=dp_n, tacticity=tact, tac_array=tac_arr)
        polymer = poly.terminate_mols(polymer, ter)
        
        # assign force feild parameters
        ff = GAFF2_mod()
        result = ff.ff_assign(polymer)
        # transform polymer coordinates to center and align chain on z-axis
        transform_polymer_coords_to_zaxis(polymer)
        # save RDkit object
        write_mol(polymer, xyz_file, ext='xyz')
        # save lammps data file
        lammps.MolToLAMMPSdata(polymer, data_file)

        # get terminal atom indicies
        termC_idx1 = polymer.GetIntProp('terminal_idx1')
        termC_idx2 = polymer.GetIntProp('terminal_idx2')
        # make strings from them
        head_atom_idxs = ''
        for idx in range(termC_idx1, termC_idx1+4): head_atom_idxs = head_atom_idxs + str(idx) + ' '
        tail_atom_idxs = ''
        for idx in range(termC_idx2, termC_idx2+4): tail_atom_idxs = tail_atom_idxs + str(idx) + ' '

        # make flag dict
        flag_dict = {'LOG_FILE':chain_id + '.log.lammps',
                     'DATA_FILE':chain_id + '.data',
                     'HEAD_ATOMS_IDX':head_atom_idxs,
                     'TAIL_ATOMS_IDX':tail_atom_idxs,
                     'SPRING_STRENGTH':spring_strength,
                     'DATA_OUTPUT':chain_id + '.output.data'}

        # create lammps input file
        if lmp_template is not None:
            make_lammps_script_from_template(lmp_template, script_file, flag_dict)

        # run the lammps script
        if opt:
            try:
                os.chdir(work_dir)
                subprocess.run(['mpirun', '-np', '8', 'lmp', '-in', script_file_name], stdout=f_log)
            finally:
                os.chdir(og_dir)
    return

# diameter of chain is approx 10.6 A
# input a list of polymers, nx, ny, [x offset, y offset], [margins]
def build_aligned_box(polymers, nx, ny, xy_offset=[13,13], xyz_margins=[2,2,2], stagger=True):

    if nx*ny != len(polymers):
        raise ValueError('Number of polymers does not match the n-dimensions of box!')
    # reshape polymers array
    polymers_2d = [polymers[i * nx: (i + 1) * nx] for i in range(ny)]

    cell = copy.deepcopy(polymers[0])
    single_chain_coords = cell.GetConformer().GetPositions().T
    z_max = single_chain_coords[2].max()
    z_min = single_chain_coords[2].min()
    z_mag = z_max - z_min

    for y, row in enumerate(polymers_2d):
        for x, mol in enumerate(row):
            if x == 0 and y == 0:
                continue
            z_offset = z_mag*np.random.uniform(-0.5, 0.5) if stagger else 0
            offset_xyz_coords(mol, (xy_offset[0]*x), (xy_offset[1]*y), z_offset)
            cell = poly.combine_mols(cell, mol)
    # resize simulation box
    coords_T = cell.GetConformer().GetPositions().T
    cell.cell.xhi = coords_T[0].max() + xyz_margins[0]
    cell.cell.xlo = coords_T[0].min() - xyz_margins[0]
    cell.cell.yhi = coords_T[1].max() + xyz_margins[1]
    cell.cell.ylo = coords_T[1].min() - xyz_margins[1]
    cell.cell.zhi = z_max + xyz_margins[2]
    cell.cell.zlo = z_min - xyz_margins[2]
    return cell

def generate_aligned_box_and_script(monomer_file, path_to_single_chains, n_chains, n_by_n, template_file, work_dir, script_flag_dict={}, xy_offset=[14, 14], xyz_margins=[2,2,1], output_data_name='input', output_script_name='input.lammps', chain_names='chain', ter_smiles='*C', quiet=True):
    # clean dirs
    if work_dir[-1] != '/': work_dir = work_dir + '/'
    if path_to_single_chains[-1] != '/': path_to_single_chains = path_to_single_chains + '/'

    # load monomer and terminater mols
    monomer = Chem.SDMolSupplier(monomer_file, removeHs=False)[0]
    ter = utils.mol_from_smiles(ter_smiles)
    # load single chains
    listofpolys = [lammps.MolFromLAMMPSdata((path_to_single_chains + chain_names + str(i) + '.data')) for i in range(n_chains)]

    for mol in listofpolys:
        term_idxs = infer_term_idxs(monomer, mol, ter)
        transform_polymer_coords_to_zaxis(mol, term_idxs)

    cell = build_aligned_box(listofpolys, n_by_n[0], n_by_n[1], xy_offset=xy_offset, xyz_margins=xyz_margins)
    lammps.MolToLAMMPSdata(cell, work_dir + output_data_name + '.data')
    write_mol(cell, work_dir + output_data_name + '.xyz', ext='xyz')

    if not quiet: print('Terminal atom indicies:', infer_term_idxs(monomer, listofpolys[0], ter, n=n_chains))
    if not quiet: print('Cell dimensions:\n', '\tx', cell.cell.xhi, cell.cell.xlo, '\n', '\ty', cell.cell.yhi, cell.cell.ylo, '\n', '\tz', cell.cell.zhi, cell.cell.zlo)

    script_flag_dict_default = {
        'LOG':'md.log',
        'MAX_PRESS':'5000.0',
        'RESTART':'md.rst',
        'DUMP_MIN':'dump.min.lammpstrj',
        'DUMP_MD':'dump.md.lammpstrj',
        'DUMP_MD_UNWRAPPED':'dump.md_unwrapped.lammpstrj',
        'DUMP_LAST':'dump.final_state.lammpstrj',
        'DATA_OUTPUT':'final.data',
        'DATA_INPUT':output_data_name + '.data'
    }
    flag_dict = {**script_flag_dict_default, **script_flag_dict}
    if template_file is not None:
        make_lammps_script_from_template(template_file, work_dir + output_script_name, flag_dict)
    return

def generate_single_rw_chains(monomer_file, dp_n, work_dir, charge_file, n_chains=1, tact='atactic', p_tact=None, tact_control=None, ter_smiles='[H][3H]', base_name='chain', log_name='all.log', quiet=True, forcefield=GAFF2_mod(), opt='lammps', mpi=8, omp=8):
    # clean dirs
    if work_dir[-1] != '/': work_dir = work_dir + '/'
    
    # load monomer
    monomer = Chem.SDMolSupplier(monomer_file, removeHs=False)[0]
    ter = utils.mol_from_smiles(ter_smiles)
    enforce_monomer_charges(monomer, charge_file)
    if ter_smiles != '[H][3H]' or '[3H][H]':
        calc.assign_charges(ter, charge='gasteiger', opt=False)

    # make directory if it does not exist
    if not os.path.isdir(work_dir): 
        if not quiet: print('Made directory: ' + work_dir)
        os.makedirs(work_dir)
    
    # check if dp_n is list
    if isinstance(dp_n, list):
        if len(dp_n) != n_chains:
            raise ValueError('Length of DPn list does not equal n_chains')
    else:
        dp_n = [dp_n for i in range(n_chains)]

    # make n chains
    for n, dp in zip(range(n_chains), dp_n):
        # set up all file paths
        chain_id = base_name + str(n)
        xyz_file = work_dir + chain_id + '.xyz' # sdf mol file output
        data_file = work_dir + chain_id + '.data' # lammps data file output

        # make boolean tacticity array
        tac_arr=None
        if tact_control == 'chain_end': tac_arr = make_chain_end_tacticity_arr(dp, p_tact)
        elif tact_control == 'site': tac_arr = make_enantiomorphic_site_tacticity_arr(dp, p_tact)

        # build polymer chain
        polymer = poly.polymerize_rw(monomer, dp, tacticity=tact, tac_array=tac_arr, opt=opt, ff=forcefield, work_dir=work_dir, mpi=mpi, omp=omp)
        polymer = poly.terminate_rw(polymer, ter, opt=opt, mpi=mpi, omp=omp, work_dir=work_dir)
        
        # assign force feild parameters
        ff = forcefield
        result = ff.ff_assign(polymer)
        # save RDkit object
        write_mol(polymer, xyz_file, ext='xyz')
        # save lammps data file
        lammps.MolToLAMMPSdata(polymer, data_file)
    return

def generate_amorphous_cell_and_script(path_to_single_chains, n_chains, template_file, work_dir, density=0.03, retry=10, retry_step=100, threshold=2.0, dec_rate=0.8, check_structure=True, script_flag_dict={}, output_data_name='input', output_script_name='input.lammps', chain_names='chain', ter_smiles='[H][3H]', quiet=True):
    # clean dirs
    if work_dir[-1] != '/': work_dir = work_dir + '/'
    if path_to_single_chains[-1] != '/': path_to_single_chains = path_to_single_chains + '/'

    # load single chains
    listofpolys = [lammps.MolFromLAMMPSdata((path_to_single_chains + chain_names + str(i) + '.data')) for i in range(n_chains)]


    n_array = np.ones(n_chains, dtype=int)
    cell = poly.amorphous_mixture_cell(listofpolys, n_array, density=density, retry=retry, retry_step=retry_step, threshold=threshold, dec_rate=dec_rate, check_structure=check_structure)
    lammps.MolToLAMMPSdata(cell, work_dir + output_data_name + '.data')
    write_mol(cell, work_dir + output_data_name + '.xyz', ext='xyz')

    if not quiet: print('Cell dimensions:\n', '\tx', cell.cell.xhi, cell.cell.xlo, '\n', '\ty', cell.cell.yhi, cell.cell.ylo, '\n', '\tz', cell.cell.zhi, cell.cell.zlo)

    script_flag_dict_default = {
        'LOG':'md.log',
        'MAX_PRESS':'5000.0',
        'RESTART':'md.rst',
        'DUMP_MIN':'dump.min.lammpstrj',
        'DUMP_MD':'dump.md.lammpstrj',
        'DUMP_MD_UNWRAPPED':'dump.md_unwrapped.lammpstrj',
        'DUMP_LAST':'dump.final_state.lammpstrj',
        'DATA_OUTPUT':'final.data',
        'DATA_INPUT':output_data_name + '.data'
    }
    flag_dict = {**script_flag_dict_default, **script_flag_dict}
    if template_file is not None:
        make_lammps_script_from_template(template_file, work_dir + output_script_name, flag_dict)
    return

def make_ac_quick(monomer_file, charge_file, dp_n, n_chains, work_dir, template_file, tact='atactic', p_tact=None, tact_control=None, density=0.03, retry=10, retry_step=100, threshold=2.0, dec_rate=0.8, script_flag_dict={}, output_data_name='input', output_script_name='input.lammps', chain_name='chain', ter_smiles='[H][3H]', log_name='all.log', quiet=True, mpi=8, forcefield=GAFF2_mod()):
    '''
    makes a amorphous cell and lammps input script from a single rw chain
    '''
    # clean dirs
    if work_dir[-1] != '/': work_dir = work_dir + '/'

    generate_single_rw_chains(monomer_file=monomer_file, dp_n=dp_n, work_dir=work_dir, charge_file=charge_file, n_chains=1, tact=tact, p_tact=p_tact, tact_control=tact_control, base_name=chain_name, ter_smiles=ter_smiles, log_name=log_name, quiet=quiet, mpi=mpi, forcefield=forcefield)


    polymer = lammps.MolFromLAMMPSdata((work_dir + chain_name + '0.data'))
    cell = poly.amorphous_cell(polymer, n=n_chains, density=density, retry=retry, retry_step=retry_step, threshold=threshold, dec_rate=dec_rate)

    lammps.MolToLAMMPSdata(cell, work_dir + output_data_name + '.data')
    write_mol(cell, work_dir + output_data_name + '.xyz', ext='xyz')

    if not quiet: print('Cell dimensions:\n', '\tx', cell.cell.xhi, cell.cell.xlo, '\n', '\ty', cell.cell.yhi, cell.cell.ylo, '\n', '\tz', cell.cell.zhi, cell.cell.zlo)

    script_flag_dict_default = {
        'LOG':'md.log',
        'MAX_PRESS':'5000.0',
        'RESTART':'md.rst',
        'DUMP_MIN':'dump.min.lammpstrj',
        'DUMP_MD':'dump.md.lammpstrj',
        'DUMP_MD_UNWRAPPED':'dump.md_unwrapped.lammpstrj',
        'DUMP_LAST':'dump.final_state.lammpstrj',
        'DATA_OUTPUT':'final.data',
        'DATA_INPUT':output_data_name + '.data'
    }
    flag_dict = {**script_flag_dict_default, **script_flag_dict}
    if template_file is not None:
        make_lammps_script_from_template(template_file, work_dir + output_script_name, flag_dict)
    return


def geom_charge_optimization_wrapper(mol, save_path, save_name='mol', charge='gasteiger', confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='charge', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br':'6-31G(d)', 'I': 'lanl2dz'}, 
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    charge_method='HF', charge_basis='6-31G(d)', charge_basis_gen={'Br':'6-31G(d)', 'I': 'lanl2dz'},
    total_charge=None, total_multiplicity=None, **kwargs):
    # pass arguements into assign charges for optimization + charge calculation
    flag = qm.assign_charges(mol, charge=charge, confId=confId, opt=opt, work_dir=work_dir, tmp_dir=tmp_dir, log_name=log_name, qm_solver=qm_solver,
            opt_method=opt_method, opt_basis=opt_basis, geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
            charge_method=charge_method, charge_basis=charge_basis, charge_basis_gen=charge_basis_gen,
            total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)
    # break early if flag is flase
    if not flag:
        return flag
    
    # save mol geom and charges
    if save_path[-1] != '/':
        save_path = save_path + '/'

    mol_save_path = save_path + save_name
    write_mol(mol, mol_save_path + '.sdf', confId=confId)
    write_mol(mol, mol_save_path + '.xyz', ext='xyz', confId=confId)

    charges = []
    charge_save_path = save_path + 'charges.json'
    for atom in mol.GetAtoms():
        charges.append(atom.GetDoubleProp('AtomicCharge'))
    with open(charge_save_path, 'w') as json_file:
        json.dump(charges, json_file)
    
    return True
    