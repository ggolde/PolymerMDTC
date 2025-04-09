import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import MDAnalysis as mda
import os

from lammps_logfile import File as read_lmp_log
from ..sim.lammps import MolFromLAMMPSdata
import matplotlib.pyplot as plt
from scipy.stats import linregress

# functions
## preprocessing
def calc_dpn(n_polymer, n_mono):
    dpn = int((n_polymer-2)/(n_mono-2))
    return dpn

def calc_monomer_bb_idxs(monomer):
    bb_mask = np.zeros(shape=monomer.GetNumAtoms(), dtype=bool)
    H3_idxs = []

    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == 'H' and atom.GetMass() > 3:
            H3_idxs.append(atom.GetIdx())
            if len(atom.GetNeighbors()) != 1:
                raise ValueError('Tritium atom bonded to more than one atom.')
            bb_mask[atom.GetNeighbors()[0].GetIdx()] = True
    return bb_mask, H3_idxs

def calc_chain_bb_idxs(dpn, bb_mask, H3_idxs, offset=0):
    bb_head = np.delete(bb_mask, H3_idxs[1]).tolist()
    bb_body = np.delete(bb_mask, H3_idxs).tolist()
    bb_tail = np.delete(bb_mask, H3_idxs[0]).tolist()
    bb_chain = bb_head + (dpn-2)*bb_body + bb_tail
    bb_idxs = [i+offset for i, mask in enumerate(bb_chain) if mask]
    return bb_idxs

def calc_cell_bb_idxs(cell, monomer):
    bb_mask, H3_idxs = calc_monomer_bb_idxs(monomer)
    n_offset = 0
    cell_bbs = []
    chain_bbs = []
    for chain in Chem.GetMolFrags(cell, asMols=True):
        dpn = calc_dpn(chain.GetNumAtoms(), monomer.GetNumAtoms())
        cell_bbs.append(calc_chain_bb_idxs(dpn, bb_mask, H3_idxs, offset=n_offset))
        chain_bbs.append(calc_chain_bb_idxs(dpn, bb_mask, H3_idxs, offset=0))
        n_offset += chain.GetNumAtoms()
    return cell_bbs, chain_bbs

def get_bb_trajectory(u, cell_bb_idxs, start_frame, stop_frame):
    frame_range = np.arange(start_frame, stop_frame+1, dtype=int)
    chain_bb_trajs = []
    cell_bb_traj = []

    for i, frame in enumerate(frame_range):
        u.trajectory[frame]
        cell_pos = np.empty(shape=(0,3))
        chain_pos = []

        for chain_bb_idxs in cell_bb_idxs:
            n_bb_atoms = len(chain_bb_idxs)
            pos = u.atoms.positions[chain_bb_idxs]
            chain_pos.append(pos)
            cell_pos = np.concatenate([cell_pos, pos], axis=0)
        
        chain_bb_trajs.append(chain_pos)
        cell_bb_traj.append(cell_pos)

    cell_bb_traj = np.array(cell_bb_traj)
    return cell_bb_traj, chain_bb_trajs

def split_chains(cell):
    chains = [chain for chain in Chem.GetMolFrags(cell, asMols=True)]
    return chains

def calc_bb_bond_vectors(chain_bb_pos):
    bond_vectors = np.empty(shape=(0,3))
    for bb_pos in chain_bb_pos:
        chain_vec = np.diff(bb_pos, axis=0)
        bond_vectors = np.concatenate([bond_vectors, chain_vec], axis=0)
    return bond_vectors

# metrics
def get_diads(stereo):
    tact = {'mm':0, 'rc':0}
    for i in range(1, len(stereo)):
        if stereo[i-1] == stereo[i]: tact['mm'] += 1
        else: tact['rc'] += 1
    if tact['mm']+tact['rc'] == 0: p_mm = 0
    else:
        p_mm = tact['mm'] / (tact['mm']+tact['rc'])
    return tact, p_mm

def get_stereo(cell, chain_bb_idxs):
    p_mm = []
    chains = split_chains(cell)
    for chain, bb_idxs in zip(chains, chain_bb_idxs):
        stereo = []
        Chem.rdmolops.AssignStereochemistryFrom3D(chain)
        for atom in chain.GetAtoms():
            if '_CIPCode' in atom.GetPropsAsDict().keys() and atom.GetIdx() in bb_idxs:
                stereo.append(atom.GetProp('_CIPCode'))
        tact, isoTact = get_diads(stereo)
        p_mm.append(isoTact)
    p_mm = np.array(p_mm)
    return p_mm

def compute_relative_shape_anisotropy(positions):
    """
    Compute the relative shape anisotropy per polymer chain.

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
    top = lambda1**4 + lambda2**4 + lambda3**4
    bottom = (lambda1**2 + lambda2**2 + lambda3**2)**2
    anisotropy = (3/2)*(top/bottom)-(1/2)

    # Compute radius of gyration
    Rg = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)

    return anisotropy, Rg

def compute_herman_orientation_factor(bond_vectors, ref_vec='avg'):
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
    # Normalize bond vectors to unit vectors
    bond_vectors_norm = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]

    # Calculate the reference vector
    avg_bond_vector = np.mean(bond_vectors_norm, axis=0)
    avg_bond_vector /= np.linalg.norm(avg_bond_vector)

    if ref_vec=='avg':
        ref_vec = avg_bond_vector 
    else:
        ref_vec = np.array(ref_vec, dtype=float)
        ref_vec /= np.linalg.norm(ref_vec)
    
    # Ensure all bond vectors point in the same general direction as the average bond vector
    dot_products = np.dot(bond_vectors_norm, ref_vec)
    bond_vectors_norm[dot_products < 0] *= -1

    # Compute the cosine squared of the angle between each bond vector and the average bond vector
    cos_squared = np.dot(bond_vectors_norm, ref_vec) ** 2

    # Compute the Herman orientation factor
    herman_factor = 1.5 * np.mean(cos_squared) - 0.5

    return herman_factor

def calc_chain_alignment_metrics(chain_bb_traj):
    herman = {'x':[], 'y':[], 'z':[], 'avg':[]}
    shape_aniso = []
    rg = []
    for frame in chain_bb_traj:
        # shape anisotropy
        shape_aniso_frame = []
        rg_frame = []
        for chain_pos in frame:
            pcf_sa, pcf_rg = compute_relative_shape_anisotropy(chain_pos)
            shape_aniso_frame.append(pcf_sa)
            rg_frame.append(pcf_rg)

        shape_aniso_frame = np.array(shape_aniso_frame)
        rg_frame = np.array(rg_frame)

        shape_aniso.append(shape_aniso_frame)
        rg.append(rg_frame)
        # herman orientation factor
        bond_vectors = calc_bb_bond_vectors(frame)
        herman['x'].append(compute_herman_orientation_factor(bond_vectors, ref_vec=[1,0,0]))
        herman['y'].append(compute_herman_orientation_factor(bond_vectors, ref_vec=[0,1,0]))
        herman['z'].append(compute_herman_orientation_factor(bond_vectors, ref_vec=[0,0,1]))
        herman['avg'].append(compute_herman_orientation_factor(bond_vectors, ref_vec='avg'))

    shape_aniso = np.array(shape_aniso)
    rg = np.array(rg)
    for key, val in herman.items():
        herman[key] = np.array(val)
    
    return herman, shape_aniso, rg

# wrapper
def calc_MW_metrics(cell, monomer):
    mws = []
    dpns = []
    for chain in Chem.GetMolFrags(cell, asMols=True):
        mws.append(Chem.Descriptors.ExactMolWt(chain))
        dpns.append(calc_dpn(chain.GetNumAtoms(), monomer.GetNumAtoms()))
    mws = np.array(mws)
    dpns = np.array(dpns)
    Mn = mws.mean()
    Mw = np.power(mws, 2).sum() / mws.sum()
    PDI = Mw/Mn
    return mws, dpns, PDI

def compute_metrics(cell, chain_bb_idxs, chain_bb_traj, monomer):
    # per frame / per chain
    pc_pmm = get_stereo(cell, chain_bb_idxs)
    pf_herman, pfc_shape_aniso, pfc_rg = calc_chain_alignment_metrics(chain_bb_traj)
    pc_mw, pc_dpn, PDI = calc_MW_metrics(cell, monomer)
    # reduction
    df = pd.DataFrame(index=['mean', 'std'])
    df['Mn'] = [pc_mw.mean(), pc_mw.std()]
    df['DPn'] = [pc_dpn.mean(), pc_dpn.std()]
    df['PDI'] = [PDI, 0]
    df['%mm'] = [pc_pmm.mean(), pc_pmm.std()]

    for key, val in pf_herman.items():
        col = 'herman_' + key
        df[col] = [val.mean(), val.std()]

    pc_shape_aniso = pfc_shape_aniso.mean(axis=0)
    df['shape_aniso'] = [pc_shape_aniso.mean(), pc_shape_aniso.std()]
    pc_rg = pfc_rg.mean(axis=0)
    df['Rg'] = [pc_rg.mean(), pc_rg.std()]
    return df

def metrics_wrapper(u, dataf, monof, start_frame, end_frame, save_out=None):
    monomer = Chem.SDMolSupplier(monof, removeHs=False)[0]
    cell = MolFromLAMMPSdata(dataf)
    bb_idxs, chain_bb_idxs = calc_cell_bb_idxs(cell, monomer)
    cell_bb_traj, chain_bb_traj = get_bb_trajectory(u, bb_idxs, start_frame, end_frame)
    metrics = compute_metrics(cell, chain_bb_idxs, chain_bb_traj, monomer)
    metrics = metrics.T
    if save_out is not None:
        metrics.to_csv(save_out)
    return metrics

# classes
class Sim:
    def __init__(self, logf):
        self.log = read_lmp_log(logf)
        self.allvars = self.log.keywords
        data = pd.DataFrame(self.log.data_dict)
        min_step = data['Step'].min()

        ts0_idxs = data[data['Step'] == min_step].index
        self.data = data.loc[ts0_idxs[-1]:].copy()
        for i, plog in enumerate(self.log.partial_logs):
            if plog['Step'][0] == self.data['Step'].iloc[0] and plog['TotEng'][0] == self.data['TotEng'].iloc[0]:
                self.partial_logs = self.log.partial_logs[i:]
                break
        self.assign_run_nums()

    def assign_run_nums(self):
        run_nums = []
        for i, plog in enumerate(self.partial_logs):
            for j in plog['Step']:
                run_nums.append(i+1)
        self.data.loc[:,'Run'] = run_nums
        self.data.set_index('Run', inplace=True)
        self.allruns = list(range(1,run_nums[-1]+1))
        self.lastrun = self.allruns[-1]
        return
    
    def get_data(self, vars, runs, includeStep=True):
        if not isinstance(vars, list): vars = [vars]
        if includeStep:
            vars = ['Step'] + vars
        return self.data.loc[runs, vars]
            
    def running_mean(self, vars, runs, N, includeStep=True):
        if not isinstance(vars, list): vars = [vars]
        df = self.get_data(vars, runs, includeStep=includeStep)
        rolling_means = {}
        if includeStep: rolling_means['Step'] = df['Step'].to_numpy()
        for var in vars:
            data = df[var].to_numpy()
            if N == 1:
                return data
            else:
                retArray = np.zeros(data.size)*np.nan
                padL = int(N/2)
                padR = N-padL-1
                retArray[padL:-padR] = np.convolve(data, np.ones((N,))/N, mode='valid')
                retArray[:padL] = retArray[padL]
                retArray[-padR:] = retArray[-(padR+1)]
            rolling_means[var] = retArray
        return pd.DataFrame(rolling_means)
    
    def check_stability(self, vars, runs, N, window_size, window_step, thresholds, n_stab_win=3, time_conv=1000):
        if not isinstance(vars, list): vars = [vars]
        rm = self.running_mean(vars, runs, N)
        x = self.data.loc[runs, 'Time'].to_numpy()/time_conv
        steps = self.data.loc[runs, 'Step'].to_numpy()
        stab_dict = {}
        if not isinstance(thresholds, list): thresholds = [thresholds for i in vars]

        for var, threshold in zip(vars, thresholds):
            stab_dict[var] = {}
            rows = []
            y = rm[var].to_numpy()
            for start in range(0, len(y) - window_size + 1, window_step):
                stop = start+window_size
                interval = (steps[start], steps[stop])
                x_window = x[start:stop]
                y_window = y[start:stop]
                reg = linregress(x_window, y_window)
                stable = np.abs(reg.slope) < threshold
                rows.append([interval, stable, reg.slope, reg.rvalue**2, reg.intercept, y_window.mean()])
            stab_data = pd.DataFrame(rows, columns=['interval', 'stable', 'slope', 'r2', 'intercept', 'meanVal'])
            
            stabilized = False
            stableSum = 0
            stableWin = 0
            for i, j in enumerate(reversed(stab_data.index)):
                row = stab_data.iloc[j,:]
                if i==0 and row.stable:
                    stab_interval = list(row.interval)
                    stableSum += row.meanVal
                    stableWin += 1
                elif row.stable:
                    stab_interval[1] = row.interval[1]
                    stableSum += row.meanVal
                    stableWin += 1
                    if i>n_stab_win:
                        stabilized = True
                else:
                    break
            if stabilized: stableVal = stableSum/stableWin
            else: 
                stableVal = stab_data['meanVal'].iloc[-3:].mean()
                stab_interval = None
            
            stab_dict[var]['val'] = stableVal
            stab_dict[var]['stable'] = stabilized
            stab_dict[var]['interval'] = stab_interval
            stab_dict[var]['data'] = stab_data

        return stab_dict
    
    def plot_var(self, var, runs, rm=False, stab=False, **kwargs):
        df = self.get_data(var, runs)
        plt.plot(df['Step'], df[var])
        title=var
        if rm:
            N = kwargs.get('N', 10)
            rm = self.running_mean(var, runs, N=N)
            plt.plot(df['Step'], rm[var], label='Running mean')
        if stab:
            N = kwargs.get('N', 10)
            window_size = kwargs.get('window_size', 100)
            window_step = kwargs.get('window_step', 50)
            thresholds = kwargs.get('thresholds', 1e-6)
            time_conv = kwargs.get('time_conv', 1000)
            stab_dict = self.check_stability(var, runs, N, window_size, window_step, thresholds, time_conv=time_conv)
            title = title + ' ' + str(stab_dict[var]['val'])
            for i, row in stab_dict[var]['data'].iterrows():
                x = np.linspace(row.interval[0], row.interval[1], num=10)
                y = (x/time_conv)*row.slope + row.intercept
                if row.stable: color='forestgreen'
                else: color = 'firebrick'
                plt.plot(x,y, c=color, linestyle='--')
                
        plt.title(title)

    def get_end_val(self, vars, n_last=10):
        if not isinstance(vars, list): vars = [vars]
        sim_metrics = {} 
        sim_metrics['mean'] = self.data.iloc[-n_last:].loc[:,vars].mean(axis=0)
        sim_metrics['std'] = self.data.iloc[-n_last:].loc[:,vars].std(axis=0)
        return pd.DataFrame(sim_metrics)
    
class ProductionSim(Sim):
    def __init__(self, logf, monof, dataf, trajf):
        super().__init__(logf)
        self.dataf = dataf
        self.trajf = trajf
        self.monof = monof
        self.mda_obj = mda.Universe(dataf, trajf, format='LAMMPSDUMP', lengthunit='A', timeunit='fs', dt=1.0)

    def analyze(self, vars=['TotEng', 'Density'], n_last=30):
        sim_metrics = self.get_end_val(vars, n_last=n_last)

        frame_range = [self.mda_obj.trajectory.n_frames-n_last-1, self.mda_obj.trajectory.n_frames-1]
        chain_metrics = metrics_wrapper(self.mda_obj, self.dataf, self.monof, start_frame=frame_range[0], end_frame=frame_range[1])

        self.metrics = pd.concat([sim_metrics.T, chain_metrics.T], axis=1)
        return self.metrics

class TCSim(Sim):
    def __init__(self, logf):
        super().__init__(logf)
    
    def analyze(self, vars=['kappa', 'kappa_inst'], n_last=[1, 30]):
        if isinstance(n_last, list):
            datas = []
            for var, n in zip(vars, n_last):
                datas.append(self.get_end_val(var, n_last=n).T)
            self.metrics = pd.concat(datas, axis=1)
        else:
            self.metrics = self.get_end_val(vars, n_last=n_last)
        return self.metrics

class GKSim(Sim):
    def __init__(self, logf):
        super().__init__(logf)
    
    def analyze(self, vars=['kappa', 'kappa_11', 'kappa_22', 'kappa_33'], n_last=30): # , 
        self.metrics = self.get_end_val(vars, n_last=n_last).T
        self.metrics.columns = ['kappa', 'kappa_x', 'kappa_y', 'kappa_z']
        return self.metrics

class xyzTCwrapper:
    def __init__(self, logfs, vars = ['kappa', 'kappa_inst'], n_last = [1, 50], dims = ['x', 'y', 'z'], ):
        sims = {}
        data_list = []
        for logf, dim in zip(logfs, dims):
            sims[dim] = TCSim(logf)
            data = sims[dim].analyze(vars, n_last)
            data.columns = [col + '_' + dim for col in data.columns]
            data_list.append(data)

        metrics = pd.concat(data_list, axis=1)
        for var in vars:
            col_names = [var + '_' + dim for dim in dims]
            metrics[var] = [metrics.loc['mean',col_names].mean(), metrics.loc['mean',col_names].std()]
        self.sims = sims
        self.metrics = metrics

    def analyze(self):
        return self.metrics
    
class Sample:
    def __init__(self, name, smp_dir, mono_dir, prod_logf='production.log', dataf='input.data', trajf='dump.md.lammpstrj', dim_tc_meths=['tc_ehex', 'tc_lav', 'tc_mp'], nd_tc_meths=['tc_gk']):
        if not os.path.isdir(smp_dir):
            raise RuntimeError('Sample directory does not exsits')
        self.name = name
        self.dir = smp_dir
        # grab production data
        prod_dir = os.path.join(smp_dir, prod_logf)
        data_dir = os.path.join(smp_dir, dataf)
        traj_dir = os.path.join(smp_dir, trajf)
        self.metrics = ProductionSim(prod_dir, mono_dir, data_dir, traj_dir).analyze().T
        # generate tc_method log file names
        tc_logs = {}
        for tc_meth in dim_tc_meths:
            tc_logs[tc_meth] = [tc_meth.split('_')[0] + '_' + dim + '_' + tc_meth.split('_')[1] + '.log' for dim in ['x', 'y', 'z']]
        # grab tc data
        self.tc = {}
        for tc_meth in dim_tc_meths:
            sim_dir = os.path.join(smp_dir, tc_meth)
            if os.path.isdir(sim_dir):
                logfs = [os.path.join(sim_dir, tc_log) for tc_log in tc_logs[tc_meth]]
                if not os.path.exists(logfs[0]): 
                    break
                sim = xyzTCwrapper(logfs)
                key = tc_meth.split('_')[1]
                self.tc[key] = sim.analyze()
        for tc_meth in nd_tc_meths:
            sim_dir = os.path.join(smp_dir, tc_meth)
            if os.path.isdir(sim_dir):
                logf = os.path.join(sim_dir, tc_meth+'.log')
                if not os.path.exists(logf): 
                    break
                sim = GKSim(logf)
                key = tc_meth.split('_')[1]
                self.tc[key] = sim.analyze()
        kappas = []
        for key, val in self.tc.items():
            data = val.loc[:,['kappa', 'kappa_x', 'kappa_y', 'kappa_z']].copy()
            data.columns = [key + '_' + col for col in data.columns]
            kappas.append(data)
        self.kappas = pd.concat(kappas, axis=1).T
        self.metrics = pd.concat([self.metrics, self.kappas])
    
    def save_data(self, save_dir):
        save_path = os.path.join(save_dir, self.name+'.csv')
        self.metrics.to_csv(save_path, index=True)
        
class MultiSample:
    def __init__(self, name, dir, mono_dir, save_dir, smp_names = ['smp1', 'smp2', 'smp3']):
        self.dir = dir
        self.name = name
        smps = []
        for smp in smp_names:
            smp_dir = os.path.join(dir, smp)
            smp_name = name + '_' + smp
            smps.append(Sample(smp_name, smp_dir, mono_dir))
        
        smp_data = []
        for smp in smps:
            smp.save_data(save_dir)
            data = smp.metrics.loc[:,['mean']].copy()
            data.columns = [smp.name]
            smp_data.append(data)

        self.metrics = pd.concat(smp_data, axis=1)
        mean = self.metrics.mean(axis=1)
        std = self.metrics.std(axis=1)
        self.metrics[name + '_mean'] = mean
        self.metrics[name + '_std'] = std
        
        self.save_data(save_dir)

    def save_data(self, save_dir):
        save_path = os.path.join(save_dir, self.name+'.csv')
        self.metrics.to_csv(save_path, index=True)