import re
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm  # Progress bar for long operations

# Function to read data from the file
def read_data(file):
    '''
    Extracts particle data and metadata from the file.
    The file is expected to have the following structure:
      * The 2nd line gives the Time step
      * The actual data starts from the 10th line
      * Data format: (Particle identifier, Molecule identifier, particle type, position(x, y, z), 
                     Periodic image)
    '''
    with open(file, 'r') as f:
        next(f)  # Skip the first line
        t_step = float(re.findall(r'[-+]?\d*\.*\d+', f.readline())[0])  # Timestep
        next(f)
        Natoms = int(re.findall(r'[-+]?\d+', f.readline())[0])  # Number of atoms
        next(f)
        X_box = list(map(float, f.readline().split()))  # X box dimensions
        Y_box = list(map(float, f.readline().split()))  # Y box dimensions
        Z_box = list(map(float, f.readline().split()))  # Z box dimensions
        next(f)  # Skip line before the data
        
        # Read particle data starting from the 10th line
        data = pd.read_csv(file, sep=' ', skiprows=9, names=['Pid', 'Mid', 'PType', 'Px', 'Py', 'Pz', 'Pix', 'Piy', 'Piz', 
            'Vx', 'Vy', 'Vz'])
        
    return t_step, Natoms, X_box, Y_box, Z_box, data

# Function to extract the number from the filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename.name)
    return int(match.group(0)) if match else 0

# Directory for the trajectory files
directory = Path(os.path.join(os.getcwd(), 'trajactory_files'))
files = [f for f in directory.iterdir() if f.is_file()]

# Sort files based on the extracted number
sorted_files = sorted(files, key=extract_number)
print("\n\tSTATISTICS:")
print(f'\t\t{len(sorted_files)} files will be processed')

# Function to get center of mass for molecules
def get_mol(file, ids=[]):
    t_step, Natoms, X_box, Y_box, Z_box, atoms = read_data(file)
    
    # If 'ids' is 'all', process all molecules
    if len(ids) == 0:
        N_mol = atoms['Mid'].nunique()
        par = [i + 1 for i in range(N_mol)]
    else:
        par = ids
        N_mol = len(par)
    
    # Masses
    m1 = 1.673  # Mass of atom type 2 (PType = 89/91)
    m2 = 19.94  # Mass of atom type 1 (PType = 88/90)
    
    # Assign mass based on particle type
    atoms['mass'] = np.where(atoms['PType'].isin([88, 90]), m2, m1)
    
    mid = atoms['Mid'].to_numpy()
    mass = atoms['mass'].to_numpy()
    x_com = atoms['Px'].to_numpy() + atoms['Pix'].to_numpy() * X_box[1]
    y_com = atoms['Py'].to_numpy() + atoms['Piy'].to_numpy() * Y_box[1]
    z_com = atoms['Pz'].to_numpy() + atoms['Piz'].to_numpy() * Z_box[1]
    
    comx = np.zeros(N_mol)
    comy = np.zeros(N_mol)
    comz = np.zeros(N_mol)
    
    # Compute COM for each molecule
    for i, j in enumerate(par):
        mask = (mid == j)
        total_mass = np.sum(mass[mask])
        comx[i] = np.sum(mass[mask] * x_com[mask]) / total_mass
        comy[i] = np.sum(mass[mask] * y_com[mask]) / total_mass
        comz[i] = np.sum(mass[mask] * z_com[mask]) / total_mass
    
    mol = np.vstack((comx, comy, comz)).T
    return mol, N_mol, t_step

def get_ids(atoms): # a custom function used for getting ids of molecules to be tracked
    #return atoms.loc[(atoms['Py'] < 250) & (atoms['Py'] > 195) & (atoms['Mid'] > 4000)]['Mid'].unique()
    return [i for i in range(4001, 8000)]

# Initialize variables
start = time.time()
file0 = os.path.join('trajactory_files', sorted_files[0].name)
t_step, Natoms, X_box, Y_box, Z_box, atoms = read_data(file0)
ids = get_ids(atoms)
mol, N_mol, t = get_mol(file0)
mol0, N_mol0, t0 = get_mol(file0, ids)

print(f'\t\t{N_mol} molecules found resulting in a total of {Natoms} atoms in the system')
print(f'\t\tOut of {N_mol} molecules only {len(ids)} will be tracked')
del mol
del N_mol 
del t

msd = np.zeros(len(sorted_files))
del_t = np.zeros(len(sorted_files))

# Loop over files and calculate MSD
for t, f in tqdm(enumerate(sorted_files[1:], start=1), total=len(sorted_files) - 1, desc="Processing files"):
    file = os.path.join('trajactory_files', f.name)
    mol_t, N_mol, t_step = get_mol(file, ids)
    
    if N_mol != N_mol0:
        print(f"Molecules lost: {N_mol - N_mol0}")
        break
    
    # Calculate squared displacement in y direction and update MSD
    del_r_sq = (mol_t[:, 1] - mol0[:, 1]) ** 2
    msd[t] = np.sum(del_r_sq) / N_mol0  # Average MSD per molecule
    del_t[t] = t_step

end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# Plot MSD vs Time
#time_steps = [t for t in range(1, len(sorted_files))]
diff = np.diff(msd)/(6*np.diff(del_t))
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel("msd")
ax[1].set_ylabel("diff coeff")
ax[0].set_xlabel("time")
ax[1].set_xlabel("time")
ax[0].plot(del_t, msd)
ax[1].plot(del_t[:-1], diff)
plt.show()
plt.savefig('diff_y_mid.png', dpi=300)