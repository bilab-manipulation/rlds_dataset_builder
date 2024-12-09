import numpy as np
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import h5py
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def create_graph(file_dir, ee_relative, dest_dir):
    # Plot the ee_relative data
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    # Plot positions (x, y, z)
    axs[0, 0].plot(ee_relative[:, 0], label="X")
    axs[0, 0].set_title("X Position")
    axs[0, 0].set_xlabel("Timestep")
    axs[0, 0].set_ylabel("Position")
    
    axs[1, 0].plot(ee_relative[:, 1], label="Y")
    axs[1, 0].set_title("Y Position")
    axs[1, 0].set_xlabel("Timestep")
    axs[1, 0].set_ylabel("Position")
    
    axs[2, 0].plot(ee_relative[:, 2], label="Z")
    axs[2, 0].set_title("Z Position")
    axs[2, 0].set_xlabel("Timestep")
    axs[2, 0].set_ylabel("Position")

    # Plot orientations (roll, pitch, yaw in degrees)
    axs[0, 1].plot(np.degrees(ee_relative[:, 3]), label="rx")
    axs[0, 1].set_title("rx")
    axs[0, 1].set_xlabel("Timestep")
    axs[0, 1].set_ylabel("Delta Angle(degree)")
    
    axs[1, 1].plot(np.degrees(ee_relative[:, 4]), label="ry")
    axs[1, 1].set_title("ry")
    axs[1, 1].set_xlabel("Timestep")
    axs[1, 1].set_ylabel("Delta Angle(degree)")
    
    axs[2, 1].plot(np.degrees(ee_relative[:, 5]), label="rz")
    axs[2, 1].set_title("rz")
    axs[2, 1].set_xlabel("Timestep")
    axs[2, 1].set_ylabel("Delta Angle(degree)")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    # base_dir = Path(__file__).parent  # This will get the directory where your Python file is located
    graph_dir = Path(dest_dir) / 'graph'  # Create the 'graph' folder path

    # Create the 'graph' directory if it doesn't exist
    if not graph_dir.exists():
        os.makedirs(graph_dir)
        print(f"Created 'graph' directory at {graph_dir}")

    # Save the figure in the 'graph' folder
    plot_filename = Path(file_dir).stem + '_ee_relative_plot.png'
    plot_path = graph_dir / plot_filename
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

# Set up argument parser
parser = argparse.ArgumentParser(description='Process HDF5 files and generate dataset.')
parser.add_argument('--h5_pth', type=str, required=True, help='Path to the directory containing HDF5 files.')
parser.add_argument('--dest_dir', type=str, required=True, help='Destination directory for the output dataset.')
parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt guideline text file.')

args = parser.parse_args()

h5_pth = Path(args.h5_pth)

# Get list of all files and folders in the directory
file_list = list(h5_pth.iterdir())

# Filter only HDF5 files
h5_files = [str(f) for f in file_list if f.suffix == '.hdf5' and f.is_file()]

# Uncomment if you have arti_info directory
# arti_pth = Path('arti_info')
# Get list of all files and folders in the arti_info directory
# arti_list = list(arti_pth.iterdir())

# Filter only .pkl files
# arti_files = [str(f) for f in arti_list if f.suffix == '.pkl' and f.is_file()]

# Split data: first into training (85%) and Valid (15%)
train_files, valid_files = train_test_split(
    h5_files,
    test_size=0.15,
    random_state=42,  # Use a fixed seed for reproducibility
    shuffle=True      # Shuffle data randomly
)

# Split temporary data into validation (15%) and test (15%)
# valid_files, test_files = train_test_split(
#     temp_files,
#     test_size=0.5,
#     random_state=42,  # Use the same seed
#     shuffle=True
# )

# file_split = {'train': train_files, 'val': valid_files, 'test': test_files}
file_split = {'train': train_files, 'val': valid_files}


for spt, file_dirs in file_split.items():
    print("SPLIT:", spt, "=====================")
    for file_dir in file_dirs:
        f = h5py.File(file_dir, 'r')

        traj_filename = Path(file_dir).stem + '.pkl'
        # Uncomment if you have arti_info directory
        # arti_pkl = arti_pth / traj_filename
        # arti_data = np.load(arti_pkl, allow_pickle=True)
        if 'arti_info' in f.keys():
            # Currently not used
            arti_rgb = f['arti_info/rgb']
            # Angle
            arti_angles = f['arti_info/angles']

        
        ee = f['observations/ee_pose'][:].astype(np.float32)


        N = ee.shape[0]
        ee_relative = ee.copy()

        
        '''
        rx ry rz -> euler angles
        '''

        # Calculate difference with previous timestep (except last timestep)
        ee_relative[:-1, :-1] = ee[1:, :-1] - ee[:-1, :-1]
        ee_relative = ee_relative[:-1]
        
        for i, ee_rel in enumerate(ee_relative):
            ee_relative[i, 3:6] = R.from_rotvec(ee_rel[3:6]).as_euler('xyz')
        
        
        
        
        
        ee = ee_relative[:]
        create_graph(file_dir, ee, args.dest_dir)

        rgb = f['observations/images']
        rgb_dict = {}
        for cam_name in rgb.keys():
            my_rgb = rgb[cam_name][:]
            rgb_dict[cam_name] = my_rgb

        timestep, h, w, _ = my_rgb.shape

        data = []
        timestep -= 1
        for i in range(0, timestep):
            d = {}
            for k, v in rgb_dict.items():
                d[k] = v[i]
            d['action'] = ee[i]
            with open(args.prompt_file, 'r') as f_prompt:
                prompt = f_prompt.read()

            if 'arti_info' in f.keys():  # Only for arti project
                prompt = prompt.replace("joint_state_1", str(arti_angles[0]))
            d['language_instruction'] = prompt

            data.append(d)

        dest_subdir = os.path.join(args.dest_dir, 'data', spt)
        os.makedirs(dest_subdir, exist_ok=True)
        # Generate file name (keep the .npy extension)
        file_name = Path(file_dir).stem + '.npy'
        file_path = os.path.join(dest_subdir, file_name)

        # Save using NumPy's save function
        np.save(file_path, data)
        
        print("Data saved to", file_path)
