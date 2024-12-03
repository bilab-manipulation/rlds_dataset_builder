import numpy as np
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import h5py
import tqdm

h5_pth = Path('h5_data')

# 디렉토리 내 모든 파일 및 폴더 목록 가져오기
file_list = list(h5_pth.iterdir())

# 파일만 필터링
h5_files = [str(f) for f in file_list if f.suffix == '.h5' and f.is_file()]

arti_pth = Path('arti_info')
# 디렉토리 내 모든 파일 및 폴더 목록 가져오기
arti_list = list(h5_pth.iterdir())

# 파일만 필터링
arti_files = [str(f) for f in arti_list if f.suffix == '.pkl' and f.is_file()]


# 데이터 분할: 먼저 학습(70%)과 임시 데이터(30%)로 분할
train_files, temp_files = train_test_split(
    h5_files, 
    test_size=0.3, 
    random_state=42,    # 재현성을 위해 고정된 시드 사용
    shuffle=True        # 데이터를 무작위로 섞기
)

# 임시 데이터를 다시 검증(15%)과 테스트(15%)로 분할
valid_files, test_files = train_test_split(
    temp_files, 
    test_size=0.5, 
    random_state=42,    # 동일한 시드 사용
    shuffle=True
)



file_split = {'train': train_files, 'val': valid_files, 'test': test_files}

cameras = ['image']

# cameras = ['image', 'wrist_image']
for spt, file_dirs in file_split.items():
    print("SPLIT:", spt, "=====================")
    for file_dir in file_dirs:
        assert file_dir.split('/')[-1].split('.')[0].split('_')[-1].isdigit(), file_dir
        file_idx = file_dir.split('/')[-1].split('.')[0].split('_')[-1]
        f = h5py.File(file_dir)
        
        traj_filename = file_dir.split('/')[-1].split('.')[0] + '.pkl'
        arti_pkl = arti_pth / traj_filename
        arti_data = np.load(arti_pkl, allow_pickle=True)
        
        # 현재는 안쓰긴 함
        arti_rgb = arti_data['rgb']
        
        # angle
        arti_angles = arti_data['angles']
        
        ee = f[f'dict_str_traj_{file_idx}']['dict_str_obs']['dict_str_ee_xyzrpy'][:]
        # 복사본 생성하여 수정
        ee_relative = ee.copy()
        
        # 이전 타임스텝과의 차이 계산 (맨 마지막 타임스텝 제외)
        ee_relative[:-1, :-1] = ee[1:, :-1] - ee[:-1, :-1]
        ee_relative = ee_relative[:-1]
        ee = ee_relative[:]
        
        rgb = f[f'dict_str_traj_{file_idx}']['dict_str_obs']['dict_str_rgb'][:]
        timestep, num_camera, _, h, w = rgb.shape
        
        rgb_dict = {}
        for i, camera in enumerate(cameras):
            my_rgb = rgb[:, i, ...].transpose(0, 2, 3, 1)
            rgb_dict[camera] = my_rgb

        data = []
        timestep -= 1
        for i in range(0, timestep, 15):
            d = {}
            for k, v in rgb_dict.items():
                d[k] = v[i]
            d['action'] = ee[i]
            with open('prompt_guideline.txt', 'r') as f:
                prompt = f.read()
                
            prompt.replace("joint_state_1", str(arti_angles[0]))
            d['language_instruction'] = prompt
            
            data.append(d)
        
        dest_dir = f'arti_dataset_two_parts/{spt}'
        os.makedirs(dest_dir, exist_ok=True)
        # 파일 이름 생성 (파일 확장자는 .npy로 유지)
        file_name = Path(file_dir).stem + '.npy'
        file_path = os.path.join(dest_dir, file_name)

        # Numpy의 save 함수로 저장
        np.save(file_path, data)
        print("data saved to", file_path)
        
        
            
                
            
        
        
            
            
        



