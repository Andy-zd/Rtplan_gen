# Rtplan_gen
We demonstrate how to generate route planning data in VLM-3R.
1. Environment setup
```
git clone https://github.com/Andy-zd/Rtplan_gen
cd Rtplan_gen
```

```
conda create -n rtplan python=3.9 cmake=3.14.0
conda activate rtplan

### habitat-sim

git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless
cd ..

### habitat-lab

git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
cd ..

### assimp.
conda install -c conda-forge cmake>=3.22

cd habitat-sim
git clone https://github.com/assimp/assimp.git
cd assimp
cmake CMakeLists.txt
cmake --build .

```

2. Data preparation

See instructions here: https://github.com/VITA-Group/VLM-3R/tree/main/vlm_3r_data_process#1-datasets-with-ground-truth-annotations


3. Data generation

```
# Path generation ✅
python generate_paths.py --dataset_path ${PATH_TO_DATASET} --assimp_path ${PATH_TO_ASSIMP} --dataset_type ${DATASET_TYPE}

## example
python generate_paths.py --dataset_path ${PATH_TO_SCANNET} --assimp_path ./habitat-sim/assimp/bin/assimp --dataset_type scannet

python generate_paths.py --dataset_path ${PATH_TO_SCANNETPP} --assimp_path ./habitat-sim/assimp/bin/assimp --dataset_type scannetpp

python generate_paths.py --dataset_path ${PATH_TO_ARKITSCENE} --assimp_path ./habitat-sim/assimp/bin/assimp --dataset_type arkitscene

# QA pairs generation 
python generate_qas.py --dataset_type scannet --processed_data_path ${processed_data_path}
python generate_qas.py --dataset_type scannetpp --processed_data_path ${processed_data_path}
python generate_qas.py --dataset_type arkitscene --processed_data_path ${processed_data_path}


# Balance choice distribution and merge datasets
python routeplan_analysis.py 

# Transfer to target format (multiple choice)
python json_transfer.py

```
We will get qa_pairs.json file for route planning.

### To-Do List

- [✅] Path generation
- [✅] QA pairs generation 
- [✅ ] Balance choice distribution and merge datasets 
- [✅] Transfer to target format (multiple choice)