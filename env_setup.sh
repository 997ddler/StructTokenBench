conda create -n pstbench python=3.11

# conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia ## original

pip install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"


pip install lmdb
pip install --upgrade packaging  
pip install hydra-core
pip install lightning
pip install transformers
pip install deepspeed
pip install -U tensorboard
pip install ipdb

# pip install esm
pip install esm torch==2.2.2

pip install cloudpathlib
pip install pipreqs
pip install lxml
pip install proteinshake
pip install tmtools

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install accelerate
pip install torch_geometric
pip install line_profiler
pip install mini3di
pip install dm-tree
pip install colorcet
pip install ogb==1.2.1
pip install sympy
pip install ase
pip install torch-cluster

pip install jax==0.4.25
pip install tensorflow
pip install biopython
pip install seaborn


#### When running
pip install httpx