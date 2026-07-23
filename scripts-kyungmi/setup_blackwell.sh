bash install_conda.sh
source /root/miniconda3/bin/activate
conda env create -f conda_env_blackwell.yml
conda activate flexgen_env_blackwell
pip install -e .