bash install_conda.sh
source /root/miniconda3/bin/activate
conda env create -f conda_env.yml
conda activate flexgen_env
pip install -e .