bash install_conda.sh
source /root/miniconda3/bin/activate
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
conda env create -f conda_env.yml
conda activate flexgen_env
pip install -e .