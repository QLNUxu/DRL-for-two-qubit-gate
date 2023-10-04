# DRL-for-two-qubit-gate

How to create an environment for this code

conda create -n envname python=3.6
conda activate envname
conda install numpy scipy cython matplotlib pytest pytest-cov jupyter notebook spyder
conda config --append channels conda-forge
conda install qutip==4.4.1
pip install tensorforce[tf]==0.5.2
pip install --upgrade numpy

This should work in Anaconda Prompt...
