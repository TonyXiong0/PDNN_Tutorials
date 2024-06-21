--------------------------------------------------------------------
Paper: 
Physics-driven neural networks for nonlinear micromechanics

Authors: Zhihao Xiong, Ping Yang, Pengyang Zhao

International Journal of Mechanical Sciences

https://doi.org/10.1016/j.ijmecsci.2024.109214

Contact: zhihao.xiong@sjtu.edu.cn

--------------------------------------------------------------------
Setup:
1. Setup environment with conda: conda create -n pytorch python=3.10

2. All the required libraries are in importlib.py

8. Setup PYTHONPATH environment by doing either a.(temporary use) or b.(permanent use): 

a. export PYTHONPATH="$PYTHONPATH:\<workingdir\>/DeepEnergyMethod"
  
b. add the above line to the end of file ~/.bashrc and execute "source ~/.bashrc"

Optional:

To use fem to compare the results with fem, we recommend to install fenics

1. conda config --add channels conda-forge

2. !conda install fenics
