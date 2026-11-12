# Physics-Driven Neural Network Tutorials

Eight PyTorch notebooks introduce function approximation, automatic differentiation, and physics-driven learning for differential equations and solid mechanics. The uniaxial-tension examples show how to train networks using either potential energy or PDE residuals.

## Quick start

From the project root, create a Python 3.11 environment and launch JupyterLab.

```bash
conda create -n pdnn python=3.11
conda activate pdnn
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open a notebook from the project root and select **Run → Run All Cells**. Use the Python kernel from the `pdnn` environment. The notebooks use CUDA when PyTorch detects it and otherwise run on CPU.

You can run Notebook 2 immediately with the bundled sine model. If you run Notebook 1 first, Notebook 2 uses your new weights from `output/Case1/model.pt`. The notebooks create their output directories on demand.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| [1. Function approximation](1_Approximate_a_function.ipynb) | Fit a multilayer network to a sine function and save the trained weights. |
| [2. Automatic differentiation](2_Autograd.ipynb) | Differentiate the model from Notebook 1 and compare with the analytical derivative. |
| [3. 1D Poisson equation](3_1D_Poisson.ipynb) | Train on the PDE residual and Dirichlet boundary conditions. |
| [4. Energy-based uniaxial tension](4_2D_UnaxisTension_EnergyBased.ipynb) | Model a 2D plate by minimizing its total potential energy. |
| [5. PDE-based uniaxial tension](5_2D_UnaxisTension_PDEBased.ipynb) | Predict displacement and stress and enforce equilibrium, constitutive relations, and boundary conditions. |
| [6. Inverse Burgers' equation](6_InverseProblem_BurgersEquation.ipynb) | Infer convection and diffusion coefficients from sampled solution data. |
| [7. Self-adaptive residual weighting](7_self-adaptive_learning_weighting_residuals.ipynb) | Adapt the spatial weights of PDE residuals in the 2D tension problem. |
| [8. U-Net function approximation](8_Unet_Approximate_a_function.ipynb) | Fit a 2D function and compare convolution-based derivatives of the reference field with analytical derivatives. |

## Training and outputs

For a quick run, set `steps = 2` in the **Training parameters** cell. In Notebooks 4, 5, and 7, also set `numElemX` and `num_test_x` to `12`, and `numElemY` and `num_test_y` to `8`. In Notebook 6, set `N_f = 64`. In Notebook 8, set `resolution = 64`. These settings exercise the training and plotting workflow. Use the original settings when evaluating solution accuracy.

Notebooks 4, 5, and 7 write structured-grid `.vts` files under `output/Case4/`, `output/Case5/`, and `output/Case7/`. Open the files in ParaView to inspect displacement and stress. Notebook 6 reads `Data/Burgers.mat` and writes a comparison figure to `output/Case6/`.

The [checkpoints directory](checkpoints/) contains model weights for Notebooks 1, 3, 4, and 5. The sine model was trained with the default settings in Notebook 1. The other three files are reference weights from the original repository. New training runs write their results under `output/`.

## Citation

Zhihao Xiong, Ping Yang, and Pengyang Zhao. “Physics-driven neural networks for nonlinear micromechanics.” *International Journal of Mechanical Sciences* 273 (2024), 109214. [DOI](https://doi.org/10.1016/j.ijmecsci.2024.109214)

## Burgers' equation sources

Notebook 6 draws on two PINN teaching repositories. The bundled `Data/Burgers.mat` matches the dataset in both repositories byte for byte. [NABLA-SciML dataset](https://github.com/jdtoscano94/NABLA-SciML/blob/main/Learning-PIML-in-Python-PINNs-DeepONets-RBA/Data/Burgers.mat) · [Physics-Informed-Neural-Networks dataset](https://github.com/prateekbhustali/Physics-Informed-Neural-Networks/blob/main/PyTorch/Burgers%27%20Equation/Data/burgers_shock_mu_01_pi.mat)

Contact Zhihao Xiong at zhihao.xiong@sjtu.edu.cn.
