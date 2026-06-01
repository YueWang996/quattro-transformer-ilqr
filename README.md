[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)



# Quattro: Transformer-Accelerated iterative Linear Quadratic Regulator (iLQR)

Quattro is an open-source framework designed to enhance the efficiency of iterative Linear Quadratic Regulators (iLQR) through the innovative integration of Transformer models. By accelerating intermediate computations in iLQR algorithms, Quattro significantly improves real-time optimal control capabilities for nonlinear robotic systems.

<div align="center">
  <img src="figures/mujoco_quadrotor.png" alt="Cart-pole and Quadrotor Visualization" style="max-width:90%; height:auto;">
</div>

## Overview

Real-time optimal control remains challenging in robotics due to the sequential nature of traditional iLQR methods. Quattro addresses this by employing a Transformer model to concurrently predict feedback and feedforward matrices, allowing parallel computation without sacrificing control accuracy.

<div align="center">
  <img src="figures/arch-ilqr-tf.png" alt="Architecture Overview" style="max-width:70%; height:auto;">
</div>


## Key Contributions
- **Transformer-Accelerated iLQR**: Parallel computation enabled by Transformer-based prediction of intermediate feedback and feedforward matrices.
- **Validated Performance**: Demonstrated substantial acceleration and accuracy on cart-pole and quadrotor systems.
- **FPGA Implementation**: Utilizes a customized Transformer accelerator IP on FPGA, achieving up to 27.3× performance improvements and significant power efficiency.

## Performance Highlights
- **Algorithm-Level Acceleration**:
<div align="center"> <img src="figures/cartpole_result.png" alt="Cart-Pole Simulation Result" width="400"/> <p><em>Figure 1: Cart-Pole simulation result using Quattro. Up to 5.3× per iteration.</em></p> </div> <div align="center"> <img src="figures/quadrotor_result.png" alt="Quadrotor Simulation Result" width="400"/> <p><em>Figure 2: Quadrotor simulation result using Quattro. Up to 27× per iteration.</em></p> </div>

- **Overall MPC Speedup (with Apple Silicon M4 Pro)**:
  - Cart-Pole: 2.8×.
  - Quadrotor: 17.8×.

## Installation
We recommend use virtual environment e.g. `conda`. The project is based on `Python 3.10` or later versions.

```bash
# Create and activate a new virtual environment
conda create --name quattro python=3.10
conda activate quattro

# Clone the repository and navigate to the project folder
git clone https://github.com/YueWang996/quattro-transformer-ilqr
cd quattro-transformer-ilqr

# Option 1 (Recommended): Install as an editable package
pip install -e .

# Option 2: Alternatively, install the dependencies manually
pip install -r requirements.txt
```


## Usage

### Running Simulations

Quattro comes with example simulations to illustrate its performance on different robotic systems.

- **Cart-Pole Simulation:**  
  Navigate to the `examples/cartpole/` directory and run:

  ```bash
  python cartpole_sim.py
  ```

- **Quadrotor Simulation:**  
  Similarly, navigate to the `examples/quadrotor/` directory and run:

  ```bash
  python quadrotor_sim.py
  ```

### Training the Transformer Model

To train the Transformer for iLQR acceleration:

- **Data Collection:**  
  Use the provided data collection script located at `examples/cartpole/training/training_data_collection.py` to generate simulation logs. This script harnesses multi-core processing for efficiency.

- **Model Training:**  
  You have two options:
  - Run `transformer_training.py` from the same directory for script-based training. Suitable to run on a remote server.
  - Alternatively, use the Jupyter Notebook `transformer_training.ipynb` for an interactive guide.


## Citation
Please cite this project if you find it useful:
```
@inproceedings{wang2025quattro,
  title={Quattro: transformer-accelerated iterative linear quadratic regulator framework for fast trajectory optimization},
  author={Wang, Yue and Wang, Haoyu and Li, Zhaoxing},
  booktitle={2025 IEEE 64th Conference on Decision and Control (CDC)},
  pages={6629--6636},
  year={2025},
  organization={IEEE}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

