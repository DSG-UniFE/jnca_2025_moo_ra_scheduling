# MO-6G: Multi-Objective Orchestration for the 6G Compute Continuum"

Code repository for "MO-6G: Multi-Objective Orchestration for the 6G Compute Continuum", submitted to  Journal of Network and Computer Applications (JNCA). The manuscript is currently under review. 

## Repository Organization

### Multi-Objective-ILP

All the Julia files are located the main directory of this repository. Each file contains the problem's formulation described in the manuscript and it is set to call the Gurobi optimizer.

### Multi-Objective Evolutionary Algorithms (MOEA)

All the python code is under the MOEA directory. To install the Python dependencies run:

```bash
# create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install requirements.txt
```

Then to run to run the 3-objs optimization, do the following:

```bash
cd MOEA
python solve.py
```



