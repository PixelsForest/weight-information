# Weight Information Project in Neural Computation

Code for our paper:

**"Quantifying information stored in synaptic connections rather than in firing activities of neural networks"**
*Neural Computation, 2026*

**Authors:**
Xinhao Fan, Shreesh P Mysore

---

## Overview

This repository contains code and simulations for reproducing the main results in the paper.
The focus is on quantifying information stored in synaptic weights and validating theoretical predictions through numerical experiments.

---

## Repository Structure

* `src/`
  Core implementation of the methods and computational routines used in the paper.
  This includes functions for approximation, simulation, and statistical analysis.

* `notebooks/`
  Jupyter notebooks for reproducing figures in the paper:

  * Each notebook corresponds to one main figure
  * Running the notebooks will regenerate the results and plots

* `data/`
  Precomputed simulation data used in the paper (e.g., for generating final figures).
  This allows reproducing results without rerunning expensive simulations.

* `requirements.txt`
  Python dependencies required to run the code.

---

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` folder to reproduce the figures.

---

## Notes

* Notebooks are pre-executed and include outputs for direct viewing.
* For best reproducibility, run all notebooks from the repository root directory.

---

## Citation

If you find this work useful, please cite:

```text
[Add BibTeX here after publication]
```

---

## License

This project is licensed under the MIT License.
