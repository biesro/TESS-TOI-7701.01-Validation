# Statistical Validation of TOI-7701.01

Code repository for the paper:  
*Statistical Validation of TOI-7701.01: A Transiting Companion at the Planet–Brown Dwarf Boundary*

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-validated%20companion-lightgreen)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20029584.svg)](https://doi.org/10.5281/zenodo.20029584)

**Author:** Biel Escolà-Rodrigo  
**Date:** May 2026  

---

## Abstract
We present the statistical validation of TOI-7701.01, a Saturn-sized transiting companion candidate identified by TESS, using a photometric–astrometric pipeline and iterative Bayesian false positive probability analysis. The signal satisfies the validation criteria of Giacalone et al. (2021) (FPP = 0.00191, NFPP < 10**-6), but its radius lies at the boundary between giant planets, brown dwarfs, and low-mass stars. We therefore classify it as a statistically validated transiting companion pending dynamical confirmation.

---

## Key Results
- Period: 20.609766 days  
- Transit depth: 1752 ppm  
- Radius: ~7.9–8 R⊕  
- SNR: ~35.1  
- FPP: 0.00243  
- NFPP: 0  

---

## Method
Detection — BLS on TESS PDCSAP light curve  
Vetting — odd/even, centroid stability, Gaia DR3 contamination  
Validation — 20× triceratops runs (Bayesian FPP estimation)

---

## Reproducibility
git clone https://github.com/biesro/TOI-7701.01-Validation.git  
cd TOI-7701.01-Validation  
pip install -r requirements.txt  
python code/01_detection_BLS.py  

For triceratops, use a clean Python 3.10 environment. I recommend following steps in the readme.md of https://github.com/JGB276/TRICERATOPS-plus/tree/main

---

## Notes
- Based on TESS Sector 3,4,30,97  
- No additional detrending applied (depth-sensitive regime)  
- Photometric validation only — RV follow-up required

---

## 📄 Citation

If you use this methodology or code, please cite the associated manuscript (currently in preparation for submission):

```bibtex
@article{escola2026toi7701,
  title={A Statistically Validated Transiting Companion at the Giant Planet Boundary: TOI-7701.01},
  author={Escolà-Rodrigo, Biel},
  journal={In Preparation},
  year={2026}
}
```

Note: This citation will be updated upon formal publication or arXiv submission.
