# Statistical Validation of TOI-7701.01

Code repository for the paper:  
*Statistical Validation of TOI-7701.01: A Transiting Companion at the Planet–Brown Dwarf Boundary*

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-validated%20companion-lightgrey)

**Author:** Biel Escolà-Rodrigo  
**Date:** May 2026  

---

## Abstract
We present the statistical validation of TOI-7701.01, a Saturn-sized transiting companion candidate identified by TESS, using a photometric–astrometric pipeline and iterative Bayesian false positive probability analysis. The signal satisfies the validation criteria of Giacalone et al. (2021) (FPP = 0.00184, NFPP = 0), but its radius lies at the boundary between giant planets, brown dwarfs, and low-mass stars. We therefore classify it as a statistically validated transiting companion pending dynamical confirmation.

---

## Key Results
- Period: 20.6116 days  
- Transit depth: 1608 ppm  
- Radius: ~7.7–8.2 R⊕  
- SNR: ~30  
- FPP: 0.00184  
- NFPP: 0  

---

## Method
Detection — BLS on TESS PDCSAP light curve  
Vetting — odd/even, centroid stability, Gaia DR3 contamination  
Validation — 20× triceratops runs (Bayesian FPP estimation)

---

## Reproducibility
git clone https://github.com/youruser/TOI-7701.01-Validation.git  
cd TOI-7701.01-Validation  
pip install -r requirements.txt  
python code/01_detection_BLS.py  

For triceratops, use a clean Python 3.10 environment.

---

## Notes
- Based on TESS Sector 97  
- No additional detrending applied (depth-sensitive regime)  
- Photometric validation only — RV follow-up required  

---

## Citation
@article{escola2026toi7701,
  title={Statistical Validation of TOI-7701.01},
  author={Escolà Rodrigo, Biel},
  year={2026}
}
