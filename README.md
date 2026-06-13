# Statistical Validation of TOI-7701.01

Code repository for the paper:  
*Statistical Validation of TOI-7701.01: A Transiting Companion at the Planet–Brown Dwarf Boundary*

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-validated%20companion-lightgreen)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20029584.svg)](https://doi.org/10.5281/zenodo.20029584)

**Author:** Biel Escolà-Rodrigo  
**Date:** June 2026  

---

## Abstract
We present the formal statistical validation of TOI-7701.01 (TIC 122522333), a sub-Saturn transiting companion candidate orbiting a bright F-type subgiant host star. Utilizing an independent vetting pipeline, we leverage multi-sector space kinematics and the `triceratops` Bayesian framework. A key methodological feature of this work is the dual photometric extraction: utilizing un-detrended Simple Aperture Photometry (SAP) to evaluate structural false positive scenarios, and instrumentally corrected PDCSAP data for geometric characterization. The multi-iteration MCMC ensemble firmly validates the companion well below the rigorous 1.5% FPP threshold.

---

## Key Results
- **Orbital Period:** 20.613811 days (PDCSAP) / 20.616076 days (SAP)
- **Transit Depth:** 1767 ppm (geometric) / 2417 ppm (natural un-detrended)
- **Radius:** ~7.86 – 8.07 R_Earth
- **SNR:** 31.49 
- **Global FPP:** 0.00191 ± 0.00247
- **NFPP:** < 10^-6

---

## Method
- **Geometric Characterization:** BLS periodogram search on Sector 97 TESS PDCSAP short-cadence (120s) light curves.
- **Spatial Vetting:** Odd/even depth comparison, sub-pixel centroid stability tracking, and Gaia DR3 astrometric contamination propagation (RUWE analysis).
- **Statistical Validation:** 20× iteration `triceratops` MCMC ensemble applied exclusively to un-detrended SAP data to preserve exact field dilution metrics and prevent artificial transit suppression.

---

## Reproducibility
```
git clone https://github.com/biesro/TOI-7701.01-Validation.git  
cd TOI-7701.01-Validation  
pip install -r requirements.txt  
python code/detection_BLS_TOI7701_01.py
...
``` 

For triceratops, use a clean Python 3.10 environment. I recommend following steps in the readme.md of https://github.com/JGB276/TRICERATOPS-plus/tree/main

---

## Notes
- The validation explicitly avoids continuous flattening routines (like Savitzky-Golay) on the SAP array to prevent data clipping prior to the Bayesian MCMC fit. 
- The companion resides squarely within the "brown dwarf desert," making its scale strongly point toward a planetary internal structure.
- Precision radial velocity (PRV) follow-up is encouraged to definitively break the mass-radius degeneracy.

---

## 📄 Citation

If you use this methodology or code, please cite the associated manuscript (currently in preparation for submission):

```bibtex
@article{escola2026toi7701,
  title={Comprehensive Statistical Validation of TOI-7701.01: A Sub-Saturn Companion at the Giant Planet Boundary},
  author={Escolà-Rodrigo, Biel},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2026}
  url={...}
}
```

Note: This citation will be updated after submission on arXiv.
