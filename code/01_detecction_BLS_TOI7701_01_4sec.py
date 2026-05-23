"""
=============================================================================
PHOTOMETRIC DETECTION AND CHARACTERIZATION PIPELINE: TOI 7701.01
=============================================================================

Performs a Two-Stage Box Least Squares (BLS) transit detection across 
multi-sector archival data:
  1. Period search on a safely detrended light curve to maximize SNR.
  2. Depth and morphology characterization on the raw (un-detrended) data.
  
Features automated cross-pipeline sector retrieval (SPOC, TESS-SPOC, QLP)
with built-in deduplication. Includes a Savitzky-Golay window sensitivity 
analysis to justify transit depths, robust epoch assignment for odd/even 
vetting, and global transit masking. Exports phase-folded master light 
curve data for downstream statistical validation.

TARGET:
    TIC ID  : 122522333
    TOI     : 7701.01
    Sectors : Multi-Sector (All available)

OUTPUT:
    - Diagnostic figure (Global LC, BLS Periodogram, Folded, Odd/Even)
    - TOI7701.01_LC_folded_binned_4sec.csv (30-min binned for plotting)
    - TOI7701.01_LC_folded_raw_4sec.csv    (Full cadence, master for TRICERATOPS)
"""

import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import gc
import warnings
import pandas as pd

warnings.simplefilter('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TIC_ID = "TIC 122522333"
TOI_ID = "TOI 7701.01"

SEARCH_PERIOD_MIN = 20.50
SEARCH_PERIOD_MAX = 20.75
DURATIONS_TO_TEST = np.linspace(0.05, 0.35, 12)

R_STAR_SOLAR = 1.76        
R_SUN_TO_EARTH = 109.076   

print(f"[INFO] Initializing Two-Stage BLS pipeline for {TOI_ID} ({TIC_ID})")
print("-" * 70)

# ---------------------------------------------------------------------------
# DATA RETRIEVAL AND BASIC CLEANING (MULTI-SECTOR ARCHITECTURE)
# ---------------------------------------------------------------------------
print("[INFO] Querying MAST for ALL available TESS sectors...")

try:
    # Unified query targeting all official TESS pipelines simultaneously
    search = lk.search_lightcurve(TIC_ID, author=("SPOC", "TESS-SPOC", "QLP"))
    
    if len(search) == 0:
        raise ValueError("No TESS light curve data found for this target in any pipeline.")

    print(f"[INFO] {len(search)} total sector(s) found across SPOC and QLP.")

    sectors_downloaded = []
    lc_collection = []
    
    for lc_item in search:
        sector_num = lc_item.mission[0] 
        
        # Deduplication filter: Prioritizes the first encountered pipeline 
        # (typically SPOC high-cadence) and skips redundant data for the same sector
        if sector_num in sectors_downloaded:
            continue 
            
        try:
            print(f"[INFO] Downloading {sector_num} (Author: {lc_item.author[0]})")
            temp_lc = lc_item.download()
            
            # Basic cleaning: remove NaNs and normalize flux to 1.0 (crucial for multi-sector stitching)
            temp_lc = temp_lc.remove_nans().normalize()
            
            # Asymmetric sigma clipping: aggressively remove positive stellar flares (4-sigma) 
            # while preserving deep planetary transits (15-sigma)
            temp_lc = temp_lc.remove_outliers(sigma_upper=4, sigma_lower=15)
            
            lc_collection.append(temp_lc)
            sectors_downloaded.append(sector_num)
            
            del temp_lc
            gc.collect()
            
        except Exception as e:
            print(f"[WARNING] Could not process {sector_num}: {e}")
            continue

    if not lc_collection:
        raise ValueError("No usable light curve data after cleaning.")

    # Stitch all independent sectors into a continuous master time series
    lc_combined = lk.LightCurveCollection(lc_collection).stitch()
    lc_combined.flux = lc_combined.flux.astype(np.float32)
    
    print(f"[INFO] ALL sectors stitched successfully. Total cadences: {len(lc_combined)}")

except Exception as e:
    print(f"[ERROR] Pipeline aborted: {e}")
    exit()

# ---------------------------------------------------------------------------
# STAGE 1: DETRENDED BLS SEARCH (Maximizing periodicity detection)
# ---------------------------------------------------------------------------
print("\n[INFO] Running high-resolution multi-duration BLS search (on flattened LC)...")
# Conservative flattening to remove low-frequency stellar/instrumental variability
lc_search = lc_combined.flatten(window_length=1501)

period_grid = np.linspace(SEARCH_PERIOD_MIN, SEARCH_PERIOD_MAX, 100000)

bls = lc_search.to_periodogram(
    method='bls',
    period=period_grid,
    duration=DURATIONS_TO_TEST
)

best_period   = bls.period_at_max_power.value
best_t0       = bls.transit_time_at_max_power.value
best_duration = bls.duration_at_max_power.value

print(f"[RESULT] BLS best-fit parameters:")
print(f"         Period   : {best_period:.6f} days")
print(f"         T0       : {best_t0:.4f} BTJD")
print(f"         Duration : {best_duration * 24:.2f} hours")

# ---------------------------------------------------------------------------
# STAGE 2: RAW CHARACTERIZATION (Unbiased Depth & SNR calculation)
# ---------------------------------------------------------------------------
print("\n[INFO] Computing transit SNR from un-detrended photometry...")

# Fold the RAW light curve using the high-precision BLS ephemeris
lc_folded = lc_combined.fold(period=best_period, epoch_time=best_t0)

# Masking for depth and SNR estimation
phase_mask_transit = (np.abs(lc_folded.time.value) < (best_duration * 0.55))
phase_mask_out     = (np.abs(lc_folded.time.value) > (best_duration * 2.0))

flux_in  = lc_folded.flux[phase_mask_transit]
flux_out = lc_folded.flux[phase_mask_out]

if len(flux_in) > 5 and len(flux_out) > 5:
    median_flux_out     = np.nanmedian(flux_out)
    median_flux_in      = np.nanmedian(flux_in)
    depth_calc_natural  = median_flux_out - median_flux_in
    noise               = np.nanstd(flux_out)
    snr_final           = (depth_calc_natural / noise) * np.sqrt(len(flux_in))
    depth_ppm_natural   = depth_calc_natural * 1e6
    noise_ppm           = noise * 1e6
else:
    print("[WARNING] Insufficient in-transit cadences for reliable SNR estimate.")
    snr_final = depth_ppm_natural = depth_calc_natural = noise_ppm = 0.0

# Geometric Earth Radius estimate (Rp = sqrt(delta) * R_star)
depth_frac = depth_ppm_natural * 1e-6
Rp_earth   = np.sqrt(depth_frac) * R_STAR_SOLAR * R_SUN_TO_EARTH

# Build global mask for diagnostic plotting
global_epochs = np.floor((lc_combined.time.value - best_t0 + 0.5 * best_period) / best_period)
expected_transit_times = best_t0 + global_epochs * best_period
is_in_transit_global = np.abs(lc_combined.time.value - expected_transit_times) < (best_duration * 0.55)

# ---------------------------------------------------------------------------
# SAVITZKY-GOLAY WINDOW SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------------
print("\n[INFO] Running detrending sensitivity analysis...")
print(f"       Reference depth (un-detrended): {depth_ppm_natural:.0f} ppm")

window_lengths = [401, 1201, 1801, 2501, 3501]
depth_ppm_flat = 0.0

for wl in window_lengths:
    lc_test        = lc_combined.flatten(window_length=wl)
    lc_test_folded = lc_test.fold(period=best_period, epoch_time=best_t0)

    f_in  = lc_test_folded.flux[np.abs(lc_test_folded.time.value) < (best_duration * 0.55)]
    f_out = lc_test_folded.flux[np.abs(lc_test_folded.time.value) > (best_duration * 2.0)]

    if len(f_in) > 0 and len(f_out) > 0:
        d_ppm = (np.nanmedian(f_out) - np.nanmedian(f_in)) * 1e6
        loss  = abs(depth_ppm_natural - d_ppm) / depth_ppm_natural * 100
        print(f"       WL: {wl:5d} cadences | Depth: {d_ppm:5.0f} ppm | "
              f"Loss vs natural: {loss:5.1f}%")
        if wl == 2501:
            depth_ppm_flat = d_ppm

depth_diff_percent = abs(depth_ppm_natural - depth_ppm_flat) / depth_ppm_natural * 100

# ---------------------------------------------------------------------------
# DIAGNOSTIC FIGURES (4 Panels)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 12))
gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

# Panel A: Global Un-detrended Light Curve with Transit Mask
ax0 = fig.add_subplot(gs[0, :])
ax0.scatter(lc_combined.time.value[~is_in_transit_global], lc_combined.flux[~is_in_transit_global], 
            color='black', s=2, alpha=0.5, label='Out of Transit')
ax0.scatter(lc_combined.time.value[is_in_transit_global], lc_combined.flux[is_in_transit_global], 
            color='red', s=10, alpha=0.9, zorder=5, label='In Transit Mask')
ax0.set_title(f"Global Un-detrended Light Curve — {TOI_ID}", fontsize=13, fontweight='bold')
ax0.set_xlabel("Time [BTJD]", fontsize=11)
ax0.set_ylabel("Normalized Flux", fontsize=11)
ax0.legend(loc='upper right')
ax0.grid(True, alpha=0.3)

# Panel B: BLS Periodogram
ax1 = fig.add_subplot(gs[1, :])
ax1.plot(bls.period, bls.power, color='black', lw=1)
ax1.axvline(best_period, color='red', alpha=0.5, lw=6, label=f'Peak: {best_period:.6f} d')
ax1.set_title("BLS Periodogram (run on flattened LC)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Period [days]", fontsize=11)
ax1.set_ylabel("BLS Power", fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel C: Phase-folded transit (un-detrended)
ax2     = fig.add_subplot(gs[2, 0])
bin_size = 30 / (24 * 60)  # 30 minutes in days
lc_binned = lc_folded.bin(time_bin_size=bin_size)

lc_folded.scatter(ax=ax2, color='gray', alpha=0.3, s=2, label='Raw flux')
lc_binned.scatter(ax=ax2, color='red', s=25, marker='o', label='30-min binned', zorder=5)
ax2.set_title(f"Phase-folded Transit — T0 = {best_t0:.4f} BTJD", fontsize=13, fontweight='bold')
ax2.set_xlabel("Phase [days]", fontsize=11)
ax2.set_ylabel("Normalized Flux", fontsize=11)
ax2.set_xlim(-0.25, 0.25)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# Panel D: Odd/Even transit vetting (Robust Epoch Assignment)
ax3 = fig.add_subplot(gs[2, 1])
transit_epochs = np.floor((lc_combined.time.value - best_t0 + 0.5 * best_period) / best_period)
is_even = (transit_epochs % 2 == 0)

lc_even = lc_combined[is_even].fold(period=best_period, epoch_time=best_t0)
lc_odd  = lc_combined[~is_even].fold(period=best_period, epoch_time=best_t0)

if len(lc_even) > 10:
    lc_even.bin(time_bin_size=bin_size).plot(ax=ax3, color='blue', lw=2.5, label='Even epochs')
if len(lc_odd) > 10:
    lc_odd.bin(time_bin_size=bin_size).plot(ax=ax3, color='orange', lw=2.5, linestyle='--', label='Odd epochs')

ax3.set_title("Odd/Even Transit Depth Comparison", fontsize=13, fontweight='bold')
ax3.set_xlabel("Phase [days]", fontsize=11)
ax3.set_ylabel("Normalized Flux", fontsize=11)
ax3.set_xlim(-0.25, 0.25)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"DETECTION REPORT — {TOI_ID} ({TIC_ID})")
print("=" * 70)
print(f"  Orbital period     : {best_period:.6f} days")
print(f"  Epoch (T0)         : {best_t0:.4f} BTJD")
print(f"  Transit duration   : {best_duration * 24:.2f} hours")
print(f"  Transit depth      : {depth_ppm_natural:.0f} ppm (un-detrended)")
print(f"  Transit depth      : {depth_ppm_flat:.0f} ppm (SG WL=2501, {depth_diff_percent:.1f}% loss)")
print(f"  SNR                : {snr_final:.2f}")
print(f"  Noise (sigma_oot)  : {noise_ppm:.0f} ppm")
print(f"  Rp (geometric)     : {Rp_earth:.2f} R_earth")
print("-" * 70)
if snr_final > 10:
    print("  Signal status: High-significance transit-like signal (SNR > 10)")
elif snr_final > 7:
    print("  Signal status: Marginal signal (7 < SNR <= 10)")
else:
    print("  Signal status: WEAK (SNR <= 7) — interpret with caution")
print("=" * 70)

# ---------------------------------------------------------------------------
# DATA EXPORT: PROFESSIONAL NAMING CONVENTION
# ---------------------------------------------------------------------------
print("\n[INFO] Exporting phase-folded data for statistical validation...")

# 1. EXPORT FOR VISUAL DIAGNOSTICS (Binned)
df_binned = pd.DataFrame({
    'phase_days'  : lc_binned.time.value,
    'flux'        : lc_binned.flux.value,
    'flux_err'    : lc_binned.flux_err.value
}).sort_values('phase_days')

fname_binned = f"{TOI_ID.replace(' ', '')}_LC_folded_binned_4sec.csv"
df_binned.to_csv(fname_binned, index=False, header=False)
print(f"[INFO] Saved Diagnostic: {fname_binned}")

# 2. EXPORT FOR TRICERATOPS AND MASTER ANALYSIS (Raw)
# Folded directly from lc_combined (no flatten, no rebinning)
df_raw = pd.DataFrame({
    'time_from_midtransit' : lc_folded.time.value,
    'flux'                 : lc_folded.flux.value,
    'flux_err'             : lc_folded.flux_err.value
}).sort_values('time_from_midtransit')

fname_raw = f"{TOI_ID.replace(' ', '')}_LC_folded_raw_4sec.csv"
df_raw.to_csv(fname_raw, index=False, header=False)

print(f"[INFO] Saved Master/TRICERATOPS Input: {fname_raw}")
print("[INFO] Pipeline complete.")