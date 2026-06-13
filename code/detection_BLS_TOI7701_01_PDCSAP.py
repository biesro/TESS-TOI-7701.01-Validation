"""
=============================================================================
PHOTOMETRIC DETECTION AND CHARACTERIZATION PIPELINE: TOI 7701.01
=============================================================================

Performs a Two-Stage Box Least Squares (BLS) transit detection:
  1. Period search on a safely detrended light curve to maximize SNR.
  2. Depth and morphology characterization on the raw (un-detrended) data.


TARGET:
    TIC ID : 122522333
    TOI    : 7701.01
    Sector : 97 (SPOC, 2-minute cadence)

OUTPUT:
    - Diagnostic figure (Global LC, BLS Periodogram, Folded, Odd/Even)"""

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
# DATA RETRIEVAL AND BASIC CLEANING
# ---------------------------------------------------------------------------
print("[INFO] Querying MAST for TESS light curves...")

try:
    search = lk.search_lightcurve(TIC_ID, author="SPOC", exptime=120)
    if len(search) == 0:
        search = lk.search_lightcurve(TIC_ID, author="TESS-SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(TIC_ID, author="QLP")
    if len(search) == 0:
        raise ValueError("No TESS light curve data found for this target.")

    print(f"[INFO] {len(search)} sector(s) found.")

    lc_collection = []
    for lc_item in search:
        try:
            print(f"[INFO] Downloading sector: {lc_item.mission[0]}")
            temp_lc = lc_item.download()
            # Initial cleaning: remove NaNs, normalize flux, and clip outliers
            temp_lc = temp_lc.remove_nans().normalize()
            temp_lc = temp_lc.remove_outliers(sigma_upper=4, sigma_lower=15)
            lc_collection.append(temp_lc)
            del temp_lc
            gc.collect()
        except Exception as e:
            print(f"[WARNING] Could not process sector: {e}")
            continue

    if not lc_collection:
        raise ValueError("No usable light curve data after cleaning.")

    lc_combined = lk.LightCurveCollection(lc_collection).stitch()
    lc_combined.flux = lc_combined.flux.astype(np.float32)
    print(f"[INFO] Light curve assembled. Total cadences: {len(lc_combined)}")

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
ax0.set_ylim(0.99, 1.01)
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
ax2.set_ylim(0.995, 1.005)
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