"""
01_detection_BLS_TOI7701_01.py
=============================================================================
PHOTOMETRIC DETECTION AND CHARACTERIZATION PIPELINE: TOI 7701.01
=============================================================================

Performs Box Least Squares (BLS) transit detection on TESS PDCSAP photometry
for TOI 7701.01 (TIC 122522333). Includes a Savitzky-Golay window sensitivity
analysis to justify the no-detrending approach, robust SNR computation, and
odd/even transit vetting. Exports phase-folded light curve data for downstream
statistical validation.

TARGET:
    TIC ID : 122522333
    TOI    : 7701.01
    Sector : 97 (SPOC, 2-minute cadence)

OUTPUT:
    - Diagnostic figure (BLS periodogram, folded transit, odd/even check)
    - TOI7701.01_Folded_Binned_RedPoints.csv  (30-min binned, phase-folded)
    - TOI7701.01_Folded_Raw_GrayPoints.csv    (un-binned, phase-folded)

DEPENDENCIES:
    lightkurve, numpy, matplotlib, pandas
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

# BLS period search range (days) — centered on known period ~20.6 d
SEARCH_PERIOD_MIN = 20.50
SEARCH_PERIOD_MAX = 20.75

# Transit duration grid (days) for multi-duration BLS
DURATIONS_TO_TEST = np.linspace(0.05, 0.35, 12)

# Host star radius from TIC (Stassun et al. 2019), used for geometric Rp estimate
R_STAR_SOLAR = 1.76        # R_sun
R_SUN_TO_EARTH = 109.076   # conversion factor

print(f"[INFO] Initializing BLS pipeline for {TOI_ID} ({TIC_ID})")
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
            # Basic quality filtering: remove NaNs, normalize, clip outliers.
            # No Savitzky-Golay flattening applied — see sensitivity analysis below.
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
# BLS TRANSIT SEARCH (MULTI-DURATION)
# ---------------------------------------------------------------------------
print("\n[INFO] Running high-resolution multi-duration BLS search...")
print(f"       Period range: [{SEARCH_PERIOD_MIN}, {SEARCH_PERIOD_MAX}] days | Steps: 100,000")

period_grid = np.linspace(SEARCH_PERIOD_MIN, SEARCH_PERIOD_MAX, 100000)

bls = lc_combined.to_periodogram(
    method='bls',
    period=period_grid,
    duration=DURATIONS_TO_TEST
)

best_period   = bls.period_at_max_power.value
best_t0       = bls.transit_time_at_max_power.value
best_depth    = bls.depth_at_max_power.value
best_duration = bls.duration_at_max_power.value

print(f"[RESULT] BLS best-fit parameters:")
print(f"         Period   : {best_period:.6f} days")
print(f"         T0       : {best_t0:.4f} BTJD")
print(f"         Duration : {best_duration * 24:.2f} hours")

# ---------------------------------------------------------------------------
# SNR AND DEPTH CALCULATION (UN-DETRENDED)
# ---------------------------------------------------------------------------
print("\n[INFO] Computing transit SNR from un-detrended photometry...")

lc_folded = lc_combined.fold(period=best_period, epoch_time=best_t0)

phase_mask_transit = (np.abs(lc_folded.phase.value) < (best_duration * 0.55))
phase_mask_out     = (np.abs(lc_folded.phase.value) > (best_duration * 2.0))

flux_in  = lc_folded.flux[phase_mask_transit]
flux_out = lc_folded.flux[phase_mask_out]

if len(flux_in) > 5 and len(flux_out) > 5:
    median_flux_out     = np.nanmedian(flux_out)
    median_flux_in      = np.nanmedian(flux_in)
    depth_calc_natural  = median_flux_out - median_flux_in
    noise               = np.nanstd(flux_out)
    # SNR = (delta / sigma_oot) * sqrt(N_in)
    snr_final           = (depth_calc_natural / noise) * np.sqrt(len(flux_in))
    depth_ppm_natural   = depth_calc_natural * 1e6
    noise_ppm           = noise * 1e6
else:
    print("[WARNING] Insufficient in-transit cadences for reliable SNR estimate.")
    snr_final = depth_ppm_natural = depth_calc_natural = noise_ppm = 0.0

# ---------------------------------------------------------------------------
# GEOMETRIC RADIUS ESTIMATE
# Rp = sqrt(delta) * R_star  [first-order; neglects limb darkening,
# aperture dilution, and impact parameter degeneracy]
# ---------------------------------------------------------------------------
depth_frac = depth_ppm_natural * 1e-6
Rp_earth   = np.sqrt(depth_frac) * R_STAR_SOLAR * R_SUN_TO_EARTH
print(f"[RESULT] First-order geometric radius: Rp = {Rp_earth:.2f} R_earth")
print(f"         (assumes delta = {depth_ppm_natural:.0f} ppm, R_star = {R_STAR_SOLAR} R_sun)")

# ---------------------------------------------------------------------------
# SAVITZKY-GOLAY WINDOW SENSITIVITY ANALYSIS
# Tests depth recovery across a range of flatten() window lengths to
# justify the no-detrending approach (Kovacs et al. 2002).
# ---------------------------------------------------------------------------
print("\n[INFO] Running detrending sensitivity analysis...")
print(f"       Reference depth (un-detrended): {depth_ppm_natural:.0f} ppm")

window_lengths = [401, 1201, 1801, 2501, 3501]
depth_ppm_flat = 0.0

for wl in window_lengths:
    lc_test        = lc_combined.flatten(window_length=wl)
    lc_test_folded = lc_test.fold(period=best_period, epoch_time=best_t0)

    f_in  = lc_test_folded.flux[phase_mask_transit]
    f_out = lc_test_folded.flux[phase_mask_out]

    if len(f_in) > 0 and len(f_out) > 0:
        d_ppm = (np.nanmedian(f_out) - np.nanmedian(f_in)) * 1e6
        loss  = abs(depth_ppm_natural - d_ppm) / depth_ppm_natural * 100
        print(f"       WL: {wl:5d} cadences | Depth: {d_ppm:5.0f} ppm | "
              f"Loss vs natural: {loss:5.1f}%")
        if wl == 2501:
            depth_ppm_flat = d_ppm

depth_diff_percent = abs(depth_ppm_natural - depth_ppm_flat) / depth_ppm_natural * 100

# ---------------------------------------------------------------------------
# DIAGNOSTIC FIGURES
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

# Panel A: BLS Periodogram
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(bls.period, bls.power, color='black', lw=1)
ax1.axvline(best_period, color='red', alpha=0.5, lw=6,
            label=f'Peak: {best_period:.6f} d')
ax1.set_title(f"BLS Periodogram — {TOI_ID}", fontsize=13, fontweight='bold')
ax1.set_xlabel("Period [days]", fontsize=11)
ax1.set_ylabel("BLS Power", fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel B: Phase-folded transit (un-detrended)
ax2     = fig.add_subplot(gs[1, 0])
bin_size = 30 / (24 * 60)  # 30 minutes in days
lc_binned = lc_folded.bin(time_bin_size=bin_size)

lc_folded.scatter(ax=ax2, color='gray', alpha=0.3, s=2,
                  label='Un-detrended flux')
lc_binned.scatter(ax=ax2, color='red', s=25, marker='o',
                  label='30-min binned', zorder=5)
ax2.set_title(f"Phase-folded Transit — T0 = {best_t0:.4f} BTJD",
              fontsize=13, fontweight='bold')
ax2.set_xlabel("Phase [days]", fontsize=11)
ax2.set_ylabel("Normalized Flux", fontsize=11)
ax2.set_xlim(-0.25, 0.25)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# Panel C: Odd/Even transit vetting
ax3 = fig.add_subplot(gs[1, 1])
transit_number = np.round((lc_combined.time.value - best_t0) / best_period)
is_even = (transit_number % 2 == 0)

lc_even = lc_combined[is_even].fold(period=best_period, epoch_time=best_t0)
lc_odd  = lc_combined[~is_even].fold(period=best_period, epoch_time=best_t0)

if len(lc_even) > 10:
    lc_even.bin(time_bin_size=bin_size).plot(
        ax=ax3, color='blue', lw=2.5, label='Even transits')
if len(lc_odd) > 10:
    lc_odd.bin(time_bin_size=bin_size).plot(
        ax=ax3, color='orange', lw=2.5, linestyle='--', label='Odd transits')

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
print(f"    (sqrt(delta) * R_star; neglects LD, dilution, impact parameter)")
print("-" * 70)
if snr_final > 10:
    print("  Signal status: ROBUST (SNR > 10)")
elif snr_final > 7:
    print("  Signal status: MARGINAL (7 < SNR <= 10)")
else:
    print("  Signal status: WEAK (SNR <= 7) — interpret with caution")
print("=" * 70)

# ---------------------------------------------------------------------------
# DATA EXPORT
# ---------------------------------------------------------------------------
print("\n[INFO] Exporting phase-folded light curve data...")

df_binned = pd.DataFrame({
    'phase_days'  : lc_binned.time.value,
    'flux'        : lc_binned.flux.value,
    'flux_err'    : lc_binned.flux_err.value
}).sort_values('phase_days')

df_raw = pd.DataFrame({
    'phase_days'  : lc_folded.time.value,
    'flux'        : lc_folded.flux.value,
    'flux_err'    : lc_folded.flux_err.value
}).sort_values('phase_days')

fname_binned = f"{TOI_ID.replace(' ', '')}_Folded_Binned_RedPoints.csv"
fname_raw    = f"{TOI_ID.replace(' ', '')}_Folded_Raw_GrayPoints.csv"

df_binned.to_csv(fname_binned, index=False, header=False)
df_raw.to_csv(fname_raw,    index=False, header=False)

print(f"[INFO] Saved: {fname_binned} ({len(df_binned)} rows)")
print(f"[INFO] Saved: {fname_raw}    ({len(df_raw)} rows)")
print("[INFO] Pipeline complete.")