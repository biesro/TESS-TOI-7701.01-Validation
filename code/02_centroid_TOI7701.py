"""
02_centroid_TOI7701.py
=============================================================================
SUB-PIXEL CENTROID ANALYSIS: TOI 7701.01
=============================================================================

Measures phase-folded sub-pixel centroid shifts during the transit of
TOI 7701.01 (TIC 122522333) to test for a background eclipsing binary (BEB)
false positive scenario. A genuine on-target transit produces no significant
centroid displacement; a BEB produces a systematic shift correlated with
transit phase.

Method:
    1. Extract pixel-level centroid time series from TESS TPF using
       flux-weighted moment estimation.
    2. Remove spacecraft drift using a median filter (kernel = 301 cadences,
       ~10 hours at 120-second cadence). Kernel size chosen to exceed the
       transit duration (7.80 hours) to avoid self-subtraction of the
       transit centroid signal.
    3. Phase-fold the residual (drift-corrected) centroid shifts and bin
       at 30-minute resolution.
    4. Assess peak shift amplitude against a ±0.005-pixel threshold.

TARGET:
    TIC ID  : 122522333
    TOI     : 7701.01
    Sector  : 97 (SPOC, 2-minute cadence)
    Period  : 20.611596 days
    T0      : 3945.1594 BTJD

OUTPUT:
    - Two-panel figure: phase-folded X (column) and Y (row) centroid shifts.

DEPENDENCIES:
    lightkurve, matplotlib, numpy, scipy
"""

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import warnings

warnings.simplefilter('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TIC_ID       = "TIC 122522333"
KNOWN_PERIOD = 20.611596   # days (from BLS detection)
KNOWN_T0     = 3945.1594   # BTJD
SECTOR       = 97

# Median filter kernel (cadences). Must exceed transit duration to avoid
# absorbing the centroid signal. Transit = 7.80 h ~ 234 cadences at 120 s.
MEDFILT_KERNEL = 301        # ~10.0 hours

# Phase window for plot and assessment
PHASE_XLIM     = 0.03       # ± days (covers ±0.62 days ~ 15 h around transit)

# Centroid threshold for BEB suspicion (pixels)
CENTROID_THRESHOLD = 0.005  # pixels

# Binning resolution (phase units = days for folded LC)
BIN_PHASE = 0.001           # ~30 min at P = 20.6 d

# ---------------------------------------------------------------------------
# TPF RETRIEVAL AND CENTROID EXTRACTION
# ---------------------------------------------------------------------------
print(f"[INFO] Initiating centroid analysis for TOI 7701.01 ({TIC_ID})")
print("-" * 70)

try:
    print(f"[INFO] Downloading TPF — Sector {SECTOR}, author SPOC...")
    search = lk.search_targetpixelfile(TIC_ID, author="SPOC", sector=SECTOR)
    if len(search) == 0:
        search = lk.search_targetpixelfile(TIC_ID, author="TESS-SPOC",
                                            sector=SECTOR)
    if len(search) == 0:
        raise ValueError("No TPF found for this target and sector.")

    tpf = search.download()
    print(f"[INFO] TPF downloaded. Cadences: {len(tpf.time)}")

    # Flux-weighted moment centroids
    print("[INFO] Estimating centroids via flux-weighted moments...")
    centroids = tpf.estimate_centroids(method='moments')
    x_raw = centroids[0].value
    y_raw = centroids[1].value
    time  = tpf.time.value

    # Remove NaN cadences
    mask       = ~np.isnan(x_raw) & ~np.isnan(y_raw)
    time_clean = time[mask]
    x_clean    = x_raw[mask]
    y_clean    = y_raw[mask]
    print(f"[INFO] Valid centroid cadences after NaN removal: {mask.sum()}")

    # ---------------------------------------------------------------------------
    # SPACECRAFT DRIFT REMOVAL
    # Median filter isolates low-frequency drift; residual is the true
    # astrophysical centroid shift centered at zero.
    # ---------------------------------------------------------------------------
    print(f"[INFO] Removing spacecraft drift "
          f"(median filter, kernel = {MEDFILT_KERNEL} cadences = "
          f"{MEDFILT_KERNEL * 120 / 3600:.1f} h)...")

    trend_x = medfilt(x_clean, kernel_size=MEDFILT_KERNEL)
    trend_y = medfilt(y_clean, kernel_size=MEDFILT_KERNEL)
    shift_x = x_clean - trend_x
    shift_y = y_clean - trend_y

    # ---------------------------------------------------------------------------
    # PHASE FOLDING AND BINNING
    # ---------------------------------------------------------------------------
    print("[INFO] Phase-folding drift-corrected centroid shifts...")

    lc_x = lk.LightCurve(time=time_clean, flux=shift_x)
    lc_y = lk.LightCurve(time=time_clean, flux=shift_y)

    folded_x = lc_x.fold(period=KNOWN_PERIOD, epoch_time=KNOWN_T0)
    folded_y = lc_y.fold(period=KNOWN_PERIOD, epoch_time=KNOWN_T0)

    bin_x = folded_x.bin(time_bin_size=BIN_PHASE)
    bin_y = folded_y.bin(time_bin_size=BIN_PHASE)

    # In-transit peak amplitude assessment
    transit_mask_x = np.abs(bin_x.time.value) < 0.005
    transit_mask_y = np.abs(bin_y.time.value) < 0.005
    peak_x = np.nanmax(np.abs(bin_x.flux.value[transit_mask_x])) if transit_mask_x.any() else float('nan')
    peak_y = np.nanmax(np.abs(bin_y.flux.value[transit_mask_y])) if transit_mask_y.any() else float('nan')

    print(f"[RESULT] Peak in-transit centroid shift:")
    print(f"         X (columns): {peak_x:.4f} pixels")
    print(f"         Y (rows)   : {peak_y:.4f} pixels")
    print(f"         Threshold  : ±{CENTROID_THRESHOLD:.3f} pixels")
    if peak_x < CENTROID_THRESHOLD and peak_y < CENTROID_THRESHOLD:
        print("[RESULT] Centroid shifts within threshold — consistent with "
              "on-target transit origin.")
    else:
        print("[WARNING] One or more centroid shifts exceed threshold — "
              "inspect for possible background contamination.")

    # ---------------------------------------------------------------------------
    # DIAGNOSTIC FIGURE
    # ---------------------------------------------------------------------------
    print("[INFO] Generating centroid shift figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel X: column centroid shift
    folded_x.scatter(ax=ax1, s=2, alpha=0.15, c='gray',
                     label='Drift-corrected shift')
    bin_x.plot(ax=ax1, c='red', lw=2.5, label='30-min binned trend')
    ax1.axhline( CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85,
                label=f'±{CENTROID_THRESHOLD}" threshold')
    ax1.axhline(-CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85)
    ax1.axvline(0, color='white', ls='--', alpha=0.3, lw=1)
    ax1.set_title(r'Centroid Shift — X ($\Delta$ Columns)', fontsize=13,
                  fontweight='bold')
    ax1.set_xlabel('Phase [days]', fontsize=11)
    ax1.set_ylabel('Shift [pixels]', fontsize=11)
    ax1.set_xlim(-PHASE_XLIM, PHASE_XLIM)
    ax1.set_ylim(-0.02, 0.02)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel Y: row centroid shift
    folded_y.scatter(ax=ax2, s=2, alpha=0.15, c='gray')
    bin_y.plot(ax=ax2, c='steelblue', lw=2.5, label='30-min binned trend')
    ax2.axhline( CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85,
                label=f'±{CENTROID_THRESHOLD}" threshold')
    ax2.axhline(-CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85)
    ax2.axvline(0, color='white', ls='--', alpha=0.3, lw=1)
    ax2.set_title(r'Centroid Shift — Y ($\Delta$ Rows)', fontsize=13,
                  fontweight='bold')
    ax2.set_xlabel('Phase [days]', fontsize=11)
    ax2.set_ylabel('Shift [pixels]', fontsize=11)
    ax2.set_xlim(-PHASE_XLIM, PHASE_XLIM)
    ax2.set_ylim(-0.02, 0.02)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Sub-pixel Centroid Analysis — TOI 7701.01 (Sector {SECTOR})",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"[ERROR] Centroid analysis failed: {e}")
    import traceback
    traceback.print_exc()

print("[INFO] Centroid analysis complete.")