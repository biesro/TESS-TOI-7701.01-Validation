"""
02_centroid_TOI7701_4sec.py
=============================================================================
SUB-PIXEL CENTROID ANALYSIS: TOI 7701.01 (MULTI-SECTOR)
=============================================================================

Measures phase-folded sub-pixel centroid shifts during the transit of
TOI 7701.01 (TIC 122522333) to test for a background eclipsing binary (BEB)
false positive scenario. A genuine on-target transit produces no significant
centroid displacement; a BEB produces a systematic shift correlated with
transit phase.

Method:
    1. Extract pixel-level centroid time series from TESS TPF using
       flux-weighted moment estimation across all available sectors.
    2. Remove spacecraft drift per sector using a median filter (kernel = 301 
       cadences, ~10 hours at 120-second cadence). Kernel size chosen to 
       exceed the transit duration (7.08 hours) to avoid self-subtraction of 
       the transit centroid signal.
    3. Concatenate and phase-fold the residual (drift-corrected) centroid 
       shifts and bin at 30-minute resolution.
    4. Assess peak shift amplitude against a ±0.005-pixel threshold.

TARGET:
    TIC ID  : 122522333
    TOI     : 7701.01
    Sectors : Multi-Sector (SPOC/TESS-SPOC)
    Period  : 20.609766 days
    T0      : 1389.5507 BTJD

OUTPUT:
    - Two-panel figure: phase-folded X (column) and Y (row) centroid shifts.
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
KNOWN_PERIOD = 20.609766   # Orbital period in days
KNOWN_T0     = 1389.5507   # Transit epoch in BTJD

# Median filter kernel (cadences). Must exceed transit duration.
# Transit duration ~7.08 h (~212 cadences at 120s cadence).
MEDFILT_KERNEL = 631        # ~21.0 hours window

# Visualization phase window
PHASE_XLIM     = 0.5        # ± days window around T0

# Centroid displacement threshold for BEB suspicion (pixels)
CENTROID_THRESHOLD = 0.005  

# Binning resolution (30 minutes converted to days)
BIN_PHASE = 30 / (24 * 60)  

# Transit half-duration in days for in-transit masking (7.08 hours / 2)
TRANSIT_HALF_DURATION_DAYS = (7.08 / 2) / 24  

# ---------------------------------------------------------------------------
# TPF RETRIEVAL AND CENTROID EXTRACTION (MULTI-SECTOR)
# ---------------------------------------------------------------------------
print(f"[INFO] Initiating centroid analysis for TOI 7701.01 ({TIC_ID})")
print("-" * 70)

try:
    print(f"[INFO] Querying MAST for ALL available TESS Target Pixel Files...")
    # For centroids, we strictly require native TPF data (SPOC/TESS-SPOC)
    search = lk.search_targetpixelfile(TIC_ID, author=("SPOC", "TESS-SPOC"))
    
    if len(search) == 0:
        raise ValueError("No TPF found for the specified target.")

    print(f"[INFO] {len(search)} TPF(s) found. Processing centroids per sector...")

    all_time = []
    all_shift_x = []
    all_shift_y = []
    sectors_downloaded = []  # Deduplication memory

    # Iterate sector by sector to center coordinates INDIVIDUALLY
    for tpf_file in search:
        try:
            sector_num = tpf_file.mission[0]
            
            # Deduplication filter: prioritize first pipeline encountered per sector
            if sector_num in sectors_downloaded:
                continue
                
            print(f"[INFO] Downloading TPF for {sector_num}...")
            tpf = tpf_file.download()
            
            # Sector centroid calculation
            centroids = tpf.estimate_centroids(method='moments')
            x_raw = centroids[0].value
            y_raw = centroids[1].value
            time  = tpf.time.value
            
            # NaN filtering
            valid_mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
            time_clean = time[valid_mask]
            x_clean    = x_raw[valid_mask]
            y_clean    = y_raw[valid_mask]
            
            if len(x_clean) < MEDFILT_KERNEL:
                print(f"[WARNING] Not enough cadences in {sector_num} for median filter. Skipping.")
                continue
                
            # -----------------------------------------------------------------------
            # SYSTEMATIC DRIFT REMOVAL (Per sector)
            # -----------------------------------------------------------------------
            trend_x = medfilt(x_clean, kernel_size=MEDFILT_KERNEL)
            trend_y = medfilt(y_clean, kernel_size=MEDFILT_KERNEL)
            
            shift_x = x_clean - trend_x
            shift_y = y_clean - trend_y
            
            # Append relative results to global lists
            all_time.extend(time_clean)
            all_shift_x.extend(shift_x)
            all_shift_y.extend(shift_y)
            
            # Log processed sector to prevent duplication
            sectors_downloaded.append(sector_num)
            
            # Free memory
            del tpf
            
        except Exception as e:
            print(f"[WARNING] Failed to process {tpf_file.mission[0]}: {e}")
            continue

    if not all_time:
        raise ValueError("No valid centroid data extracted from any sector.")

    # Convert to master numpy arrays
    time_master = np.array(all_time)
    shift_x_master = np.array(all_shift_x)
    shift_y_master = np.array(all_shift_y)
    
    print(f"[INFO] All sectors processed and concatenated. Total cadences: {len(time_master)}")

    # ---------------------------------------------------------------------------
    # PHASE FOLDING AND BINNING
    # ---------------------------------------------------------------------------
    print("[INFO] Folding and binning centroid shift time-series...")

    lc_x = lk.LightCurve(time=time_master, flux=shift_x_master)
    lc_y = lk.LightCurve(time=time_master, flux=shift_y_master)

    folded_x = lc_x.fold(period=KNOWN_PERIOD, epoch_time=KNOWN_T0)
    folded_y = lc_y.fold(period=KNOWN_PERIOD, epoch_time=KNOWN_T0)

    bin_x = folded_x.bin(time_bin_size=BIN_PHASE)
    bin_y = folded_y.bin(time_bin_size=BIN_PHASE)

    # Statistical assessment of peak in-transit amplitude
    transit_mask_x = np.abs(bin_x.time.value) < TRANSIT_HALF_DURATION_DAYS
    transit_mask_y = np.abs(bin_y.time.value) < TRANSIT_HALF_DURATION_DAYS
    
    peak_x = np.nanmax(np.abs(bin_x.flux.value[transit_mask_x])) if transit_mask_x.any() else float('nan')
    peak_y = np.nanmax(np.abs(bin_y.flux.value[transit_mask_y])) if transit_mask_y.any() else float('nan')

    print(f"[RESULT] Maximum in-transit centroid displacement:")
    print(f"         Peak ΔX (columns): {peak_x:.4f} pixels")
    print(f"         Peak ΔY (rows)   : {peak_y:.4f} pixels")
    print(f"         Decision limit   : ±{CENTROID_THRESHOLD:.3f} pixels")

    if peak_x < CENTROID_THRESHOLD and peak_y < CENTROID_THRESHOLD:
        print("[RESULT] Centroid stability confirmed. Origin is consistent with the target star.")
    else:
        print("[WARNING] Significant centroid shift detected. Potential background eclipsing binary (BEB).")

    # ---------------------------------------------------------------------------
    # DIAGNOSTIC VISUALIZATION
    # ---------------------------------------------------------------------------
    print("[INFO] Rendering diagnostic figures...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel X: Column-axis centroid shift
    folded_x.scatter(ax=ax1, s=2, alpha=0.15, c='gray', label='Drift-corrected shift')
    bin_x.plot(ax=ax1, c='red', lw=2.5, label='30-min binned trend')
    ax1.axhline( CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85, label=f'±{CENTROID_THRESHOLD}" threshold')
    ax1.axhline(-CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85)
    ax1.axvline(0, color='white', ls='--', alpha=0.3, lw=1)
    
    # Highlight full transit duration window
    ax1.axvspan(-TRANSIT_HALF_DURATION_DAYS, TRANSIT_HALF_DURATION_DAYS, color='red', alpha=0.1, label='Transit Window')

    ax1.set_title(r'Centroid Shift — X ($\Delta$ Columns)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Phase [days]', fontsize=11)
    ax1.set_ylabel('Shift [pixels]', fontsize=11)
    ax1.set_xlim(-PHASE_XLIM, PHASE_XLIM)
    ax1.set_ylim(-0.02, 0.02)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel Y: Row-axis centroid shift
    folded_y.scatter(ax=ax2, s=2, alpha=0.15, c='gray')
    bin_y.plot(ax=ax2, c='steelblue', lw=2.5, label='30-min binned trend')
    ax2.axhline( CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85, label=f'±{CENTROID_THRESHOLD}" threshold')
    ax2.axhline(-CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85)
    ax2.axvline(0, color='white', ls='--', alpha=0.3, lw=1)
    
    # Highlight full transit duration window
    ax2.axvspan(-TRANSIT_HALF_DURATION_DAYS, TRANSIT_HALF_DURATION_DAYS, color='steelblue', alpha=0.1, label='Transit Window')

    ax2.set_title(r'Centroid Shift — Y ($\Delta$ Rows)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Phase [days]', fontsize=11)
    ax2.set_ylabel('Shift [pixels]', fontsize=11)
    ax2.set_xlim(-PHASE_XLIM, PHASE_XLIM)
    ax2.set_ylim(-0.02, 0.02)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Sub-pixel Centroid Analysis — TOI 7701.01 (Multi-Sector)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"[ERROR] Centroid analysis pipeline failure: {e}")
    import traceback
    traceback.print_exc()

print("[INFO] Analysis concluded.")