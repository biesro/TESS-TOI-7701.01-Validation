"""
01.2_spatial_contamination_check_TOI7701.py
=============================================================================
SPATIAL CONTAMINATION ANALYSIS: TPF APERTURE + GAIA DR3 STAR MAP
=============================================================================

Performs an astrometric contamination check for TOI 7701.01 (TIC 122522333)
by combining TESS Target Pixel File (TPF) aperture visualization with a
Gaia DR3 field query. Identifies nearby sources that could contribute diluted
flux to the TESS aperture and mimic a planetary transit signal.

Diagnostics produced:
    1. TESS TPF pixel image with pipeline aperture mask overlay.
    2. Gaia DR3 star map: angular offsets (arcsec) from target, symbol size
       proportional to G-band flux, RUWE flagging for astrometric binaries.
       Reference circles mark 1 TESS pixel (21") and a typical aperture (10").

TARGET:
    TIC ID     : 122522333
    TOI        : 7701.01
    Coordinates: RA 02:34:19.93, Dec -31:50:54.38 (J2000)

OUTPUT:
    - Two-panel diagnostic figure (TPF image + Gaia star map)
    - Console report: 5 brightest Gaia sources within 1 arcmin,
      with separation, G magnitude, and RUWE.

DEPENDENCIES:
    lightkurve, matplotlib, astropy, astroquery, numpy

REFERENCES:
    Gaia Collaboration et al. (2023), A&A, 674, A1
    Stassun et al. (2019), AJ, 158, 138
"""

import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import numpy as np
import warnings

warnings.simplefilter('ignore')

# ---------------------------------------------------------------------------
# TARGET CONFIGURATION
# ---------------------------------------------------------------------------
# ICRS coordinates for TIC 122522333 / TOI 7701.01
target_coords = SkyCoord("02:34:19.93 -31:50:54.38", unit=(u.hourangle, u.deg))

TESS_PIXEL_ARCSEC  = 21.0   # TESS pixel scale [arcsec]
TYPICAL_AP_ARCSEC  = 10.0   # Representative SPOC aperture radius [arcsec]
GAIA_SEARCH_ARCMIN = 1.0    # Gaia query radius [arcmin]
RUWE_BINARY_FLAG   = 1.4    # RUWE threshold for astrometric binary suspicion

# ---------------------------------------------------------------------------
# TPF RETRIEVAL
# ---------------------------------------------------------------------------
print("[INFO] Querying MAST for TESS Target Pixel Files...")
search_result = lk.search_targetpixelfile(target_coords, radius=60)

if len(search_result) == 0:
    print("[ERROR] No TPF data found for this target at MAST.")
    exit()

print(f"[INFO] {len(search_result)} sector(s) found. Downloading first available...")
tpf = search_result[0].download()
print(f"[INFO] Sector: {tpf.sector if hasattr(tpf, 'sector') else 'N/A'} | "
      f"Mission: {tpf.mission}")

# ---------------------------------------------------------------------------
# FIGURE SETUP
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: TPF pixel image with pipeline aperture mask
tpf.plot(ax=ax[0], aperture_mask=tpf.pipeline_mask,
         title=f"TESS TPF — TIC 122522333 (Sector {tpf.sector})")

# Panel 2: Gaia DR3 star map
ax[1].set_facecolor('#0a0e27')
ax[1].grid(True, alpha=0.2, color='white', linestyle='--')
ax[1].set_xlabel('$\\Delta$RA (arcsec)', fontsize=12, color='white')
ax[1].set_ylabel('$\\Delta$Dec (arcsec)', fontsize=12, color='white')
ax[1].set_title('Gaia DR3 Field Map — TIC 122522333', fontsize=13,
                color='white', fontweight='bold', pad=12)
ax[1].tick_params(colors='white')

# ---------------------------------------------------------------------------
# GAIA DR3 QUERY AND STAR MAP
# ---------------------------------------------------------------------------
print(f"\n[INFO] Querying Gaia DR3 (Vizier I/355/gaiadr3) within "
      f"{GAIA_SEARCH_ARCMIN} arcmin...")

try:
    v = Vizier(columns=["*", "+_r"], row_limit=50)
    result = v.query_region(target_coords,
                            radius=GAIA_SEARCH_ARCMIN * u.arcmin,
                            catalog="I/355/gaiadr3")

    if not result or len(result) == 0:
        raise ValueError("No Gaia sources returned.")

    gaia_table = result[0]
    print(f"[INFO] {len(gaia_table)} Gaia DR3 sources retrieved.")

    # Compute angular offsets relative to target (cosine-corrected RA)
    offsets_ra  = []
    offsets_dec = []
    magnitudes  = []
    ruwes       = []

    for star in gaia_table:
        star_coord = SkyCoord(ra=star['RA_ICRS'] * u.deg,
                              dec=star['DE_ICRS'] * u.deg)
        d_ra  = ((star_coord.ra.deg - target_coords.ra.deg) * 3600
                 * np.cos(np.radians(target_coords.dec.deg)))
        d_dec = (star_coord.dec.deg - target_coords.dec.deg) * 3600
        offsets_ra.append(d_ra)
        offsets_dec.append(d_dec)
        magnitudes.append(star['Gmag'] if 'Gmag' in star.colnames else 15.0)
        ruwes.append(star['RUWE']      if 'RUWE' in star.colnames else 1.0)

    offsets_ra  = np.array(offsets_ra)
    offsets_dec = np.array(offsets_dec)
    magnitudes  = np.array(magnitudes)
    ruwes       = np.array(ruwes)

    # Symbol size proportional to flux (logarithmic in magnitude)
    sizes        = 1000 * 10 ** (-magnitudes / 5)
    colors_stars = plt.cm.plasma(
        (magnitudes - magnitudes.min()) /
        (magnitudes.max() - magnitudes.min() + 0.01)
    )

    # Plot each source
    target_idx = np.argmin(magnitudes)
    for i, (ra, dec, mag, size, color, ruwe) in enumerate(
            zip(offsets_ra, offsets_dec, magnitudes, sizes, colors_stars, ruwes)):

        if i == target_idx:
            ax[1].scatter(ra, dec, s=size * 2, c='gold', marker='*',
                          edgecolors='yellow', linewidths=3, zorder=100,
                          label=f'Target — G = {mag:.2f}')
            ax[1].text(ra, dec - 3, 'TIC 122522333', ha='center', va='top',
                       color='yellow', fontsize=9, fontweight='bold')
        else:
            edge_color = 'red' if ruwe > RUWE_BINARY_FLAG else 'white'
            ax[1].scatter(ra, dec, s=size, c=[color], marker='o',
                          edgecolors=edge_color, linewidths=1.5,
                          alpha=0.85, zorder=50)
            if mag < 16:
                ax[1].text(ra, dec + 2, f'G={mag:.1f}', ha='center',
                           va='bottom', color='white', fontsize=7, alpha=0.8)

    # Reference circles
    ax[1].add_patch(plt.Circle((0, 0), TESS_PIXEL_ARCSEC, color='red',
                               fill=False, lw=2.5, ls='--', alpha=0.75,
                               label=f'1 TESS pixel ({TESS_PIXEL_ARCSEC:.0f}")'))
    ax[1].add_patch(plt.Circle((0, 0), TYPICAL_AP_ARCSEC, color='lime',
                               fill=False, lw=1.8, ls=':', alpha=0.6,
                               label=f'Typical aperture (~{TYPICAL_AP_ARCSEC:.0f}")'))

    # Axis limits
    max_offset = max(np.abs(offsets_ra).max(), np.abs(offsets_dec).max()) * 1.2
    max_offset = max(max_offset, 35.0)
    ax[1].set_xlim(-max_offset, max_offset)
    ax[1].set_ylim(-max_offset, max_offset)
    ax[1].set_aspect('equal')
    ax[1].legend(loc='upper right', fontsize=8, facecolor='#0a0e27',
                 edgecolor='white', labelcolor='white')

    # -----------------------------------------------------------------------
    # CONSOLE REPORT: 5 brightest sources
    # -----------------------------------------------------------------------
    print("\n[RESULT] Nearest Gaia DR3 sources (sorted by G magnitude):")
    print(f"{'Rank':<5} {'G mag':>6} {'Sep (arcsec)':>13} {'RUWE':>6}  Note")
    print("-" * 55)

    sort_idx    = magnitudes.argsort()
    gaia_sorted = gaia_table[sort_idx[:5]]
    ra_sorted   = offsets_ra[sort_idx[:5]]
    dec_sorted  = offsets_dec[sort_idx[:5]]

    for i, (star, o_ra, o_dec) in enumerate(
            zip(gaia_sorted, ra_sorted, dec_sorted)):
        mag  = star['Gmag'] if 'Gmag' in star.colnames else float('nan')
        sep  = np.sqrt(o_ra**2 + o_dec**2)
        ruwe = star['RUWE'] if 'RUWE' in star.colnames else float('nan')
        note = ""
        if i == 0:
            note = "<-- target"
        elif sep < TESS_PIXEL_ARCSEC and mag < 15:
            note = "WITHIN 1 TESS PIXEL — potential contaminant"
        print(f"{i+1:<5} {mag:>6.2f} {sep:>13.1f} {ruwe:>6.2f}  {note}")

except Exception as e:
    print(f"[ERROR] Gaia query failed: {e}")
    ax[1].text(0.5, 0.5, f"Gaia query error:\n{str(e)}",
               ha='center', va='center', transform=ax[1].transAxes,
               color='red', fontsize=11)
    import traceback
    traceback.print_exc()

plt.tight_layout()
plt.show()
print("[INFO] Spatial contamination analysis complete.")