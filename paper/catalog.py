# catalog.py
#
# Catalogue of X-ray QPO claims in AGN. Fixed, numbered order: to select a
# subset, pass the list of indices in the desired order to select_sources().
# The returned list follows the order of the given indices (not the canonical
# one), so that it also controls stacking/colour order in plots.
#
# Indices 0-11 are unchanged with respect to the thesis version, so existing
# scripts that use them (e.g. [9], [10]) keep working. New entries start at 12.
#
# mass_range = None   -> no independent mass estimate: use the generic AGN
#                        range (M_AGN_MIN, M_AGN_MAX) from setup.py.
# mass_range = (lo, hi) -> range covering ALL published independent estimates,
#                        including discordant ones (rule applied uniformly)
#                        (see 'mass_ref'). Estimates derived from the QPO
#                        itself are excluded.
#                        Since the 2026-11-14 revision no source has lo == hi:
#                        the two former point estimates (J1257, NGC 4945) are
#                        now bracketed by lower/upper limits. The lo == hi
#                        branch downstream (single contour; 'exploratory'
#                        +/-1 dex in pif_timescales) is therefore not triggered
#                        by the present catalogue.
#
# nu0 is the OBSERVED frequency (not corrected for redshift). The rest-frame
# frequency nu_rest = nu0 * (1 + z) is added below and used by the analysis.
#
# Extra fields (ignored by the analysis code):
#   z          redshift (used for nu_rest, see below)
#   qpo_ref    discovery paper of the QPO
#   mass_ref   origin of mass_range
#   notes      caveats (significance, re-analyses, classification)
#
# Mass ranges are quoted as in the cited works; values in comments were
# checked against the sources listed in the paper's catalogue table.
#
# Revision 2026-11-14 (entries marked 'REV'):
#   idx 2      Mrk 766      (5.75e6, 7.41e6) -> (1.3e6, 7.4e6)
#   idx 9      J1257        (2e6, 2e6)       -> (1.0e6, 5.9e6)
#   idx 10,11  1ES 1927+654 (6.5e5, 4.3e6)   -> (6.5e5, 5.1e6)
#   idx 12,13  1H 0707-495  (2.0e6, 7.0e7)   -> (2.0e6, 8.6e7)
#   idx 14     NGC 1365     (4e6, 1e7)       -> (2e6, 6.3e7)
#   idx 18     NGC 4945     (1.4e6, 1.4e6)   -> (7e5, 7.4e6)

CATALOG = [
    # idx  name                    nu0 [Hz]   mass_range [Msun]
    dict(idx=0, name="RE J1034+396", nu0=2.7e-4, mass_range=(4.4e5, 5.5e7), z=0.04244,
         qpo_ref="Gierlinski+2008; persistent in Alston+2014",
         mass_ref="Czerny+2016 compilation, QPO-based values excluded: "
                  "X-ray variance 4.4e5 to M-sigma (McConnell & Ma) 5.5e7; most 1e6-1e7",
         notes="Only QPO confirmed by GV12 and Zhang+2023 re-analyses."),
    dict(idx=1, name="MS 2254.9-3712", nu0=1.5e-4, mass_range=(4e6, 1e7), z=0.039,
         qpo_ref="Alston+2015",
         mass_ref="R_BLR-L: 4e6 (Grupe+2004,2010); M-sigma([OIII]): 1e7 (Shields+2003)",
         notes="[OIII] width is only a surrogate of sigma_*."),
    # REV: lower end from the original RM value and the X-ray scaling estimate
    dict(idx=2, name="Mrk 766", nu0=1.55e-4, mass_range=(1.3e6, 7.4e6), z=0.0129,
         qpo_ref="Zhang+2017",
         mass_ref="RM database value log M = 6.82 +0.05/-0.06 (Bentz & Katz 2015); "
                  "original RM 1.76 +1.56/-1.40 e6 (Bentz+2009); "
                  "X-ray spectral scaling 1.26 +1.00/-0.77 e6 (Giacche+2014)",
         notes="Lower end is the lowest central value; with its 1-sigma error it "
               "would be 3.6e5 (Bentz+2009)."),
    dict(idx=3, name="MCG-06-30-15", nu0=2.7e-4, mass_range=(1.2e6, 5.8e7), z=0.00775,
         qpo_ref="Gupta+2018 (3670 s)",
         mass_ref="RM (1.6+/-0.4)e6 (Bentz+2016); stellar dyn. (4.4+/-1.4)e7 (Das+2025)",
         notes="Das+2025 warn their value may be biased high (counter-rotating disc)."),
    dict(idx=4, name="ESO 113-G010 (a)", nu0=1.24e-4, mass_range=(4.07e6, 1.0e7), z=0.0257,
         qpo_ref="Zhang+2020", mass_ref="log M = 6.85 +0.15/-0.24 (Porquet+2007 via Cackett+2013)",
         notes="QPOs (a), (b) in 2:1 ratio."),
    dict(idx=5, name="ESO 113-G010 (b)", nu0=6.8e-5, mass_range=(4.07e6, 1.0e7), z=0.0257,
         qpo_ref="Zhang+2020", mass_ref="as idx 4", notes=""),
    dict(idx=6, name="J1347 (a)", nu0=1.17e-5, mass_range=(6.2e6, 2.8e7), z=0.045,
         qpo_ref="Carpano+2018 (Swift, 23.82 h)",
         mass_ref="log M = 6.99 +0.46/-0.20 (Carpano & Jin 2018)", notes="Model-dependent mass."),
    dict(idx=7, name="J1347 (b)", nu0=3.89e-6, mass_range=(6.2e6, 2.8e7), z=0.045,
         qpo_ref="Carpano+2018 (71.44 h)", mass_ref="as idx 6", notes=""),
    dict(idx=8, name="J1231", nu0=7.31e-5, mass_range=(3e4, 1e5), z=0.11871,
         qpo_ref="Lin+2013 (3.8 h)",
         mass_ref="slim disc (6+/-3)e4 (Cao+2025); sigma[OIII] ~1e5 (Ho+2012)",
         notes="Classified LF-QPO by Smith+2021; Cao+2025 discuss QPE-like nature."),
    # REV: point estimate -> bracket.
    #   lo = virial value minus 0.3 dex (10^6.0); 0.3 dex is the method
    #        uncertainty quoted by Liu+2021 for their multi-epoch values.
    #   hi = M-sigma (Kormendy & Ho 2013, Eq. 3) with sigma* = 81 km/s:
    #        log M = 8.490 + 4.377 log(81/200) = 6.77  ->  5.9e6.
    #        (McConnell & Ma 2013 gives 1.3e6, inside the range.)
    #   The M-sigma masses are computed here from the published sigma*, they
    #   are not quoted in the literature.
    dict(idx=9, name="J1257", nu0=3.3e-5, mass_range=(1.0e6, 5.9e6), z=0.02068,
         qpo_ref="Imbrogno+2026 (A&A 706, A359)",
         mass_ref="broad-Halpha virial, log M = 6.31, 6.35, 6.19, 6.33 at four epochs "
                  "(Liu+2021, Table 1), minus 0.3 dex method uncertainty; "
                  "M-sigma with sigma* = 81+/-4 km/s (Liu+2021, Table 2): "
                  "1.3e6 (McConnell & Ma 2013) to 5.9e6 (Kormendy & Ho 2013)",
         notes="Few cycles; authors also consider a QPE interpretation. "
               "Frequency range 33-50 muHz (20-30 ks): nu0 is the Lomb-Scargle "
               "peak (~30 ks); their sinusoid fit gives 25.4+/-0.4 ks (3.94e-5 Hz)."),
    # REV: mass_ref corrected (3.56e9 is the host STELLAR mass, not a BH mass);
    #      upper end raised to the M-sigma central value.
    #      lo = adopted M_BH-M_bulge value minus 1 sigma (1.38 - 0.73).
    #      Full 1-sigma envelope of the three estimates would be (2.7e5, 1.44e7).
    dict(idx=10, name="1ES 1927+654 (a)", nu0=2.44e-3, mass_range=(6.5e5, 5.1e6), z=0.01942,
         qpo_ref="Masterson+2025; Masterson+2026 (persistence)",
         mass_ref="host-galaxy estimates of Li+2022: M_BH-M_bulge 1.38 +1.57/-0.73 e6 "
                  "(adopted by them); M_BH-M_star 1.08 +3.21/-0.81 e6; "
                  "M_BH-sigma (gas sigma) 5.1 +9.3/-3.2 e6",
         notes="Changing-look AGN; frequency evolved 0.9-2.4 mHz in 2022-2024. "
               "Broad-line virial mass 1.9e7 (Trakhtenbrot+2019) excluded: the BLR "
               "is not virialised (Li+2022)."),
    dict(idx=11, name="1ES 1927+654 (b)", nu0=9.0e-4, mass_range=(6.5e5, 5.1e6), z=0.01942,
         qpo_ref="Masterson+2025 (lowest frequency of the 2022-2024 drift)",
         mass_ref="as idx 10", notes=""),
    # REV: upper end now includes the quoted error, 6.8e7 * (1 + 0.27)
    dict(idx=12, name="1H 0707-495 (a)", nu0=2.63e-4, mass_range=(2.0e6, 8.6e7), z=0.0411,
         qpo_ref="Pan+2016 (3800 s)",
         mass_ref="log M = 6.36 +0.24/-0.06 (Zhou & Wang 2005); <= 4e6 (Done & Jin 2016); "
                  "6.8e7 (1 +/- 0.27) X-ray scaling method (Titarchuk+2025)",
         notes="Range spans discordant estimates (same rule as MCG-06-30-15)."),
    dict(idx=13, name="1H 0707-495 (b)", nu0=1.2e-4, mass_range=(2.0e6, 8.6e7), z=0.0411,
         qpo_ref="Zhang+2018 (second QPO, same data)", mass_ref="as idx 12", notes=""),
    # REV: both ends widened with the estimates of Risaliti+2009.
    #      If the bulge-luminosity value is dropped (the authors attribute it
    #      to starburst light), hi = 2.0e7 (their M-sigma value, log M = 7.3).
    dict(idx=14, name="NGC 1365", nu0=2.19e-4, mass_range=(2e6, 6.3e7), z=0.0055,
         qpo_ref="Yan+2024 (A&A 691, A7; 4566 s, 3.6 sigma)",
         mass_ref="M-sigma and broad Paalpha (5-10)e6 (Fazeli+2019); "
                  "molecular-torus dynamics 4e6 (Combes+2019); "
                  "Risaliti+2009: Hbeta width ~2e6, M-sigma log M = 7.3+/-0.4, "
                  "bulge K luminosity log M = 7.8+/-0.4",
         notes="Gurpide & Middleton 2025 ('Mind the gaps'): QPO significant only "
               "at ~91% (~1.7 sigma); not supported."),
    dict(idx=15, name="NGC 5506", nu0=6.24e-5, mass_range=(4.0e6, 1.0e8), z=0.0062,
         qpo_ref="Zhang, Meng, Zhang & Dai 2025 (16024 s, 3.6-3.7 sigma)",
         mass_ref="10^(7.3+/-0.7) (Gofford+2015)",
         notes="Their own 10^(7.75+/-0.25) assumes an ADAF: not independent."),
    dict(idx=16, name="Mrk 142", nu0=7.045e-5, mass_range=(6.0e5, 3.9e6), z=0.045,
         qpo_ref="Zhong+2024 (RAA 24, 065015)",
         mass_ref="RM dyn. modelling log M = 6.23 +0.26/-0.45 (Li+2018); RM virial 3.4 +0.5/-1.2 e6 (Du+2014)", notes=""),
    dict(idx=17, name="NGC 4151", nu0=5.9e-4, mass_range=(2.5e6, 5.1e7), z=0.00326,
         qpo_ref="Yan+2025 (PRD 111, 043027; 5.91e-4 Chandra, 5.68e-4 XMM)",
         mass_ref="RM 3.6-4.6e7; gas dyn. 3.0e7; stellar dyn. 0.25-3e7 (Roberts+2021)",
         notes=""),
    # REV: point estimate -> bracket.
    #   lo = published lower limit from maser accelerations (Braatz+2003);
    #        the same value follows from M = 233 v^2 r with v = 100 km/s,
    #        r = 0.3 pc (low end of the +/-(100-150) km/s maser velocities).
    #   hi = enclosed mass within 6.5 pc, (4.4 + 3)e6 (Gaspar+2022): an upper
    #        limit, because the enclosed mass cannot decrease with radius.
    dict(idx=18, name="NGC 4945", nu0=2.74e-7, mass_range=(7e5, 7.4e6), z=0.00187,
         qpo_ref="Smith, Robles & Perlman 2020 (RXTE, ~42 d)",
         mass_ref="H2O maser ~1e6-1.4e6 within 0.3 pc (Greenhill+1997); "
                  "lower limit > 7e5 from maser accelerations (Braatz+2003); "
                  "upper limit: enclosed mass (4.4+/-3)e6 within 6.5 pc (Gaspar+2022)",
         notes="Gurpide & Middleton 2025 ('Mind the gaps'): QPO at ~98.7% "
               "(~2.5 sigma) in the selected segment, 39% over the full light "
               "curve (Smith+2020 claimed 10.2 sigma)."),
]

# Rest-frame QPO frequency. The model frequency nu_solid = g / M is emitted in
# the source frame, while nu0 is observed: cosmological time dilation gives
# nu_rest = nu0 (1 + z), hence P = M nu_rest. All the analysis scripts use
# 'nu_rest'; 'nu0' is kept as published (catalogue table).
for _source in CATALOG:
    _source['nu_rest'] = _source['nu0'] * (1.0 + _source['z'])
del _source

# Non-AGN or non-X-ray claims: NOT used by the analysis (kept for reference
# and for a possible table in the paper). Values from Smith+2021 Table 2 and
# Zhang+2025 Table 1.
EXTRA_CATALOG = [
    dict(name="ASASSN-14li", kind="TDE", nu0=7.7e-3, logM=(6.23, 0.35, 0.35)),
    dict(name="Sw J164449.3+573451", kind="TDE", nu0=5.01e-3, logM=(7.0, 0.30, 0.35)),
    dict(name="TON S 180", kind="EUV", nu0=5.56e-6, logM=(6.85, 0.5, 0.5)),
    dict(name="1H 0419-577", kind="EUV", nu0=2.0e-6, logM=(8.11, 0.5, 0.5)),
    dict(name="RX J0437.4-4711", kind="EUV", nu0=1.27e-5, logM=(7.77, 0.5, 0.5)),
]


def select_sources(indices):
    """
    Return the sources corresponding to `indices`, IN THE GIVEN ORDER (not the
    canonical one), so that both the subset and the plotting/colour order can
    be chosen.

    Example: select_sources([9, 0, 10]) -> [J1257, RE J1034+396, 1ES 1927+654 (a)]
    """
    by_idx = {s["idx"]: s for s in CATALOG}
    missing = [i for i in indices if i not in by_idx]
    if missing:
        raise ValueError(f"Indices not in the catalogue: {missing}")
    return [by_idx[i] for i in indices]