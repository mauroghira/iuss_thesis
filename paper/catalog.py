# catalog.py
#
# Catalogo delle osservazioni AGN (Sez. 1.3 della tesi). Ordine fisso e
# numerato: per scegliere un sottoinsieme da visualizzare, passare la
# lista di indici nell'ordine desiderato a select_sources(). L'ordine
# della lista restituita segue l'ordine degli indici forniti (non quello
# canonico), cosi' da poter controllare anche l'ordine di stacking /
# colore nei plot.
#
# mass_range = None  -> nessuna stima indipendente di massa: nei pannelli
#   dove serve un prior di massa (proiezione (a,R)) si usa il range AGN
#   generico (M_AGN_MIN, M_AGN_MAX) da setup.py.
# mass_range = (lo, hi) -> stima riportata nel testo della tesi. Dove
#   lo == hi (stima puntuale, es. J1257, 2XMM J123103.2) va gestita a
#   valle come contorno di livello singolo, non come banda piena.

CATALOG = [
    # idx  name                    nu0 [Hz]   mass_range [Msun]
    #
    # Masse aggiornate da letteratura/web.
    # Quando esistono stime indipendenti fortemente discordanti, il range
    # copre esplicitamente le stime pubblicate.
    dict(idx=0,  name="RE J1034+396",      nu0=2.7e-4,  mass_range=(1e6, 4e6)),
    dict(idx=1,  name="MS 2254.9-3712",    nu0=1.5e-4,  mass_range=(4e6, 1e7)),
    dict(idx=2,  name="Mrk 766",           nu0=1.55e-4, mass_range=(5.75e6, 7.41e6)),
    dict(idx=3,  name="MCG-06-30-15",      nu0=2.7e-4,  mass_range=(1.2e6, 5.8e7)),
    dict(idx=4,  name="ESO 113-G010 (a)",  nu0=1.24e-4, mass_range=(4.07e6, 1.0e7)),
    dict(idx=5,  name="ESO 113-G010 (b)",  nu0=6.8e-5,  mass_range=(4.07e6, 1.0e7)),
    dict(idx=6,  name="XMMU J134736 (a)",  nu0=1.17e-5, mass_range=(9.8e6, 9.8e6)),
    dict(idx=7,  name="XMMU J134736 (b)",  nu0=3.89e-6,  mass_range=(9.8e6, 9.8e6)),
    dict(idx=8,  name="2XMM J123103.2",    nu0=7.31e-5, mass_range=(3e4, 9e4)),
    dict(idx=9,  name="NGC 4945",          nu0=2.76e-7, mass_range=(1.4e6, 1.4e6)),
    dict(idx=10, name="J1257",             nu0=3.3e-5,  mass_range=(2e6, 2e6)),
    dict(idx=11, name="1ES 1927+654",      nu0=2.5e-3,  mass_range=(6.5e5, 4.3e6)),
]

# Note sulle stime:
# - RE J1034+396: intervallo più probabile ~1--4e6 Msun.
# - MS 2254.9-3712: ~4e6 Msun da R_BLR-L_5100; ~1e7 Msun da M_BH-sigma.
# - Mrk 766: log10(M/Msun)=6.82 (+0.05/-0.06).
# - MCG-06-30-15: 1.6(+/-0.4)e6 Msun da reverberation mapping; un lavoro
#   dinamico del 2025 trova (4.4+/-1.4)e7 Msun ma segnala possibili bias
#   sistematici. Il range copre entrambe le stime.
# - ESO 113-G010: log10(M/Msun)=6.85 (+0.15/-0.24).
# - XMMU J134736.6+173403: 9.8e6 Msun da broad-band SED fitting;
#   stima esplicitamente model-dependent.
# - 2XMM J123103.2+110648: (6+/-3)e4 Msun da slim-disc fitting (2025).
# - NGC 4945: ~1.4e6 Msun da dinamica del disco H2O maser.
# - J1257 = 2MASX J12571076+2724177: log10(M_BH/Msun)~6.3.
# - 1ES 1927+654: stime recenti ~1.08e6, 1.38e6 e 3.56e6 Msun,
#   con incertezze ampie; il range copre le estremità delle stime pubblicate.



def select_sources(indices):
    """
    Restituisce la lista di sorgenti corrispondenti a `indices`, NELL'ORDINE
    fornito (non nell'ordine canonico del catalogo). Permette di scegliere
    sia il sottoinsieme sia l'ordine di plotting/colorazione.

    Esempio: select_sources([10, 0, 11]) -> [J1257, RE J1034+396, 1ES 1927+654]
    """
    by_idx = {s["idx"]: s for s in CATALOG}
    missing = [i for i in indices if i not in by_idx]
    if missing:
        raise ValueError(f"Indici non presenti nel catalogo: {missing}")
    return [by_idx[i] for i in indices]