from minicasp.utils import load_reaction_records_csv, extract_retro_template_smarts

recs = load_reaction_records_csv(
    "/home/gzbrown/minicasp/data/reactions/uspto_original.csv",
    rxn_col="rxn_smiles",
    limit=50,
    shuffle=False,
)

ok = 0
for i, r in enumerate(recs):
    smarts = extract_retro_template_smarts(r.rxn_smiles, radius=1, debug=(i < 3))
    if smarts:
        ok += 1
        if ok <= 3:
            print("EXAMPLE SMARTS:", smarts)

print("Templates extracted:", ok, "/", len(recs), "=", ok / len(recs))