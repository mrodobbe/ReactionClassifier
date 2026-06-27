"""Minimal usage example."""
from reactionclassifier import ReactionClassifier

clf = ReactionClassifier()

reactions = [
    "CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1",             # amide coupling
    "OCC1CNC1.COc1cnc(Cl)cc1>>COc1cnc(N2CC(CO)C2)cc1",  # SNAr / N-arylation
]
for rxn in reactions:
    r = clf.classify(rxn)
    print(rxn)
    if r.reaction_code:                     # deterministically confirmed
        print(f"  -> {r.reaction_code}  {r.reaction_name}  (conf {r.confidence})")
        print(f"     tiers: {r.tier_path}")
    else:                                   # fall back to the neural prediction
        print(f"  -> unconfirmed; neural guess: {r.neural_code}  {r.neural_name}")
    print()
