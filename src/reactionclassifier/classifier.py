"""Hybrid-strict reaction classifier.

Pipeline: a Morgan difference--product (MDP) fingerprint MLP gate predicts a
class; the exact rr0rp1_ring0 templates in that class's tier-3 subtree are then
applied to the query reaction, and a label is returned only if one of them
reproduces the recorded product (otherwise the classifier abstains). This
preserves the determinism guarantee: a positive label always corresponds to a
template that actually fired.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, List, Optional, Tuple


@dataclass
class ClassificationResult:
    """Result of :meth:`ReactionClassifier.classify`.

    ``reaction_code``/``reaction_name`` are the **deterministically confirmed**
    classification: they are populated only when a template in the predicted
    class's subtree actually fired on the reaction. If they are ``None``, the
    deterministic layer could not confirm a class — fall back to
    ``neural_code``/``neural_name``, the (unconfirmed) neural-gate prediction,
    which is always available.

    Attributes
    ----------
    reaction_code : confirmed class code, or ``None`` if unconfirmed.
    reaction_name : confirmed class name, or ``None`` if unconfirmed.
    neural_code   : neural-gate predicted class (always present unless the
                    reaction could not be parsed/fingerprinted).
    neural_name   : name of ``neural_code``.
    confidence    : neural-gate softmax confidence in [0, 1].
    tier_path     : ancestor codes from tier 2 down to ``reaction_code``.
    """

    reaction_code: Optional[str]
    reaction_name: Optional[str]
    neural_code: Optional[str]
    neural_name: Optional[str]
    confidence: float = 0.0
    tier_path: List[str] = field(default_factory=list)

import torch
import torch.nn.functional as F

from ._match import match_first
from .features import FingerprintConfig, reaction_features
from .model import MLPClassifier
from .taxonomy import full_class_name, tier_path


def _data(name: str):
    return resources.files("reactionclassifier.data").joinpath(name)


def _prefix(code: str, depth: int) -> str:
    return ".".join(str(code).split(".")[:depth])


class ReactionClassifier:
    """Load the bundled gate + template library + taxonomy and classify reactions.

    Example
    -------
    >>> clf = ReactionClassifier()
    >>> clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")["name"]
    """

    def __init__(self, device: str = "cpu", subset_tier: int = 3):
        self.device = torch.device(device)
        self.subset_tier = int(subset_tier)

        meta = json.loads(_data("gate/meta.json").read_text(encoding="utf-8"))
        self.fp_meta = meta["fp"]
        m = meta["model"]
        label_map = json.loads(_data("gate/label_map.json").read_text(encoding="utf-8"))
        self.id_to_label = {int(k): str(v) for k, v in label_map.items()}

        nbits = int(self.fp_meta.get("nbits", 2048))
        in_dim = nbits * (
            int(bool(self.fp_meta.get("include_diff_fp", True)))
            + int(bool(self.fp_meta.get("include_prod_fp", True)))
            + int(bool(self.fp_meta.get("include_react_fp", False)))
        )
        self.model = MLPClassifier(
            in_dim=in_dim,
            num_classes=int(meta["num_classes"]),
            hidden_dim=int(m["hidden_dim"]),
            depth=int(m["depth"]),
            dropout=float(m["dropout"]),
            activation=str(m["activation"]),
        ).to(self.device)
        with _data("gate/model.pt").open("rb") as fh:
            self.model.load_state_dict(torch.load(fh, map_location=self.device))
        self.model.eval()

        # Aggregator "CONFLICT:" markers are ambiguous, non-taxonomy codes; drop
        # them so they can never be emitted as a label.
        conflict_ids = [i for i, lab in self.id_to_label.items() if lab.startswith("CONFLICT")]
        self._conflict_idx = (
            torch.tensor(conflict_ids, dtype=torch.long, device=self.device) if conflict_ids else None
        )

        # exact-template library, indexed by tier-N prefix for fast subsetting
        c2t: Dict[str, List[str]] = json.loads(_data("class_to_templates.json").read_text(encoding="utf-8"))
        self._by_prefix: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        for code, templates in c2t.items():
            if str(code).startswith("CONFLICT"):
                continue
            pref = _prefix(code, self.subset_tier)
            for t in templates:
                nreact = len(t.split(">>")[0].split("."))
                self._by_prefix[pref].append((code, t, nreact))

    # -- gate ---------------------------------------------------------------
    def _featurize(self, rxn: str):
        fp = FingerprintConfig(
            radius=int(self.fp_meta.get("radius", 2)),
            nbits=int(self.fp_meta.get("nbits", 2048)),
        )
        return reaction_features(
            rxn,
            fp=fp,
            include_diff_fp=bool(self.fp_meta.get("include_diff_fp", True)),
            include_prod_fp=bool(self.fp_meta.get("include_prod_fp", True)),
            include_react_fp=bool(self.fp_meta.get("include_react_fp", False)),
        )

    def gate(self, rxn: str) -> Tuple[Optional[str], float]:
        """Return the MDP-gate's (class_code, softmax_confidence)."""
        try:
            feat = self._featurize(rxn)
        except Exception:
            return None, 0.0
        if feat is None:
            return None, 0.0
        x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            if self._conflict_idx is not None:
                logits[:, self._conflict_idx] = float("-inf")   # never predict CONFLICT classes
            probs = F.softmax(logits, dim=-1)
        p, idx = probs.max(dim=1)
        return self.id_to_label.get(int(idx[0].item())), float(p[0].item())

    # -- full classify ------------------------------------------------------
    def classify(self, rxn: str) -> ClassificationResult:
        """Classify a reaction SMILES (``reactants>>products`` or
        ``reactants>reagents>products``).

        Returns a :class:`ClassificationResult`. ``reaction_code``/
        ``reaction_name`` are set when a template deterministically confirms the
        class; otherwise they are ``None`` and you can fall back to
        ``neural_code``/``neural_name`` (the unconfirmed neural-gate prediction)."""
        gate_code, conf = self.gate(rxn)
        code = None
        if gate_code:
            records = self._by_prefix.get(_prefix(gate_code, self.subset_tier), [])
            if records:
                code = match_first(rxn, records)
        return ClassificationResult(
            reaction_code=code,
            reaction_name=full_class_name(code),
            neural_code=gate_code,
            neural_name=full_class_name(gate_code),
            confidence=round(conf, 4),
            tier_path=tier_path(code),
        )
