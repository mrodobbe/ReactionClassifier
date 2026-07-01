"""Self-contained tests (no external data paths) — CI friendly."""
import pytest

from reactionclassifier import (
    ClassificationResult,
    ReactionClassifier,
    load_granularity,
    load_taxonomy,
    name_for,
    tier_path,
)
from reactionclassifier.taxonomy import full_class_name


@pytest.fixture(scope="module")
def clf():
    return ReactionClassifier()


def test_taxonomy_loads():
    tax = load_taxonomy()
    assert len(tax) > 1000
    assert name_for("1.3.5.5")  # known class has a name


def test_tier_path():
    assert tier_path("1.3.5.5") == ["1.3", "1.3.5", "1.3.5.5"]
    assert tier_path(None) == []


def test_full_class_name_is_l3_plus():
    # built only from levels 3..5; tier-1/2 names are excluded
    tax = load_taxonomy()
    fn = full_class_name("1.3.5.5")
    assert fn and tax.get("1") not in fn.split(" | ") and tax.get("1.3") not in fn.split(" | ")
    assert tax["1.3.5"] in fn and tax["1.3.5.5"] in fn


def test_granularity_loads():
    g = load_granularity()
    assert "granularity_examples" in g and "granularity_examples_medchem" in g


def test_result_type_and_fields(clf):
    r = clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")
    assert isinstance(r, ClassificationResult)
    for attr in ("reaction_code", "reaction_name", "neural_code", "neural_name",
                 "confidence", "tier_path"):
        assert hasattr(r, attr)


def test_classify_amide_coupling(clf):
    r = clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")
    assert r.reaction_code is not None
    assert r.reaction_code.startswith("2.1.2")     # N-acylation to amide
    assert r.reaction_name
    assert r.tier_path[0] == "2.1"
    assert 0.0 <= r.confidence <= 1.0


def test_classify_snar(clf):
    r = clf.classify("OCC1CNC1.COc1cnc(Cl)cc1>>COc1cnc(N2CC(CO)C2)cc1")
    assert r.reaction_code is not None
    assert r.reaction_code.startswith("1.3")       # N-arylation with Ar-X


def test_unconfirmed_falls_back_to_neural(clf):
    r = clf.classify("[Na+].[Cl-]>>[Na+].[Cl-]")
    assert r.reaction_code is None and r.reaction_name is None
    assert r.neural_code is not None               # neural fallback always available
    assert r.neural_name == full_class_name(r.neural_code)


def test_invalid_input_does_not_crash(clf):
    r = clf.classify("not a reaction")
    assert r.reaction_code is None


def test_no_conflict_codes(clf):
    # aggregator "CONFLICT:" markers must never be emitted as labels
    codes = {code for recs in clf._by_prefix.values() for (code, _, _) in recs}
    assert not any(str(c).startswith("CONFLICT") for c in codes)
    r = clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")
    assert not str(r.reaction_code or "").startswith("CONFLICT")
    assert not str(r.neural_code or "").startswith("CONFLICT")
