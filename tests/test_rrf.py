from zhi.retriever.rrf import reciprocal_rank_fusion


def test_commutativity():
    a = reciprocal_rank_fusion([["x", "y"], ["y", "z"]])
    b = reciprocal_rank_fusion([["y", "z"], ["x", "y"]])
    assert [x[0] for x in a] == [x[0] for x in b]
    assert [round(x[1], 9) for x in a] == [round(x[1], 9) for x in b]


def test_higher_when_in_both_rankings():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    scores = dict(fused)
    assert scores["a"] > scores["b"]


def test_score_formula():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    d = dict(fused)
    assert abs(d["a"] - 1.0 / (60 + 0 + 1)) < 1e-12
    assert abs(d["b"] - 1.0 / (60 + 1 + 1)) < 1e-12
