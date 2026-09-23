from arcrl.evaluation.scoring import reference_game_score, reference_level_score


def test_level_score_caps():
    assert reference_level_score(10, 10, True) == 100.0
    assert reference_level_score(20, 10, True) == 115.0
    assert reference_level_score(10, 20, True) == 25.0
    assert reference_level_score(10, 5, False) == 0.0


def test_game_score_weights_later_levels():
    s = reference_game_score([(True, 10, 10), (False, 50, 10)])
    assert abs(s - 100 / 3) < 1e-9
    assert reference_game_score([]) == 0.0
