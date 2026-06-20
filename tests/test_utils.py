import numpy as np
import pandas as pd

from utils import clean_id_series, to_binary, add_post_flags, within_distance_threshold, make_interaction_column


def test_clean_id_series_removes_artificial_float_suffix():
    s = clean_id_series(pd.Series(["001.0", "abc", None, " 45 ", "nan"]))
    assert s.iloc[0] == "001"
    assert s.iloc[1] == "abc"
    assert pd.isna(s.iloc[2])
    assert s.iloc[3] == "45"
    assert pd.isna(s.iloc[4])


def test_to_binary_preserves_missing_and_uses_positive_rule():
    out = to_binary(pd.Series([-1, 0, 0.2, 2, None, "bad"]))
    assert out.iloc[0] == 0
    assert out.iloc[1] == 0
    assert out.iloc[2] == 1
    assert out.iloc[3] == 1
    assert np.isnan(out.iloc[4])
    assert np.isnan(out.iloc[5])


def test_add_post_flags_from_year_and_wave_fallback():
    by_year = add_post_flags(pd.DataFrame({"survey_year": [2008, 2010, 2014]}))
    assert by_year["post_2009"].tolist() == [0.0, 1.0, 1.0]
    assert by_year["post_2013"].tolist() == [0.0, 0.0, 1.0]

    by_wave = add_post_flags(pd.DataFrame({"wave": [1, 2, 4]}))
    assert by_wave["post_2009"].tolist() == [0.0, 1.0, 1.0]
    assert by_wave["post_2013"].tolist() == [0.0, 0.0, 1.0]


def test_within_distance_threshold():
    out = within_distance_threshold(pd.Series([0, 24.9, 25, 25.1, None]), 25)
    assert out.iloc[:4].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert np.isnan(out.iloc[4])


def test_make_interaction_column():
    df = pd.DataFrame({"treated": [0, 1, 1], "post": [0, 0, 1]})
    out, col = make_interaction_column(df, "treated", "post")
    assert col == "treated_X_post"
    assert out[col].tolist() == [0, 0, 1]
