# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.mark.parametrize(
    "action,answer,kind,expected",
    [
        ("Click(0.2,0.3)", "[0.1,0.2,0.3,0.4]", "Click", 2),
        ("Click(0.1,0.3)", "[0.1,0.2,0.3,0.4]", "Click", 1),
        ("Click(0.2,0.3)", "[0.2,0.3]", "Click", 2),
        ("Click(0.8,0.8)", "[0.2,0.3]", "Click", 1),
        ("Input_text(hello)", "hello", "Input_text", 2),
        ("Scroll(down)", "up", "Scroll", 1),
        ("Open(Settings)", "Settings", "Open", 2),
        ("Home()", "", "Home", 2),
        ("Back()", "", "Home", 0),
        ("Click(nan,0.2)", "[0.2,0.3]", "Click", 0),
        ("Click(0.2,0.3)junk", "[0.2,0.3]", "Click", 0),
    ],
)
def test_reward_adapter(reward, action, answer, kind, expected):
    score = reward.compute_score(
        data_source="androidcontrol",
        solution_str=f"<action>{action}</action>",
        ground_truth=answer,
        extra_info={"action_type": kind},
    )
    assert score == pytest.approx(expected)


def test_coordinate_string_cannot_execute_code(reward, tmp_path):
    marker = tmp_path / "must_not_exist"
    answer = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    assert reward.score_action("<action>Click(0.1,0.2)</action>", answer, {"action_type": "Click"}) == 0
    assert not marker.exists()
