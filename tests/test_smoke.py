from finplan import smoke_test


def test_smoke() -> None:
    assert smoke_test() == "FinPlan is working!"
