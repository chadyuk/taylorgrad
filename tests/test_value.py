import pytest
from taylorgrad import Value


@pytest.fixture
def plus_values():
    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    c.backward()
    return {"a": a, "b": b, "c": c}


@pytest.fixture
def mul_values():
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)
    d = a * b + c
    d.backward()
    return {"a": a, "b": b, "c": c, "d": d}


def test_plus_values(plus_values):
    assert plus_values["a"].data == 1.0
    assert plus_values["b"].data == 2.0
    assert plus_values["c"].data == 3.0
    assert plus_values["a"].grad == 1.0
    assert plus_values["b"].grad == 1.0
    assert plus_values["c"].grad == 1.0
    assert plus_values["a"].delta_in == 1.00001
    assert plus_values["b"].delta_in == 2.00001
    assert plus_values["c"].delta_in == 3.00002
    assert plus_values["a"].delta_out == 3.00002


def test_mul_values(mul_values):
    assert mul_values["d"].data == 5.0
    assert mul_values["a"].grad == 2.0
    assert mul_values["b"].grad == 1.0
    assert mul_values["c"].grad == 1.0
    assert mul_values["d"].grad == 1.0
    assert mul_values["a"].delta_in == 1.00001
    assert mul_values["b"].delta_in == 2.00001
    assert mul_values["c"].delta_in == 3.00001
    assert mul_values["d"].delta_in == 5.0000400001
    assert mul_values["a"].delta_out == 5.0000400001
    assert mul_values["b"].delta_out == 5.0000400001
    assert mul_values["c"].delta_out == 5.0000400001
