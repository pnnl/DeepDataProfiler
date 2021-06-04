import pytest
import deep_data_profiler as ddp


def test_jaccard(profile_example):
    pe = profile_example
    p1 = pe.p1
    p2 = pe.p2
    assert ddp.jaccard(p1, p2) - 0.5714285 < 10e-5


def test_avg_jaccard(profile_example):
    pe = profile_example
    p1 = pe.p1
    p2 = pe.p2
    assert ddp.avg_jaccard(p1, p2) == 0.75


def test_instance_jaccard(profile_example):
    pe = profile_example
    p1 = pe.p1
    p2 = pe.p2
    p3 = pe.p3
    assert ddp.instance_jaccard(p1, p1 + p2) == 1
    assert ddp.instance_jaccard(p3, p1 + p2) == 0.75
