import torch
import pytest
from torch_tiltxcorr.utils import (
    calculate_cross_correlation, get_shift_from_correlation_image
)


def test_shift_detection():
    a = torch.zeros((10, 10))
    a[6, 6] = 1
    b = torch.zeros((10, 10))
    b[5, 5] = 1

    ccc = calculate_cross_correlation(a, a)
    shift = get_shift_from_correlation_image(ccc)
    assert shift.tolist() == [0.0, 0.0]

    ccc = calculate_cross_correlation(a, b)
    dy, dx = get_shift_from_correlation_image(ccc)
    assert dy == pytest.approx(1., 1e-3)
    assert dx == pytest.approx(1., 1e-3)

    b = torch.zeros((10, 10))
    b[5, 5] = 2.5
    b[4, 5] = 2.5

    ccc = calculate_cross_correlation(a, b)
    dy, dx = get_shift_from_correlation_image(ccc)
    assert dy == pytest.approx(1.5, 1e-2)
    assert dx == pytest.approx(1.0, 1e-2)
