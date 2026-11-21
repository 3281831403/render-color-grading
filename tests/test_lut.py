"""Red test-suite for the ``render-color-grading`` exercise.

Exactly 15 tests: one per documented defect plus five cross-cutting
combinations.  Every test calls a public stub from ``grading.lut`` and is
expected to fail with ``NotImplementedError`` until that stub is implemented.

Conventions
-----------
* Channels are floats in ``[0.0, 1.0]``.
* Images are ``list[list[tuple[float, float, float]]]``.
* LUTs are ``list[list[list[tuple[float, float, float]]]]``, ``lut[r][g][b]``.
* Comparisons use an absolute tolerance of ``1e-6``.
"""

import math

import pytest

from grading.lut import (
    GradeError,
    apply_lut,
    build_lut,
    contrast,
    curve_map,
    grade_batch,
    sample_lut,
    saturation,
    validate_lut,
)

TOL = 1e-6


def _identity_lut(size=3):
    """Build a LUT of *size* samples per axis that reproduces its input."""
    return build_lut(size, lambda r, g, b: (r, g, b))


def _assert_image_close(got, want, tol=TOL):
    """Assert two nested image structures are close channel by channel."""
    assert len(got) == len(want), (got, want)
    for row_got, row_want in zip(got, want):
        assert len(row_got) == len(row_want), (row_got, row_want)
        for pixel_got, pixel_want in zip(row_got, row_want):
            assert len(pixel_got) == len(pixel_want), (pixel_got, pixel_want)
            for channel_got, channel_want in zip(pixel_got, pixel_want):
                assert math.isclose(channel_got, channel_want, abs_tol=tol), (
                    got,
                    want,
                )


def test_build_lut_hits_both_grid_endpoints():
    """Defect 1: sampling with ``i / size`` drops the 1.0 endpoint."""
    lut = _identity_lut(4)

    assert len(lut) == 4
    assert all(len(row) == 4 for row in lut)
    assert all(len(column) == 4 for row in lut for column in row)

    assert lut[0][0][0] == pytest.approx((0.0, 0.0, 0.0), abs=TOL)
    assert lut[3][3][3] == pytest.approx((1.0, 1.0, 1.0), abs=TOL)
    assert lut[1][0][0] == pytest.approx((1.0 / 3.0, 0.0, 0.0), abs=TOL)
    assert lut[0][2][0] == pytest.approx((0.0, 2.0 / 3.0, 0.0), abs=TOL)


def test_sample_lut_interpolates_instead_of_snapping():
    """Defect 2: nearest-corner lookup turns gradients into bands."""
    lut = _identity_lut(3)

    assert sample_lut(lut, 0.5, 0.5, 0.5) == pytest.approx(
        (0.5, 0.5, 0.5), abs=TOL
    )
    assert sample_lut(lut, 0.31, 0.62, 0.87) == pytest.approx(
        (0.31, 0.62, 0.87), abs=TOL
    )

    samples = [sample_lut(lut, x, 0.0, 0.0)[0] for x in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert samples == sorted(samples)
    steps = [later - earlier for earlier, later in zip(samples, samples[1:])]
    assert max(steps) - min(steps) < TOL


def test_sample_lut_clamps_out_of_range_coordinates():
    """Defect 3: no clamp means negative indices wrap around the cube."""
    lut = _identity_lut(3)

    assert sample_lut(lut, -1.5, 0.0, 0.0) == pytest.approx(
        (0.0, 0.0, 0.0), abs=TOL
    )
    assert sample_lut(lut, 1.0, 2.5, 7.0) == pytest.approx(
        (1.0, 1.0, 1.0), abs=TOL
    )
    assert sample_lut(lut, 0.25, -0.75, 0.5) == pytest.approx(
        (0.25, 0.0, 0.5), abs=TOL
    )


def test_curve_map_rejects_non_positive_gamma():
    """Defect 4: ``gamma == 0`` reaches the division and blows up."""
    for gamma in (0.0, -0.5):
        with pytest.raises(GradeError):
            curve_map(0.5, {"lift": 0.0, "gamma": gamma, "gain": 1.0})

    assert curve_map(0.5, {"lift": 0.0, "gamma": 1.0, "gain": 1.0}) == (
        pytest.approx(0.5, abs=TOL)
    )
    assert curve_map(0.0, {"lift": 0.25, "gamma": 1.0, "gain": 1.0}) == (
        pytest.approx(0.25, abs=TOL)
    )


def test_saturation_uses_luminance_not_rgb_mean():
    """Defect 5: the RGB mean makes green as bright as red."""
    red_grey = saturation([[(1.0, 0.0, 0.0)]], 0.0)[0][0]
    green_grey = saturation([[(0.0, 1.0, 0.0)]], 0.0)[0][0]

    assert red_grey[0] == pytest.approx(red_grey[1], abs=TOL)
    assert red_grey[1] == pytest.approx(red_grey[2], abs=TOL)
    assert green_grey[0] > 0.70
    assert red_grey[0] < 0.25
    assert green_grey[0] > 3.0 * red_grey[0]

    lum = green_grey[0]
    mixed = saturation([[(0.0, 1.0, 0.0)]], 0.5)[0][0]
    assert mixed[0] == pytest.approx(0.5 * lum, abs=TOL)
    assert mixed[1] == pytest.approx(lum + 0.5 * (1.0 - lum), abs=TOL)
    assert mixed[2] == pytest.approx(0.5 * lum, abs=TOL)


def test_contrast_scales_around_the_given_pivot():
    """Defect 6: a hard-coded 0.5 pivot crushes the shadows."""
    img = [[(0.0, 0.0, 0.0), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4)]]
    out = contrast(img, 0.2, 2.0)

    assert out[0][0] == pytest.approx((0.0, 0.0, 0.0), abs=TOL)
    assert out[0][1] == pytest.approx((0.2, 0.2, 0.2), abs=TOL)
    assert out[0][2] == pytest.approx((0.4, 0.4, 0.4), abs=TOL)
    assert out[0][3] == pytest.approx((0.6, 0.6, 0.6), abs=TOL)

    clamped_pivot = contrast([[(0.0, 0.0, 0.0)]], 1.7, 0.5)
    assert clamped_pivot[0][0] == pytest.approx((0.5, 0.5, 0.5), abs=TOL)


def test_validate_lut_reports_size_cube_and_range_problems():
    """Defect 7: LUT dimensions are never checked before indexing."""
    good = _identity_lut(2)
    assert validate_lut(good, 2) == []

    mirrored = build_lut(2, lambda r, g, b: (b, g, r))
    valid_problems = validate_lut(mirrored, 2)
    assert valid_problems == []
    assert all(isinstance(problem, str) for problem in valid_problems)

    size_mismatch = validate_lut(good, 4)
    assert size_mismatch

    non_cubic = [
        [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]],
        [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]],
    ]
    assert validate_lut(non_cubic, 2)

    out_of_range = [
        [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]],
        [
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        ],
    ]
    range_problems = validate_lut(out_of_range, 2)
    assert range_problems
    assert any("range" in problem.lower() or "1.5" in problem for problem in range_problems)


def test_graded_output_is_clamped_to_unit_range():
    """Defect 8: unclamped highlights wrap around to black."""
    assert curve_map(1.0, {"lift": 0.0, "gamma": 1.0, "gain": 2.0}) == pytest.approx(
        1.0, abs=TOL
    )
    assert curve_map(0.0, {"lift": -0.5, "gamma": 1.0, "gain": 1.0}) == pytest.approx(
        0.0, abs=TOL
    )

    boosted = saturation([[(1.0, 0.0, 0.0)]], 3.0)[0][0]
    assert boosted == pytest.approx((1.0, 0.0, 0.0), abs=TOL)

    crushed = contrast([[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]], 0.5, 4.0)[0]
    assert crushed[0] == pytest.approx((0.0, 0.0, 0.0), abs=TOL)
    assert crushed[1] == pytest.approx((1.0, 1.0, 1.0), abs=TOL)


def test_grade_batch_does_not_share_one_output_buffer():
    """Defect 9: a shared buffer leaks one image's pixels into the next."""
    lut = _identity_lut(3)
    first = [[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]
    second = [[(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]]

    result = grade_batch([first, second], lut, {})
    images = result["images"]

    assert len(images) == 2
    assert result["count"] == 2
    _assert_image_close(images[0], first)
    _assert_image_close(images[1], second)
    assert images[0] is not images[1]
    assert images[0][0] is not images[1][0]

    _assert_image_close(first, [[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]])
    _assert_image_close(second, [[(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]])


def test_saturation_clamps_negative_amount():
    """Defect 10: a negative amount inverts the image instead of clamping."""
    base = [[(1.0, 0.0, 0.0), (0.2, 0.6, 0.9)]]

    neutral = saturation(base, 0.0)
    negative = saturation(base, -1.0)

    _assert_image_close(negative, neutral)
    for row in negative:
        for pixel in row:
            assert pixel[0] == pytest.approx(pixel[1], abs=TOL)
            assert pixel[1] == pytest.approx(pixel[2], abs=TOL)
            assert 0.0 <= pixel[0] <= 1.0


def test_identity_lut_leaves_image_unchanged():
    """An identity LUT must round-trip every channel within 1e-6."""
    img = [
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.25, 0.5, 0.75)],
        [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7), (0.5, 0.5, 0.5)],
    ]

    out = apply_lut(img, _identity_lut(5))

    assert len(out) == 2
    assert all(len(row) == 3 for row in out)
    _assert_image_close(out, img)
    assert out is not img
    assert out[0] is not img[0]


def test_saturation_zero_produces_grayscale():
    """``saturation=0`` yields r == g == b == luminance."""
    img = [[(0.8, 0.2, 0.4), (0.0, 1.0, 0.0)]]
    out = saturation(img, 0.0)

    for index, pixel in enumerate(out[0]):
        r, g, b = img[0][index]
        expected = 0.2126 * r + 0.7152 * g + 0.0722 * b
        assert pixel[0] == pytest.approx(expected, abs=TOL)
        assert pixel[1] == pytest.approx(expected, abs=TOL)
        assert pixel[2] == pytest.approx(expected, abs=TOL)


def test_single_sample_lut_is_rejected():
    """``size=1`` cannot be interpolated and must raise ``GradeError``."""
    tiny = [[[(0.5, 0.5, 0.5)]]]

    with pytest.raises(GradeError):
        sample_lut(tiny, 0.5, 0.5, 0.5)

    with pytest.raises(GradeError):
        build_lut(1, lambda r, g, b: (r, g, b))

    with pytest.raises(GradeError):
        apply_lut([[(0.1, 0.2, 0.3)]], tiny)


def test_empty_image_stays_empty():
    """Empty inputs are legal and produce empty outputs."""
    lut = _identity_lut(2)

    assert apply_lut([], lut) == []
    assert saturation([], 0.5) == []
    assert contrast([], 0.5, 1.0) == []
    assert validate_lut(lut, 2) == []

    batch = grade_batch([], lut, {})
    assert batch["images"] == []
    assert batch["count"] == 0


def test_grade_batch_preserves_input_order():
    """A fully successful batch keeps the original image order."""
    lut = _identity_lut(3)
    images = [
        [[(0.1, 0.1, 0.1), (0.1, 0.1, 0.1)]],
        [[(0.4, 0.4, 0.4), (0.4, 0.4, 0.4)]],
        [[(0.9, 0.9, 0.9), (0.9, 0.9, 0.9)]],
    ]

    result = grade_batch(images, lut, {})

    assert result["count"] == 3
    assert result["lut_size"] == 3
    out = result["images"]
    assert len(out) == 3

    levels = []
    for expected, actual in zip(images, out):
        _assert_image_close(actual, expected)
        levels.append(actual[0][0][0])

    assert levels == sorted(levels)
    assert levels == [pytest.approx(0.1, abs=TOL), pytest.approx(0.4, abs=TOL), pytest.approx(0.9, abs=TOL)]
