"""Colour-grading building blocks for the ``render-color-grading`` exercise.

The module ships as a deliberately failing skeleton: every public function
below raises :class:`NotImplementedError`.  ``grading/lut.py`` is the only file
that has to be implemented, ``tests/test_lut.py`` is the frozen specification
and must not be edited.

Shared contract
---------------
* Colour channels are floats in the closed range ``[0.0, 1.0]``; every public
  result is clamped back into that range.
* A LUT has at least ``2`` samples per axis, i.e. ``size >= 2``.
* Numeric results are compared in the test-suite with an absolute tolerance of
  ``1e-6``.
* An image is ``list[list[tuple[float, float, float]]]``: row-major, all rows
  have the same length, every pixel is an ``(r, g, b)`` tuple.
* A LUT is ``list[list[list[tuple[float, float, float]]]]`` indexed
  ``lut[r][g][b]``; each axis is sampled uniformly and both ``0.0`` and
  ``1.0`` are grid points.
* Invalid input raises :class:`GradeError`.
"""

from __future__ import annotations

import math

__all__ = [
    "GradeError",
    "build_lut",
    "sample_lut",
    "apply_lut",
    "curve_map",
    "saturation",
    "contrast",
    "validate_lut",
    "grade_batch",
]

# Rec. 709 luma coefficients.
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


class GradeError(Exception):
    """Raised when an argument violates the documented grading contract."""


def _clamp(value: float) -> float:
    """Clamp *value* into ``[0.0, 1.0]``."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _is_real(value):
    """Return True for finite ints/floats (booleans excluded)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _check_cubic(lut):
    """Validate *lut* as a cube with at least two samples per axis.

    Returns the axis length.  Raises :class:`GradeError` for any structural
    problem (non-list axes, ragged dimensions, axis length below 2).
    """
    if not isinstance(lut, list) or len(lut) < 2:
        raise GradeError("LUT must be a cube with at least 2 samples per axis")
    size = len(lut)
    for row in lut:
        if not isinstance(row, list) or len(row) != size:
            raise GradeError("LUT is not cubic: every axis must have the same length")
        for column in row:
            if not isinstance(column, list) or len(column) != size:
                raise GradeError("LUT is not cubic: every axis must have the same length")
    return size


def build_lut(size, fn):
    """Sample *fn* on a uniform cubic grid and return a 3-D LUT.

    Parameters
    ----------
    size:
        Number of samples per axis, an ``int`` with ``size >= 2``.  Smaller
        sizes raise :class:`GradeError`.  Axis ``i`` is sampled at
        ``i / (size - 1)``, so ``i == 0`` maps to ``0.0`` and
        ``i == size - 1`` maps to ``1.0``; the ``1.0`` endpoint must be present.
    fn:
        Callable ``fn(r, g, b) -> (r, g, b)`` with channel arguments in
        ``[0.0, 1.0]``.  Its return value is stored as an ``(r, g, b)`` tuple
        of floats without further reordering.

    Returns
    -------
    list
        ``lut`` with ``len(lut) == size``, ``len(lut[r]) == size``,
        ``len(lut[r][g]) == size`` and ``lut[r][g][b]`` the sample for
        ``(r / (size - 1), g / (size - 1), b / (size - 1))``.
    """
    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        raise GradeError("LUT size must be an int >= 2")

    last = size - 1
    lut = []
    for ri in range(size):
        r = ri / last
        plane = []
        for gi in range(size):
            g = gi / last
            row = []
            for bi in range(size):
                b = bi / last
                row.append(tuple(float(channel) for channel in fn(r, g, b)))
            plane.append(row)
        lut.append(plane)
    return lut


def sample_lut(lut, r, g, b):
    """Trilinearly interpolate *lut* at the colour ``(r, g, b)``.

    ``r``, ``g`` and ``b`` are floats; coordinates outside ``[0.0, 1.0]`` are
    clamped to the nearest valid grid coordinate instead of raising or wrapping
    around.  The eight surrounding lattice points are blended with the exact
    fractional weights, so a LUT built from the identity function returns the
    requested coordinate back within ``1e-6``.

    Raises :class:`GradeError` when *lut* is not cubic or when its axis length
    is smaller than ``2`` (interpolation is undefined for a single sample).

    Returns
    -------
    tuple
        ``(r, g, b)`` floats clamped into ``[0.0, 1.0]``.
    """
    size = _check_cubic(lut)
    last = size - 1

    # Clamp coordinates first so out-of-range values never become negative or
    # oversized indices into the cube.
    coords = (_clamp(r), _clamp(g), _clamp(b))
    lowers = []
    uppers = []
    fracs = []
    for coord in coords:
        scaled = coord * last
        index_lo = int(math.floor(scaled))
        if index_lo > last - 1:
            index_lo = last - 1
        index_hi = index_lo + 1
        frac = scaled - index_lo
        lowers.append(index_lo)
        uppers.append(index_hi)
        fracs.append(frac)

    r0, g0, b0 = lowers
    r1, g1, b1 = uppers
    fr, fg, fb = fracs

    out = [0.0, 0.0, 0.0]
    for dr in (0, 1):
        wr = fr if dr == 1 else 1.0 - fr
        ri = r1 if dr == 1 else r0
        for dg in (0, 1):
            wg = fg if dg == 1 else 1.0 - fg
            gi = g1 if dg == 1 else g0
            for db in (0, 1):
                wb = fb if db == 1 else 1.0 - fb
                bi = b1 if db == 1 else b0
                weight = wr * wg * wb
                corner = lut[ri][gi][bi]
                for channel in range(3):
                    out[channel] += weight * corner[channel]

    return tuple(_clamp(channel) for channel in out)


def apply_lut(img, lut):
    """Look every pixel of *img* up in *lut*.

    The image keeps its shape: the result has the same number of rows and the
    same number of pixels per row, and it is built from freshly allocated row
    and pixel containers (the input is never mutated).  An empty image yields
    an empty list.  Out-of-range coefficients are handled by
    :func:`sample_lut`, and the resulting channels are clamped into
    ``[0.0, 1.0]``.

    Returns
    -------
    list
        ``list[list[tuple[float, float, float]]]`` of the same shape as *img*.
    """
    # Reject a non-cubic / single-sample LUT before touching the image.
    _check_cubic(lut)
    return [
        [sample_lut(lut, r, g, b) for (r, g, b) in row]
        for row in img
    ]


def curve_map(x, curve):
    """Map one channel value *x* through a ``lift``/``gamma``/``gain`` curve.

    ``curve`` is a mapping with optional keys ``"lift"`` (default ``0.0``),
    ``"gamma"`` (default ``1.0``) and ``"gain"`` (default ``1.0``).  The
    operation is, in order::

        base = clamp(clamp(x) * gain + lift)
        out  = clamp(base ** (1.0 / gamma))

    ``x`` and the result live in ``[0.0, 1.0]``.  ``gamma <= 0`` is invalid and
    raises :class:`GradeError` before any division happens.  The single
    intermediate clamp keeps a negative ``lift`` from feeding a negative base
    into the power, and the final clamp makes a large ``gain`` saturate at
    ``1.0`` instead of overshooting.

    Returns
    -------
    float
        The mapped channel value in ``[0.0, 1.0]``.
    """
    lift = curve.get("lift", 0.0)
    gamma = curve.get("gamma", 1.0)
    gain = curve.get("gain", 1.0)

    if not _is_real(gamma) or gamma <= 0.0:
        raise GradeError("gamma must be a positive number")

    base = _clamp(_clamp(x) * gain + lift)
    return _clamp(base ** (1.0 / gamma))


def saturation(img, amount):
    """Blend *img* towards its luminance by *amount*.

    Luminance uses the Rec. 709 weights
    ``0.2126 * r + 0.7152 * g + 0.0722 * b`` — not the arithmetic RGB mean, so
    a pure green pixel is much brighter than a pure red one.  Per channel::

        out = clamp(lum + amount * (channel - lum))

    ``amount == 0.0`` therefore produces a purely grey image (``r == g == b``)
    and ``amount == 1.0`` leaves the image untouched.  ``amount < 0`` is clamped
    to ``0.0`` (never an inversion), and the output is clamped into
    ``[0.0, 1.0]``.  Shapes and allocation behave like :func:`apply_lut`, and
    an empty image yields an empty list.

    Returns
    -------
    list
        A new image with the same shape as *img*.
    """
    if amount < 0.0:
        amount = 0.0

    result = []
    for row in img:
        out_row = []
        for (r, g, b) in row:
            lum = _LUMA_R * r + _LUMA_G * g + _LUMA_B * b
            out_row.append(
                tuple(
                    _clamp(lum + amount * (channel - lum))
                    for channel in (r, g, b)
                )
            )
        result.append(out_row)
    return result


def contrast(img, pivot, amount):
    """Scale every channel around *pivot* by *amount*.

    Per channel::

        out = clamp(pivot + (channel - pivot) * amount)

    *pivot* is clamped into ``[0.0, 1.0]`` first, so an out-of-range pivot can
    never push the whole image to one end.  The pivot itself is a fixed point
    for every *amount*, ``amount == 1.0`` is the identity, values above ``1.0``
    expand and values below ``1.0`` compress.  The output is clamped into
    ``[0.0, 1.0]``.  Shapes and allocation behave like :func:`apply_lut`, and an
    empty image yields an empty list.

    Returns
    -------
    list
        A new image with the same shape as *img*.
    """
    pivot = _clamp(pivot)

    result = []
    for row in img:
        out_row = []
        for pixel in row:
            out_row.append(
                tuple(
                    _clamp(pivot + (channel - pivot) * amount)
                    for channel in pixel
                )
            )
        result.append(out_row)
    return result


def validate_lut(lut, size):
    """Return a list of human readable problems found in *lut*.

    *size* is the expected number of samples per axis.  Checked problems:

    * *size* is not an ``int`` or is smaller than ``2``;
    * ``len(lut) != size`` (size mismatch);
    * some row or column has a length different from *size* (not a cube);
    * some entry is not a length-3 sequence of finite numbers;
    * some channel falls outside ``[0.0, 1.0]`` (value range violation).

    A valid LUT produces an empty list.  The messages are free-form strings but
    must name the kind of problem so the caller can print them.  Unlike
    :func:`sample_lut` this function never raises for bad data: it reports it.

    Returns
    -------
    list[str]
        One string per detected problem, in the order described above.
    """
    problems = []

    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        problems.append(
            f"invalid LUT size {size!r}: expected an int with size >= 2"
        )
        return problems

    if not isinstance(lut, list) or len(lut) != size:
        actual = len(lut) if isinstance(lut, list) else type(lut).__name__
        problems.append(
            f"size mismatch: LUT has {actual} planes, expected {size}"
        )
        return problems

    cubic = True
    for ri, plane in enumerate(lut):
        if not isinstance(plane, list) or len(plane) != size:
            problems.append(
                f"LUT is not cubic: plane {ri} has length "
                f"{len(plane) if isinstance(plane, list) else type(plane).__name__}, "
                f"expected {size}"
            )
            cubic = False
            continue
        for gi, column in enumerate(plane):
            if not isinstance(column, list) or len(column) != size:
                problems.append(
                    f"LUT is not cubic: column lut[{ri}][{gi}] has length "
                    f"{len(column) if isinstance(column, list) else type(column).__name__}, "
                    f"expected {size}"
                )
                cubic = False

    if not cubic:
        # Entry-level checks would index into malformed structures.
        return problems

    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                entry = lut[ri][gi][bi]
                if (
                    not isinstance(entry, (tuple, list))
                    or len(entry) != 3
                    or not all(_is_real(channel) for channel in entry)
                ):
                    problems.append(
                        f"invalid entry at lut[{ri}][{gi}][{bi}]: "
                        f"expected a length-3 sequence of finite numbers, got {entry!r}"
                    )
                    continue
                for channel_index, channel in enumerate(entry):
                    if channel < 0.0 or channel > 1.0:
                        problems.append(
                            f"value range violation at lut[{ri}][{gi}][{bi}] "
                            f"channel {channel_index}: {channel} is outside [0.0, 1.0]"
                        )

    return problems


def _validate_image(img):
    """Raise :class:`GradeError` if *img* violates the image contract."""
    if not isinstance(img, list):
        raise GradeError("image must be a list of rows")

    row_length = None
    for ri, row in enumerate(img):
        if not isinstance(row, list):
            raise GradeError(f"image row {ri} must be a list of pixels")
        if row_length is None:
            row_length = len(row)
        elif len(row) != row_length:
            raise GradeError(
                f"image row {ri} has length {len(row)}, expected {row_length}"
            )
        for pi, pixel in enumerate(row):
            if not isinstance(pixel, (tuple, list)) or len(pixel) != 3:
                raise GradeError(
                    f"pixel [{ri}][{pi}] must be a length-3 (r, g, b) sequence"
                )
            for channel in pixel:
                if not _is_real(channel) or channel < 0.0 or channel > 1.0:
                    raise GradeError(
                        f"pixel [{ri}][{pi}] has a channel outside finite [0.0, 1.0]: "
                        f"{channel!r}"
                    )


def grade_batch(images, lut, config):
    """Grade a whole batch of images, all or nothing.

    Every image is validated before any output is produced.  An image is valid
    when it is a list of equally long rows of ``(r, g, b)`` tuples whose
    channels are finite floats in ``[0.0, 1.0]``; as soon as one image violates
    that, :class:`GradeError` is raised and no partial result exists (the batch
    is rolled back, inputs are never mutated).

    *config* is a mapping with optional keys, applied to every image in this
    fixed order: ``"curve"`` (per channel through :func:`curve_map`),
    ``"saturation"`` (through :func:`saturation`), ``"contrast"`` together with
    ``"pivot"`` (through :func:`contrast`).  A missing or ``None`` key means
    "skip this step".  :func:`apply_lut` always runs first.  Each image gets
    its own output buffer, so two images in the same batch can never leak
    values into each other.

    Returns
    -------
    dict
        ``{"images": [...], "count": len(images), "lut_size": len(lut)}`` where
        ``images[i]`` is the graded version of input ``i`` in the original
        order and both ``images`` and ``count`` are empty/``0`` for an empty
        batch.
    """
    # Validate everything up front: no output exists until the whole batch is
    # known to be legal, so a failure rolls back cleanly.
    for img in images:
        _validate_image(img)
    _check_cubic(lut)

    curve = config.get("curve")
    sat_amount = config.get("saturation")
    contrast_amount = config.get("contrast")
    pivot = config.get("pivot")

    graded = []
    for img in images:
        # Freshly allocated buffers per image: nothing is shared across the
        # batch, and the caller's input is never mutated.
        out = apply_lut(img, lut)

        if curve is not None:
            out = [
                [
                    tuple(curve_map(channel, curve) for channel in pixel)
                    for pixel in row
                ]
                for row in out
            ]

        if sat_amount is not None:
            out = saturation(out, sat_amount)

        if contrast_amount is not None:
            pivot_value = 0.5 if pivot is None else pivot
            out = contrast(out, pivot_value, contrast_amount)

        graded.append(out)

    return {
        "images": graded,
        "count": len(graded),
        "lut_size": len(lut),
    }
