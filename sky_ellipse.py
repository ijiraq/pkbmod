"""Sky/tangent-plane ellipse inclusion tests for RA/Dec (or lon/lat).

The implementation uses a **flat-sky / tangent plane** at the ellipse center. That
matches typical small-patch survey math and is accurate when ``a``, ``b`` are small
compared to a radian (often stated as “within a few degrees” for percent-level errors).

**Position angle** is in degrees **East of North**, measured from North toward East,
for the **major** semi-axis ``a``. The minor semi-axis ``b`` is perpendicular.

If your ellipse has the major axis along the minor parameter, swap ``a`` and ``b``
before calling.
"""
from __future__ import annotations

import numpy as np

__all__ = ["point_in_sky_ellipse", "point_in_sky_ellipse_batch", "tangent_plane_offsets"]


def _unwrap_ra_diff_rad(ra1: np.ndarray, ra0: float) -> np.ndarray:
    """Shortest signed ΔRA in radians, array or scalar ``ra1``."""
    d = (np.asarray(ra1, dtype=float) - ra0 + np.pi) % (2 * np.pi) - np.pi
    return d


def tangent_plane_offsets(
    ra0_deg: float,
    dec0_deg: float,
    ra1_deg: float | np.ndarray,
    dec1_deg: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """East- and North-pointing offsets in radians (small-angle tangent plane).

    ``dx`` is eastward (ΔRA × cos(dec0)); ``dy`` is northward (ΔDec).
    """
    r0 = np.radians(ra0_deg)
    d0 = np.radians(dec0_deg)
    r1 = np.radians(np.asarray(ra1_deg, dtype=float))
    d1 = np.radians(np.asarray(dec1_deg, dtype=float))
    dx = _unwrap_ra_diff_rad(r1, r0) * np.cos(d0)
    dy = d1 - d0
    return dx, dy


def point_in_sky_ellipse(
    ra: float,
    dec: float,
    a: float,
    b: float,
    position_angle_deg: float,
    x1: float,
    y1: float,
    *,
    degrees: bool = True,
) -> bool:
    """Return True if ``(x1, y1)`` lies inside the ellipse centered at ``(ra, dec)``.

    Parameters
    ----------
    ra, dec
        Ellipse center (e.g. ICRS RA/Dec in degrees if ``degrees`` is True).
    a, b
        Major and minor **semi-axes** in angular units (degrees if ``degrees``,
        else radians). ``a`` is along ``position_angle_deg``.
    position_angle_deg
        East of North (degrees), direction of the **major** axis ``a``.
    x1, y1
        Test point, same coordinate system as ``ra``/``dec``.
    degrees
        If True, ``ra``, ``dec``, ``a``, ``b``, ``position_angle_deg``, ``x1``, ``y1``
        are all in degrees.

    Notes
    -----
    Tangent-plane approximation at ``(ra, dec)``. For very large ellipses, use a
    dedicated spherical polygon / spherical ellipse library.
    """
    if a <= 0 or b <= 0:
        return False

    if degrees:
        ra0 = np.radians(ra)
        dec0 = np.radians(dec)
        ra1 = np.radians(x1)
        dec1 = np.radians(y1)
        a_ = np.radians(a)
        b_ = np.radians(b)
        pa = np.radians(position_angle_deg)
    else:
        ra0, dec0, ra1, dec1 = ra, dec, x1, y1
        a_, b_, pa = a, b, position_angle_deg

    dx = float(_unwrap_ra_diff_rad(ra1, ra0) * np.cos(dec0))
    dy = float(dec1 - dec0)

    sin_pa = np.sin(pa)
    cos_pa = np.cos(pa)
    # u along major axis (PA E of N), v along minor
    u = dx * sin_pa + dy * cos_pa
    v = -dx * cos_pa + dy * sin_pa

    return (u / a_) ** 2 + (v / b_) ** 2 <= 1.0


def point_in_sky_ellipse_batch(
    ra: float,
    dec: float,
    a: float,
    b: float,
    position_angle_deg: float,
    ra1: np.ndarray,
    dec1: np.ndarray,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """Vectorized :func:`point_in_sky_ellipse` for arrays ``ra1``, ``dec1``."""
    if a <= 0 or b <= 0:
        return np.zeros(np.shape(ra1), dtype=bool)

    ra1 = np.asarray(ra1, dtype=float)
    dec1 = np.asarray(dec1, dtype=float)

    if degrees:
        ra0 = np.radians(ra)
        dec0 = np.radians(dec)
        r1 = np.radians(ra1)
        d1 = np.radians(dec1)
        a_ = np.radians(a)
        b_ = np.radians(b)
        pa = np.radians(position_angle_deg)
    else:
        ra0, dec0 = ra, dec
        r1, d1 = ra1, dec1
        a_, b_, pa = a, b, position_angle_deg

    dx = _unwrap_ra_diff_rad(r1, ra0) * np.cos(dec0)
    dy = d1 - dec0
    sin_pa = np.sin(pa)
    cos_pa = np.cos(pa)
    u = dx * sin_pa + dy * cos_pa
    v = -dx * cos_pa + dy * sin_pa
    return (u / a_) ** 2 + (v / b_) ** 2 <= 1.0
