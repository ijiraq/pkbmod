"""Sky coordinate frame for detection x/y in numpy image pixel coordinates.

Detection rows use pixel positions in the same frame as ``stack_inputs['datas'][0]``
(and as plant ``x0``/``y0``): integer indices into the 2D numpy arrays (local origin).

For LSST Butler warps, ``parent_origin_xy`` is ``(getX0(), getY0())`` of the reference
exposure; parent pixel = local + origin for :meth:`lsst.afw.geom.SkyWcs.pixelToSky`.

For FITS-based :class:`~data_models.ExtractedDataModel`, ``parent_origin_xy`` is
``(0, 0)`` and :class:`astropy.wcs.WCS` matches ``all_world2pix`` / numpy indices.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def detection_xy_to_radec_deg(
    frame: Mapping[str, Any] | None, x: float, y: float
) -> tuple[float | None, float | None]:
    """Return ICRS RA/Dec in degrees for detection pixel (x, y), or (None, None)."""
    if frame is None:
        return None, None
    ox, oy = frame["parent_origin_xy"]
    lsst = frame.get("lsst_sky_wcs")
    if lsst is not None:
        from lsst.geom import Point2D

        sp = lsst.pixelToSky(Point2D(float(x) + float(ox), float(y) + float(oy)))
        return sp.getRa().asDegrees(), sp.getDec().asDegrees()
    aw = frame.get("wcs_astropy")
    if aw is not None:
        sky = aw.pixel_to_world(float(x), float(y))
        return float(sky.ra.deg), float(sky.dec.deg)
    return None, None


def write_detection_frame_sidecar(path: str | Path, frame: Mapping[str, Any]) -> None:
    """Write JSON with serializable parts of ``detection_frame`` (no WCS objects)."""
    path = Path(path)
    out: dict[str, Any] = {
        "source": frame.get("source", "unknown"),
        "parent_origin_xy": list(frame["parent_origin_xy"]),
    }
    if "reference_visit" in frame:
        out["reference_visit"] = frame["reference_visit"]
    if "reference_data_id" in frame:
        out["reference_data_id"] = _json_safe(frame["reference_data_id"])
    if "dataset_type" in frame:
        out["dataset_type"] = frame["dataset_type"]
    if frame.get("fits_header_text"):
        out["fits_header_text"] = frame["fits_header_text"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as han:
        json.dump(out, han, indent=2)
    logger.info("Wrote detection frame metadata to %s", path)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
