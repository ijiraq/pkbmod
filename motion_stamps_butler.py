"""Motion-compensated stamp coadds for Butler skymap/tract/patch (CFHT-style pickles).

Reloads difference warps and builds per-detection cutouts by shifting each epoch
to account for linear motion (pixels/day × dmjd), then stacks with masked
mean and median — matching the behavior intended by ``make_stamps_CFHT.py``.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy import ma

logger = logging.getLogger(__name__)

DETECTION_LINE_RE = re.compile(
    r"snr:\s*(\S+)\s+flux:\s*(\S+)\s+x:\s*(\S+)\s+y:\s*(\S+)\s+x_v:\s*(\S+)\s+y_v:\s*(\S+)"
)

CUTOUT_NARROW = 21
CUTOUT_WIDE = 43


def parse_sns_detections_txt(path: str | Path) -> dict[str, np.ndarray]:
    """Parse ``sns_*_detections.txt`` lines written by :func:`stack.run`."""
    path = Path(path)
    rows: list[tuple[float, ...]] = []
    with open(path, encoding="utf-8") as han:
        for line in han:
            line = line.strip()
            if not line:
                continue
            m = DETECTION_LINE_RE.search(line)
            if not m:
                logger.warning("Skipping unparsable line: %s", line[:120])
                continue
            rows.append(tuple(map(float, m.groups())))
    if not rows:
        raise ValueError(f"No detections parsed from {path}")
    a = np.array(rows, dtype=np.float64)
    return {
        "snr": a[:, 0],
        "flux": a[:, 1],
        "x": a[:, 2],
        "y": a[:, 3],
        "x_v": a[:, 4],
        "y_v": a[:, 5],
    }


def _badflags_bitmask(flag_keys: Sequence[str], bitmask: dict[str, int]) -> int:
    flags = 0
    for name in flag_keys:
        if name not in bitmask:
            logger.warning(
                "Mask plane %r not in bitmask (available: %s)",
                name,
                sorted(bitmask.keys()),
            )
            continue
        flags |= 1 << int(bitmask[name])
    return flags


def _build_padded_cubes(
    datas: np.ndarray,
    masks: np.ndarray,
    badflags: int,
    pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad ``datas`` and build integer ``mask_data`` (1 = bad, 0 = good)."""
    n, h, w = datas.shape
    im_data = np.zeros((n, h + 2 * pad, w + 2 * pad), dtype=datas.dtype)
    mask_data = np.ones((n, h + 2 * pad, w + 2 * pad), dtype=np.int32)
    for i in range(n):
        data = datas[i]
        mask = masks[i]
        sl_y = slice(pad, pad + h)
        sl_x = slice(pad, pad + w)
        im_data[i, sl_y, sl_x] = data
        m = mask.astype(np.uint64)
        good = ((m & badflags) == 0) & np.isfinite(data)
        mask_data[i, sl_y, sl_x][good] = 0
        im_data[i, sl_y, sl_x][~good] = 0
    return im_data, mask_data


def _motion_coadd_one_cutout_size(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dmjds: np.ndarray,
    im_data: np.ndarray,
    mask_data: np.ndarray,
    pad: int,
    cutout_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_stamps, median_stamps) with shape (n_src, cutout, cutout)."""
    n_src = len(x)
    n_t = len(dmjds)
    hcos = cutout_size // 2
    times_rel = dmjds - dmjds[0]
    stamps_mean = np.full((n_src, cutout_size, cutout_size), 0, dtype=np.float64)
    stamps_med = np.full((n_src, cutout_size, cutout_size), 0, dtype=np.float64)

    cutout_data = np.zeros((n_t, cutout_size, cutout_size), dtype=im_data.dtype)
    cutout_mask = np.ones((n_t, cutout_size, cutout_size), dtype=bool)

    for i in range(n_src):
        cutout_mask.fill(True)
        for j in range(n_t):
            xc = x[i] + times_rel[j] * vx[i]
            yc = y[i] + times_rel[j] * vy[i]
            xi = int(np.floor(xc)) + pad
            yi = int(np.floor(yc)) + pad
            y0, y1 = yi - hcos, yi + hcos + 1
            x0, x1 = xi - hcos, xi + hcos + 1
            if y0 < 0 or x0 < 0 or y1 > im_data.shape[1] or x1 > im_data.shape[2]:
                continue
            cutout_data[j] = im_data[j, y0:y1, x0:x1]
            cutout_mask[j] = mask_data[j, y0:y1, x0:x1].astype(bool)

        cube = ma.masked_array(cutout_data, mask=cutout_mask)
        stamps_mean[i] = ma.mean(cube, axis=0).filled(0)
        stamps_med[i] = ma.median(cube, axis=0).filled(0)

    out_dtype = im_data.dtype
    return stamps_mean.astype(out_dtype, copy=False), stamps_med.astype(out_dtype, copy=False)


def run_motion_stamps(
    *,
    butler: str,
    collections: str,
    day_obs: int,
    skymap: str,
    tract: int,
    patch: int,
    band: str,
    dataset_type: str,
    instrument: str,
    psf_dataset_type: str,
    detections_path: str | Path,
    output_dir: str | Path | None,
    flag_keys: Sequence[str],
    variance_trim: float,
    data_dtype: np.dtype = np.float32,
    cutout_sizes: tuple[int, ...] = (CUTOUT_NARROW, CUTOUT_WIDE),
) -> dict[str, Path]:
    """Load warps, build motion-compensated stamps, write CFHT-style pickles.

    Returns a map label -> path written.
    """
    from butler_data_model import ButlerDataModel

    detections_path = Path(detections_path)
    dets = parse_sns_detections_txt(detections_path)
    dm = ButlerDataModel(
        butler=butler,
        collections=collections,
        day_obs=day_obs,
        skymap=skymap,
        tract=tract,
        patch=patch,
        band=band,
        instrument=instrument,
        dataset_type=dataset_type,
        data_dtype=data_dtype,
        psf_dataset_type=psf_dataset_type,
    )
    dm.mask_variance(variance_trim)
    dm.pack_inputs()
    si = dm.stack_inputs
    datas = si["datas"]
    masks = si["masks"]
    dmjds = np.asarray(si["dmjds"], dtype=np.float64)
    bitmask = si["bitmask"]
    del dm

    if not isinstance(bitmask, dict):
        raise TypeError("stack_inputs['bitmask'] must be dict")
    bad_int = _badflags_bitmask(flag_keys, bitmask)

    max_d = float(np.max(np.abs(dmjds))) if dmjds.size else 0.0
    speeds = np.hypot(dets["x_v"], dets["y_v"])
    max_v = float(np.max(speeds)) if speeds.size else 0.0
    max_hcos = max(cs // 2 for cs in cutout_sizes)
    pad = int(np.ceil(max_d * max_v)) + max_hcos + 100
    pad = max(pad, max_hcos + 50)

    im_data, mask_data = _build_padded_cubes(datas, masks, bad_int, pad)
    logger.info(
        "Padded cube shape %s pad=%d max_dmjd=%.4f max_speed=%.4f",
        im_data.shape,
        pad,
        max_d,
        max_v,
    )

    out_dir = Path(output_dir) if output_dir is not None else detections_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    base = f"stamps_tg_{day_obs}_{band}_{tract}_{patch}"

    for cs in cutout_sizes:
        mean_s, med_s = _motion_coadd_one_cutout_size(
            dets["x"],
            dets["y"],
            dets["x_v"],
            dets["y_v"],
            dmjds,
            im_data,
            mask_data,
            pad,
            cs,
        )
        if cs == CUTOUT_NARROW:
            mean_path = out_dir / f"{base}_sr.npy"
            med_path = out_dir / f"{base}_sr_med.npy"
        else:
            mean_path = out_dir / f"{base}_w_sr.npy"
            med_path = out_dir / f"{base}_w_sr_med.npy"

        np.save(mean_path, mean_s)
        written[f"mean_{cs}"] = mean_path
        logger.info("Wrote %s", mean_path)

        np.save(med_path, med_s)
        written[f"median_{cs}"] = med_path
        logger.info("Wrote %s", med_path)

    return written


def load_params_json(params_path: str | Path) -> dict[str, Any]:
    with open(params_path, encoding="utf-8") as han:
        return json.load(han)


def resolve_variance_trim_and_badflags(
    *,
    params_path: Path,
    variance_trim_cli: float | None,
    badflags_cli: list[str] | None,
    default_badflags: Sequence[str] | None = None,
) -> tuple[float, list[str]]:
    variance_trim = 1.3
    badflags: list[str] = []
    if params_path.is_file():
        p = load_params_json(params_path)
        variance_trim = float(p.get("variance_trim", variance_trim))
        bf = p.get("badflags")
        if isinstance(bf, list):
            badflags = [str(x) for x in bf]
    if badflags_cli:
        badflags = list(badflags_cli)
    elif not badflags and default_badflags:
        badflags = list(default_badflags)
    if variance_trim_cli is not None:
        variance_trim = float(variance_trim_cli)
    if not badflags:
        raise ValueError(
            "badflags empty: pass --flagkeys (before parse) / --badflags or ensure params.json has badflags"
        )
    return variance_trim, badflags
