from __future__ import annotations

from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.table import vstack
from astropy.time import Time
from lsst.daf.butler import DatasetRef
from lsst.source.injection.utils import sso


def get_injected_source_catalog(
    day_obs: int,
    skymap: str,
    tract: int,
    patch: int,
    butler,
    collections: str | Sequence[str],
    instrument: str = "HSC",
    injection_catalog_dataset_type: str = "injection_catalog",
    warp_refs: Sequence[DatasetRef] | None = None,
    warp_dataset_type: str = "injected_diff_directWarp",
):
    """Load and merge injected-source catalogs from the Butler.

    Orbital-element tables are propagated to the first and last warp visits,
    projected into each visit's pixel frame, and used to derive average motion
    rates in pixels per day.

    Returns:
        astropy.table.Table: ``plant_id``, ``ra``, ``dec``, ``x0``, ``y0``,
        ``rate_x``, ``rate_y``, ``mag`` (``ra``/``dec`` from the initial epoch).
    """
    data_id = {
        "skymap": skymap,
        "tract": tract,
        "patch": patch,
        "instrument": instrument,
    }
    injected_catalog_refs = butler.query_datasets(
        injection_catalog_dataset_type,
        collections=collections,
        data_id=data_id,
        find_first=False,
    )
    if not injected_catalog_refs:
        raise ValueError(
            f"No datasets of type {injection_catalog_dataset_type!r} for "
            f"collections={collections!r} data_id={data_id!r}"
        )

    injection_catalog = vstack([butler.get(ref) for ref in injected_catalog_refs])

    if warp_refs is None:
        warp_data_id = dict(data_id)
        warp_data_id["day_obs"] = day_obs
        warp_refs = butler.query_datasets(
            warp_dataset_type,
            collections=collections,
            data_id=warp_data_id,
            order_by="visit",
        )
    if len(warp_refs) < 1:
        raise ValueError(
            f"No warp datasets for collections={collections!r} day_obs={day_obs}"
        )

    positions = []
    times = []
    bbox = None
    for ref in [warp_refs[-1], warp_refs[0]]:
        visit = butler.get(ref)
        bbox = bbox is not None and bbox or visit.getBBox()
        catalog = sso.propagate_injection_catalog(injection_catalog, visit)
        times.append(Time(catalog.meta["day_obs"]))
        x, y = visit.getWcs().skyToPixelArray(
            catalog["ra"],
            catalog["dec"],
            degrees=True,
        )
        origin_x, origin_y = visit.getXY0()
        x -= origin_x
        y -= origin_y
        positions.append([x, y])

    positions = np.array(positions)
    dx = positions[1][0] - positions[0][0]
    dy = positions[1][1] - positions[0][1]
    dt = (times[1] - times[0]).to(u.day).value

    cat = catalog
    cat["x0"] = x
    cat["y0"] = y
    cat["rate_x"] = dx / dt
    cat["rate_y"] = dy / dt
    cat.rename_column("injection_id", "plant_id")

    nx = bbox.getWidth()
    ny = bbox.getHeight()
    on_image = (
        (cat["x0"] > 0)
        & (cat["x0"] < nx)
        & (cat["y0"] > 0)
        & (cat["y0"] < ny)
    )
    return cat[
        "plant_id",
        "ra",
        "dec",
        "x0",
        "y0",
        "rate_x",
        "rate_y",
        "mag",
    ][on_image]
