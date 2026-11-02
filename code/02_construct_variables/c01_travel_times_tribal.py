r"""
c01_travel_times_tribal.py -- construct two new ZIP-level variables.

1. Road-network travel times from every New Mexico ZIP-code centroid to every populated Texas, Oklahoma, and
   Utah ZIP-code centroid within 150 straight-line miles, from the public OSRM routing service
   (router.project-osrm.org, car profile on OpenStreetMap data). Requests are paced at about one per second.
   The raw duration matrix is cached in data/intermediate/osrm_durations_nm_to_nonlegal.csv together with the
   retrieval date; later runs reuse the cache. These are modelled free-flow driving times, not observed trips.
2. An indicator for ZIP-code centroids inside American Indian / Alaska Native / Native Hawaiian areas
   (Census cartographic boundary file cb_2023_us_aiannh_500k).
Output: data/intermediate/zip_travel_tribal.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import json, time, datetime, urllib.request, zipfile
import numpy as np
import pandas as pd
from paths import INTER, REF, LOGS
from _common import load_zx, us_zips, hav, NONLEGAL_STATES

CACHE = INTER / "osrm_durations_nm_to_nonlegal.csv"
zx = load_zx()
z = us_zips()
nl = z[z.state_id.isin(NONLEGAL_STATES)].copy()
dmin = hav(zx.lat.to_numpy()[:, None], zx.lon.to_numpy()[:, None], nl.lat.to_numpy()[None, :], nl.lng.to_numpy()[None, :]).min(0)
dest = nl[dmin <= 150].reset_index(drop=True)
src = zx[["zip5", "lat", "lon"]].reset_index(drop=True)
print(f"sources {len(src)} NM ZIPs; destinations {len(dest)} nonlegal-state ZIPs within 150 mi "
      f"(population {dest.population.sum():,.0f})")

if CACHE.exists():
    D = pd.read_csv(CACHE, index_col=0)
    D.index, D.columns = D.index.astype(str).str.zfill(5), D.columns.astype(str).str.zfill(5)
    print("using cached OSRM matrix", CACHE.name)
else:
    D = pd.DataFrame(np.nan, index=src.zip5, columns=dest.zip.astype(str))
    SB, DB = 25, 70
    for i0 in range(0, len(src), SB):
        s = src.iloc[i0:i0 + SB]
        for j0 in range(0, len(dest), DB):
            t = dest.iloc[j0:j0 + DB]
            coords = ";".join(f"{x:.5f},{y:.5f}" for x, y in zip(s.lon, s.lat)) + ";" + \
                     ";".join(f"{x:.5f},{y:.5f}" for x, y in zip(t.lng, t.lat))
            si = ";".join(str(k) for k in range(len(s)))
            di = ";".join(str(len(s) + k) for k in range(len(t)))
            url = f"https://router.project-osrm.org/table/v1/driving/{coords}?sources={si}&destinations={di}&annotations=duration"
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "academic-research/1.0"}), timeout=60) as r:
                        js = json.load(r)
                    if js.get("code") == "Ok":
                        break
                except Exception as e:
                    time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"OSRM failed for block {i0},{j0}")
            D.iloc[i0:i0 + len(s), j0:j0 + len(t)] = np.array(js["durations"], dtype=float)
            time.sleep(1.1)
        print(f"  rows {i0}-{i0 + len(s) - 1} done")
    D.to_csv(CACHE)
    (INTER / "osrm_retrieval_note.txt").write_text(
        f"OSRM table service, router.project-osrm.org, car profile; retrieved {datetime.date.today().isoformat()}; "
        f"{len(src)} NM ZIP centroids x {len(dest)} TX/OK/UT ZIP centroids within 150 straight-line miles; seconds.\n")

mins = D.to_numpy(dtype=float) / 60.0
pop = dest.set_index(dest.zip.astype(str)).loc[D.columns, "population"].to_numpy()
out = pd.DataFrame({"zip5": D.index})
for m in (60, 90, 120):
    out[f"tt_pop_{m}"] = np.nansum(pop[None, :] * (mins <= m), axis=1)
out["tt_decay_60"] = np.nansum(pop[None, :] * np.exp(-mins / 60.0), axis=1)
big = pop >= 10000
out["tt_min_bigzip"] = np.nanmin(np.where(big[None, :], mins, np.inf), axis=1)
print(f"missing durations: {np.isnan(mins).sum()} of {mins.size}")

# tribal areas
zf = REF / "cb_2023_us_aiannh_500k.zip"
import geopandas as gpd
from shapely.geometry import Point
tr = gpd.read_file(f"zip://{zf}")
tr = tr.to_crs("EPSG:4326")
pts = gpd.GeoDataFrame(zx[["zip5"]].copy(), geometry=[Point(x, y) for x, y in zip(zx.lon, zx.lat)], crs="EPSG:4326")
j = gpd.sjoin(pts, tr[["NAME", "geometry"]], how="left", predicate="within")
first = j.groupby("zip5").NAME.first()
out["tribal_centroid"] = out.zip5.map(first.notna()).astype(int)
out["tribal_area_name"] = out.zip5.map(first)
out.to_csv(INTER / "zip_travel_tribal.csv", index=False)
grp = dict(zip(zx.zip5, zx.group))
o = out.assign(group=out.zip5.map(grp))
print(o.groupby("group")[["tt_pop_90", "tt_min_bigzip", "tribal_centroid"]].agg(["mean", "median"]).round(1).to_string())
print("tribal ZIP centroids by group:", o.groupby("group").tribal_centroid.sum().to_dict())
print("tribal areas (top):", o[o.tribal_centroid == 1].tribal_area_name.value_counts().head(12).to_dict())
