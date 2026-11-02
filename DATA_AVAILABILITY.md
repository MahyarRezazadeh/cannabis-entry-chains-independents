# Data availability

| Data | Source and access | Version / date | In package? |
|---|---|---|---|
| Monthly cannabis sales by licensee and address, April 2022 -- July 2026 | New Mexico Cannabis Control Division, Cannabis Reporting Online Portal, data catalog: https://crop.rld.nm.gov/data-catalog.html | 51 monthly workbooks as downloaded by the author (file names preserved) | Yes (`data/raw/nm_crop/`) |
| Storefront geocodes | U.S. Census Bureau batch geocoder (benchmark Public_AR_Current), run by `code/01_import_clean/01_ingest_geocode.py` | geocoded in 2026; archived in `data/intermediate/build/storefronts.csv` | Yes (derived) |
| ZIP-code populations, centroids, density | SimpleMaps US ZIP Codes Database, Basic version, https://simplemaps.com/data/us-zips | accessed 2026; 33,099 ZIP codes | Yes (`data/raw/ref/uszips_simplemaps.csv`), CC BY 4.0 with attribution |
| State boundaries | Census cartographic boundary file cb_2023_us_state_500k | 2023 | Yes |
| Tribal areas | Census cartographic boundary file cb_2023_us_aiannh_500k, https://www2.census.gov/geo/tiger/GENZ2023/shp/ | downloaded 2026-10-23 | Yes |
| 2020 Census blocks, New Mexico | Census TIGER/Line tl_2020_35_tabblock20 (https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_35_tabblock20.zip) | 2020 | No (large); only needed to rebuild `sf_local_population.csv`, which is included. Place the unzipped shapefile in `data/raw/census_blocks/tl_2020_35_tabblock20/` |
| ZIP Codes Business Patterns, NM ZIP codes, 2018--2023 | U.S. Census Bureau, https://www.census.gov/programs-surveys/cbp.html | extract built by the author | Yes (`data/raw/zbp/nm_zip_zbp_raw.csv`) |
| Road travel times | Open Source Routing Machine table service (router.project-osrm.org), car profile, OpenStreetMap data | retrieved 2026-10-23; cached matrix 368 x 277 | Yes (`data/intermediate/osrm_durations_nm_to_nonlegal.csv`); OpenStreetMap data (c) OpenStreetMap contributors, ODbL |
| Colorado county sales and licensees (Colorado comparison only) | Colorado Department of Revenue, Colorado Information Marketplace | 2026 | No; the Colorado tables are reproduced from saved outputs (see README) |

No restricted or confidential data are used. The sales records are public administrative records published by the
State of New Mexico; they identify licensed businesses, not individuals.

**Statement for the manuscript.** The cannabis sales records are public monthly workbooks published by the New Mexico
Cannabis Control Division. ZIP-code data come from the SimpleMaps U.S. ZIP Codes Database (Basic version, CC BY 4.0);
census blocks, tribal-area boundaries, state boundaries, and ZIP Codes Business Patterns are public U.S. Census Bureau
products; travel times were computed with the public OSRM service on OpenStreetMap data. The replication package
contains all code, constructed panels, and shareable raw inputs. Repository and DOI: [to be added by the author].
