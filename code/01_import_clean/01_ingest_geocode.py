"""
01_ingest_geocode.py  --  entry game project (self-contained)

Parse the 51 NM CROP "Sales by Licensee" monthly workbooks -> storefront master +
storefront-month revenue.  Geocode storefront addresses via the U.S. Census Bureau
batch geocoder (public, no key), with a city-centroid fallback for non-matches.

Outputs (build/):
  storefronts.csv          one row per storefront (address) : name, city, zip, coords, entry/exit month
  sf_month_revenue.csv      storefront_id x year x month : total/medical/adultuse sales
"""
import os, re, io, glob, time, unicodedata
import numpy as np, pandas as pd, requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW  = os.path.join(ROOT, "data", "raw", "nm_crop")
REF  = os.path.join(ROOT, "data", "raw", "ref")
BUILD= os.path.join(ROOT, "data", "intermediate", "build_rebuilt"); os.makedirs(BUILD, exist_ok=True)
LOG  = os.path.join(ROOT, "output", "logs", "01_ingest_geocode.log")
_l=[]
def p(*a):
    s=" ".join(str(x) for x in a); print(s); _l.append(s)

MONTHS={m:i+1 for i,m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"])}
def ym(fn):
    b=os.path.basename(fn)
    return int(re.match(r"(\d{4})_",b).group(1)), next(MONTHS[k] for k in MONTHS if k in b)
def N(s):
    if pd.isna(s): return ""
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode()
    return re.sub(r"\s+"," ",s.strip().upper())
def Naddr(a):
    a=N(a); return a.rstrip(".") if a else None

# ---------- 1. stack raw ----------
rows=[]
for fn in sorted(glob.glob(os.path.join(RAW,"*Licensee*"))):
    y,m=ym(fn)
    d=pd.read_excel(fn); d.columns=[str(c).strip() for c in d.columns]
    d=d[["Licensee","Address","City","State","Zip","Medical Sales","Adult-Use Sales","Total Sales"]]
    d=d.dropna(subset=["Licensee"])
    d=d[~d["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().isin(
        ["totals","total","grand total","grand totals"])]
    for c in ["Medical Sales","Adult-Use Sales","Total Sales"]:
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0.0)
    d["year"],d["month"]=y,m
    rows.append(d)
raw=pd.concat(rows,ignore_index=True)
raw["addr"]=raw["Address"].map(Naddr)
raw["name"]=raw["Licensee"].map(N)
raw["city"]=raw["City"].map(N)
raw["zip5"]=raw["Zip"].astype(str).str.extract(r"(\d{5})")[0]
raw=raw.dropna(subset=["addr","zip5"]).copy()
raw["t"]=raw["year"]*12+raw["month"]
p(f"raw licensee-month rows: {len(raw)};  distinct addresses: {raw['addr'].nunique()};  "
  f"months: {raw['t'].nunique()} ({raw['year'].min()}-{raw['month'].min():02d} .. {raw['year'].max()}-{raw['month'].max():02d})")
p(f"total lifetime sales in files: ${raw['Total Sales'].sum():,.0f}")

# storefront id = normalized address
sid=raw[["addr"]].drop_duplicates().reset_index(drop=True); sid["storefront_id"]=sid.index
raw=raw.merge(sid,on="addr")

def modal(s):
    s=s.dropna(); return s.value_counts().index[0] if len(s) else None
sf=raw.groupby("storefront_id").agg(
    addr=("addr","first"),
    name_modal=("name",modal),
    name_first=("name",lambda s: raw.loc[raw.loc[s.index,"t"].idxmin(),"name"]),
    n_name_variants=("name","nunique"),
    all_names=("name",lambda s:" || ".join(sorted(s.unique()))),
    city=("city",modal), zip5=("zip5",modal),
    first_t=("t","min"), last_t=("t","max"), n_months=("t","nunique"),
    lifetime_sales=("Total Sales","sum"),
).reset_index()
p(f"storefronts: {len(sf)}")

# ---------- 2. Census batch geocode ----------
def parse_street(a):
    # census likes "123 Main St" without suite; strip trailing 'STE/SUITE/UNIT X'
    s=re.sub(r"\b(STE|SUITE|UNIT|#)\s*\S+\s*$","",a).strip()
    return s or a
sf["street"]=sf["addr"].map(parse_street)

def census_batch(df):
    buf=io.StringIO()
    for r in df.itertuples():
        buf.write(f'{r.storefront_id},"{r.street}","{r.city}","NM","{r.zip5}"\n')
    buf.seek(0)
    for attempt in range(4):
        try:
            resp=requests.post(
                "https://geocoding.geo.census.gov/geocoder/locations/addressbatch",
                files={"addressFile":("addr.csv",buf.getvalue(),"text/csv")},
                data={"benchmark":"Public_AR_Current","returntype":"locations"},
                timeout=300)
            resp.raise_for_status()
            out=pd.read_csv(io.StringIO(resp.text),header=None,dtype=str,
                            names=["id","input","match","mtype","matched","coord","tiger","side"])
            return out
        except Exception as ex:
            p(f"  census batch attempt {attempt+1} failed: {ex}"); time.sleep(5)
    return pd.DataFrame(columns=["id","input","match","mtype","matched","coord","tiger","side"])

geo_parts=[]
for i in range(0,len(sf),1000):
    part=census_batch(sf.iloc[i:i+1000])
    geo_parts.append(part)
    p(f"  geocoded batch {i}-{i+len(part)}: {(part['match']=='Match').sum()} matched")
geo=pd.concat(geo_parts,ignore_index=True)
geo["storefront_id"]=pd.to_numeric(geo["id"],errors="coerce")
geo=geo.dropna(subset=["storefront_id"]); geo["storefront_id"]=geo["storefront_id"].astype(int)
ll=geo["coord"].str.split(",",expand=True)
geo["lon"]=pd.to_numeric(ll[0],errors="coerce"); geo["lat"]=pd.to_numeric(ll[1],errors="coerce")
geo=geo[["storefront_id","match","mtype","matched","lon","lat"]].drop_duplicates("storefront_id")
sf=sf.merge(geo,on="storefront_id",how="left")
sf["geo_src"]=pd.Series(np.where(sf["lat"].notna(),"census_batch",None),index=sf.index,dtype=object)
p(f"census match: {sf['lat'].notna().sum()} / {len(sf)}")

# ---------- 3. city-centroid fallback ----------
uz=pd.read_csv(os.path.join(REF,"uszips_simplemaps.csv"),dtype={"zip":str})
uz_nm=uz[uz["state_id"]=="NM"].copy()
city_ll=(uz_nm.assign(city=uz_nm["city"].map(N))
         .groupby("city").apply(lambda d: pd.Series({
             "lat":np.average(d["lat"],weights=d["population"].clip(lower=1)),
             "lon":np.average(d["lng"],weights=d["population"].clip(lower=1))}))
         .reset_index())
zip_ll=uz_nm.assign(zip5=uz_nm["zip"]).set_index("zip5")[["lat","lng"]].rename(columns={"lng":"lon"})
miss=sf["lat"].isna()
# try ZIP centroid first, then city centroid
sf.loc[miss,"lat"]=sf.loc[miss,"zip5"].map(zip_ll["lat"])
sf.loc[miss,"lon"]=sf.loc[miss,"zip5"].map(zip_ll["lon"])
sf.loc[sf["lat"].notna() & sf["geo_src"].isna(),"geo_src"]="zip_centroid"
miss=sf["lat"].isna()
sf=sf.merge(city_ll.rename(columns={"lat":"clat","lon":"clon"}),on="city",how="left")
sf.loc[miss,"lat"]=sf.loc[miss,"clat"]; sf.loc[miss,"lon"]=sf.loc[miss,"clon"]
sf.loc[sf["lat"].notna() & sf["geo_src"].isna(),"geo_src"]="city_centroid"
sf=sf.drop(columns=["clat","clon"])
p(f"final coords: {sf['lat'].notna().sum()} / {len(sf)}   by source:\n{sf['geo_src'].value_counts(dropna=False).to_string()}")

# ---------- 4. storefront-month revenue + entry/exit ----------
sfm=(raw.groupby(["storefront_id","year","month"])
     .agg(total=("Total Sales","sum"),medical=("Medical Sales","sum"),
          adultuse=("Adult-Use Sales","sum")).reset_index())
sfm["t"]=sfm["year"]*12+sfm["month"]
T0,T1=int(raw["t"].min()),int(raw["t"].max())
first=sfm[sfm["total"]>0].groupby("storefront_id")["t"].min()
last =sfm[sfm["total"]>0].groupby("storefront_id")["t"].max()
sf["entry_t"]=sf["storefront_id"].map(first); sf["last_t_active"]=sf["storefront_id"].map(last)
sf["left_censored"]=sf["entry_t"]<=T0
sf["exited"]=sf["last_t_active"]<=T1-6        # >=6 silent months before panel end
sf["exit_t"]=np.where(sf["exited"],sf["last_t_active"],np.nan)
p(f"panel months {T0}..{T1};  left-censored storefronts {int(sf['left_censored'].sum())};  "
  f"observed entrants {int((~sf['left_censored']).sum())};  exiters {int(sf['exited'].sum())}")

sf.to_csv(os.path.join(BUILD,"storefronts.csv"),index=False)
sfm.to_csv(os.path.join(BUILD,"sf_month_revenue.csv"),index=False)
with open(LOG,"w",encoding="utf-8") as fh: fh.write("\n".join(_l))
p("WROTE build/storefronts.csv, build/sf_month_revenue.csv")
