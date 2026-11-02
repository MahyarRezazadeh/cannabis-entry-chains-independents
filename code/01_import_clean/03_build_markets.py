"""
03_build_markets.py  --  entry game project (self-contained)

Build every analysis panel from build/storefronts.csv + build/sf_month_revenue.csv +
build/brand_crosswalk.csv + public reference data.

Border definition (predetermined, fixed):
  a ZIP/storefront is BORDER if its point is within 75 mi of the NM state line facing a
  state with NO legal recreational cannabis market (TX, OK, UT -- CO & AZ are legal and
  excluded), AND >=1,000 residents of a non-legal state live within 75 mi.
Placebo border: same rule vs the LEGAL neighbour line (CO / AZ).

Outputs (build/):
  zip_xwalk.csv, city_xwalk.csv, panel_sf_quarter.csv, panel_zip_quarter.csv,
  panel_city_quarter.csv, city_cross_section.csv, panel_brand_quarter.csv,
  panel_brand_zip_quarter.csv
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import hav, asinh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF=os.path.join(ROOT, "data", "raw", "ref"); BUILD=os.path.join(ROOT, "data", "intermediate", "build_rebuilt")
LOG=os.path.join(ROOT, "output", "logs","03_build_markets.log")
_l=[]
def p(*a):
    s=" ".join(str(x) for x in a); print(s); _l.append(s)

sf=pd.read_csv(os.path.join(BUILD,"storefronts.csv"),dtype={"zip5":str})
sfm=pd.read_csv(os.path.join(BUILD,"sf_month_revenue.csv"))
xw=pd.read_csv(os.path.join(BUILD,"brand_crosswalk.csv"))
bm=pd.read_csv(os.path.join(BUILD,"brand_master.csv"))
sf=sf.merge(xw[["storefront_id","brand_id","brand_name","ownership_change_flag"]],on="storefront_id",how="left")

uz=pd.read_csv(os.path.join(REF,"uszips_simplemaps.csv"),dtype={"zip":str}).dropna(subset=["lat","lng","population"])
legal=pd.read_csv(os.path.join(REF,"legal_dates_new.csv"))
legal["rd"]=pd.to_datetime(legal["retail_date"],errors="coerce")
LEGAL=set(legal.loc[legal["rd"].notna() & (legal["rd"]<=pd.Timestamp("2023-12-31")),"state_abbr"])
p(f"legal-retail states @2023-12-31 ({len(LEGAL)}): {sorted(LEGAL)}")

# ---------------- state polygons: non-legal line & legal-neighbour line ----------------
st=gpd.read_file(os.path.join(REF,"cb_2023_us_state_500k","cb_2023_us_state_500k.shp"))
st=st[["STUSPS","geometry"]].rename(columns={"STUSPS":"s"}).to_crs(5070)
nonleg_poly=st[~st["s"].isin(LEGAL) & (st["s"]!="NM")].union_all()
legalneigh_poly=st[st["s"].isin(["CO","AZ"])].union_all()

# ---------------- ZIP crosswalk ----------------
z=uz[uz["state_id"]=="NM"].copy().rename(columns={"zip":"zip5","lng":"lon","population":"zip_pop"})
z["county_fips"]=z["county_fips"].astype(str).str.zfill(5)
pts=gpd.GeoSeries([Point(xy) for xy in zip(z["lon"],z["lat"])],crs=4326).to_crs(5070)
z["dist_nonleg_mi"]=pts.distance(nonleg_poly).to_numpy()/1609.34
z["dist_legalneigh_mi"]=pts.distance(legalneigh_poly).to_numpy()/1609.34
nl=uz[~uz["state_id"].isin(LEGAL) & (uz["state_id"]!="NM")]
ln=uz[uz["state_id"].isin(["CO","AZ"])]
nl_lat,nl_lon,nl_pop=nl["lat"].to_numpy(),nl["lng"].to_numpy(),nl["population"].to_numpy()
ln_lat,ln_lon,ln_pop=ln["lat"].to_numpy(),ln["lng"].to_numpy(),ln["population"].to_numpy()
for RAD in (25,50,75,100):
    z[f"nonleg_pop_{RAD}"]=[nl_pop[hav(a,o,nl_lat,nl_lon)<=RAD].sum() for a,o in zip(z["lat"],z["lon"])]
z["legalneigh_pop_75"]=[ln_pop[hav(a,o,ln_lat,ln_lon)<=75].sum() for a,o in zip(z["lat"],z["lon"])]
z["border_z"]=((z["dist_nonleg_mi"]<=75)&(z["nonleg_pop_75"]>=1000)).astype(int)
z["placebo_border_z"]=((z["dist_legalneigh_mi"]<=75)&(z["legalneigh_pop_75"]>=1000)).astype(int)
z["log_zip_pop"]=np.log(z["zip_pop"].clip(lower=1))
z["log_dist_nonleg"]=np.log(z["dist_nonleg_mi"].clip(lower=0.1))
z["mkt_size_zip"]=z["zip_pop"]+z["nonleg_pop_75"]        # crude combined-market-size index
z[["zip5","city","county_fips","county_name","lat","lon","zip_pop","log_zip_pop",
   "dist_nonleg_mi","log_dist_nonleg","dist_legalneigh_mi","border_z","placebo_border_z",
   "nonleg_pop_25","nonleg_pop_50","nonleg_pop_75","nonleg_pop_100","legalneigh_pop_75","mkt_size_zip"
   ]].to_csv(os.path.join(BUILD,"zip_xwalk.csv"),index=False)
p(f"NM ZIPs: {len(z)};  border {int(z['border_z'].sum())};  placebo-border {int(z['placebo_border_z'].sum())}")

# ---------------- storefront exposure (own coords) ----------------
def N(s):
    import unicodedata,re
    if pd.isna(s): return ""
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode()
    return re.sub(r"\s+"," ",s.strip().upper())
sf["city"]=sf["city"].map(N)
sf=sf.merge(z[["zip5","county_fips","county_name"]].drop_duplicates("zip5"),on="zip5",how="left")
gpts=gpd.GeoSeries([Point(xy) for xy in zip(sf["lon"],sf["lat"])],crs=4326).to_crs(5070)
sf["dist_nonleg_mi"]=gpts.distance(nonleg_poly).to_numpy()/1609.34
sf["dist_legalneigh_mi"]=gpts.distance(legalneigh_poly).to_numpy()/1609.34
for RAD in (25,50,75,100):
    sf[f"sf_nonleg_pop_{RAD}"]=[nl_pop[hav(a,o,nl_lat,nl_lon)<=RAD].sum() for a,o in zip(sf["lat"],sf["lon"])]
al_lat,al_lon,al_pop=uz["lat"].to_numpy(),uz["lng"].to_numpy(),uz["population"].to_numpy()
sf["sf_localpop_25"]=[al_pop[hav(a,o,al_lat,al_lon)<=25].sum() for a,o in zip(sf["lat"],sf["lon"])]
sf["border_sf"]=((sf["dist_nonleg_mi"]<=75)&(sf["sf_nonleg_pop_75"]>=1000)).astype(int)
sf["placebo_border_sf"]=((sf["dist_legalneigh_mi"]<=75)&
                         (np.array([ln_pop[hav(a,o,ln_lat,ln_lon)<=75].sum() for a,o in zip(sf["lat"],sf["lon"])])>=1000)).astype(int)
p(f"storefronts: border {int(sf['border_sf'].sum())} / interior {int((sf['border_sf']==0).sum())}")

# ---------------- city crosswalk (BR market unit) ----------------
zc=z.assign(city=z["city"].map(N))
citypop=zc.groupby("city").agg(city_pop=("zip_pop","sum"),
                               lat=("lat",lambda s:np.average(s,weights=zc.loc[s.index,"zip_pop"].clip(lower=1))),
                               lon=("lon",lambda s:np.average(s,weights=zc.loc[s.index,"zip_pop"].clip(lower=1))),
                               county_fips=("county_fips",lambda s:s.value_counts().index[0])).reset_index()
citypop["dist_nonleg_mi"]=[gpd.GeoSeries([Point(o,a)],crs=4326).to_crs(5070).distance(nonleg_poly).iloc[0]/1609.34
                           for a,o in zip(citypop["lat"],citypop["lon"])]
citypop["dist_legalneigh_mi"]=[gpd.GeoSeries([Point(o,a)],crs=4326).to_crs(5070).distance(legalneigh_poly).iloc[0]/1609.34
                               for a,o in zip(citypop["lat"],citypop["lon"])]
for RAD in (25,50,75,100):
    citypop[f"nonleg_pop_{RAD}"]=[nl_pop[hav(a,o,nl_lat,nl_lon)<=RAD].sum() for a,o in zip(citypop["lat"],citypop["lon"])]
citypop["legalneigh_pop_75"]=[ln_pop[hav(a,o,ln_lat,ln_lon)<=75].sum() for a,o in zip(citypop["lat"],citypop["lon"])]
citypop["border_city"]=((citypop["dist_nonleg_mi"]<=75)&(citypop["nonleg_pop_75"]>=1000)).astype(int)
citypop["placebo_border_city"]=((citypop["dist_legalneigh_mi"]<=75)&(citypop["legalneigh_pop_75"]>=1000)).astype(int)
# ABQ metro grouping
ABQ={"ALBUQUERQUE","RIO RANCHO","BERNALILLO","CORRALES","LOS LUNAS","BELEN","BOSQUE FARMS",
     "PLACITAS","TIJERAS","EDGEWOOD","PERALTA","RIO COMMUNITIES"}
citypop["metro"]=np.where(citypop["city"].isin(ABQ),"ALBUQUERQUE METRO",citypop["city"])
metro_pop=citypop.groupby("metro")["city_pop"].sum().rename("metro_pop")   # true resident pop
citypop.to_csv(os.path.join(BUILD,"city_xwalk.csv"),index=False)
p(f"NM cities: {len(citypop)};  border {int(citypop['border_city'].sum())}")

# ---------------- storefront x quarter panel ----------------
sfm["t"]=sfm["year"]*12+sfm["month"]
sfm["q"]=(sfm["month"]-1)//3+1
sfm["qt"]=sfm["year"]*4+(sfm["q"]-1)
QMIN=2022*4+1; QMAX=2026*4+1
sq=(sfm[sfm["qt"].between(QMIN,QMAX)].groupby(["storefront_id","qt"])
    .agg(revenue=("total","sum"),medical=("medical","sum"),n_mo=("month","nunique")).reset_index())
sq["active"]=(sq["revenue"]>0).astype(int)
firstq=sq[sq["active"]==1].groupby("storefront_id")["qt"].min()
lastq =sq[sq["active"]==1].groupby("storefront_id")["qt"].max()
meta=sf.set_index("storefront_id")
grid=[(sid,qt) for sid in firstq.index for qt in range(int(firstq[sid]),QMAX+1)]
sfq=pd.DataFrame(grid,columns=["storefront_id","qt"]).merge(sq,on=["storefront_id","qt"],how="left")
sfq["revenue"]=sfq["revenue"].fillna(0.0); sfq["active"]=sfq["active"].fillna(0).astype(int)
sfq["first_qt"]=sfq["storefront_id"].map(firstq); sfq["last_qt"]=sfq["storefront_id"].map(lastq)
sfq["left_censored"]=sfq["first_qt"]<=QMIN
sfq["exited"]=sfq["last_qt"]<=QMAX-2
sfq["exit_qt"]=np.where(sfq["exited"],sfq["last_qt"],np.nan)
sfq["age_q"]=sfq["qt"]-sfq["first_qt"]
sfq["entry_q"]=((sfq["qt"]==sfq["first_qt"])&(~sfq["left_censored"])).astype(int)
sfq["exit_next"]=((sfq["active"]==1)&(sfq["qt"]==sfq["exit_qt"])).astype(int)
sfq["in_exit_sample"]=((sfq["active"]==1)&(sfq["qt"]<=QMAX-2)).astype(int)
for c in ["brand_id","brand_name","city","zip5","county_name","lat","lon","dist_nonleg_mi",
          "dist_legalneigh_mi","border_sf","placebo_border_sf","sf_localpop_25","sf_nonleg_pop_25",
          "sf_nonleg_pop_50","sf_nonleg_pop_75","sf_nonleg_pop_100","ownership_change_flag","geo_src"]:
    sfq[c]=sfq["storefront_id"].map(meta[c])
sfq["log_revenue"]=np.log(sfq["revenue"].where(sfq["revenue"]>0))

# competition + network, per quarter
g=sfq[(sfq["active"]==1)&sfq["lat"].notna()].copy()
COLS=["n_disp_5mi","n_disp_10mi","n_disp_25mi","n_brand_5mi","n_brand_25mi",
      "samebrand_nearest_mi","samebrand_25mi","samebrand_100mi","brand_footprint_nm","n_disp_zip","n_brand_zip"]
acc={c:{} for c in COLS}
for qt,idx in g.groupby("qt").indices.items():
    s=g.iloc[idx]; la,lo=s["lat"].to_numpy(),s["lon"].to_numpy()
    bid=s["brand_id"].to_numpy(); zp=s["zip5"].to_numpy(); rid=s["storefront_id"].to_numpy()
    D=hav(la[:,None],lo[:,None],la[None,:],lo[None,:]); np.fill_diagonal(D,np.inf)
    for j in range(len(s)):
        d=D[j]; k=(int(rid[j]),int(qt))
        m5,m10,m25,m100=d<=5,d<=10,d<=25,d<=100
        sb=(bid==bid[j])&(np.arange(len(s))!=j)
        acc["n_disp_5mi"][k]=int(m5.sum()); acc["n_disp_10mi"][k]=int(m10.sum()); acc["n_disp_25mi"][k]=int(m25.sum())
        acc["n_brand_5mi"][k]=int(np.unique(bid[m5]).size); acc["n_brand_25mi"][k]=int(np.unique(bid[m25]).size)
        acc["samebrand_nearest_mi"][k]=float(d[sb].min()) if sb.any() else np.nan
        acc["samebrand_25mi"][k]=int((sb&m25).sum()); acc["samebrand_100mi"][k]=int((sb&m100).sum())
        acc["brand_footprint_nm"][k]=int(sb.sum()+1)
        acc["n_disp_zip"][k]=int((zp==zp[j]).sum()); acc["n_brand_zip"][k]=int(np.unique(bid[zp==zp[j]]).size)
kk=list(zip(sfq["storefront_id"].astype(int),sfq["qt"].astype(int)))
for c in COLS: sfq[c]=[acc[c].get(x,np.nan) for x in kk]
sfq.to_csv(os.path.join(BUILD,"panel_sf_quarter.csv"),index=False)
p(f"panel_sf_quarter: {len(sfq)} rows, {sfq['storefront_id'].nunique()} storefronts, "
  f"entrants {int(sfq['entry_q'].sum())}, exit sample {int(sfq['in_exit_sample'].sum())}, "
  f"exits {int(sfq.loc[sfq['in_exit_sample']==1,'exit_next'].sum())} "
  f"(rate {sfq.loc[sfq['in_exit_sample']==1,'exit_next'].mean():.4f})")

# ---------------- brand x quarter ----------------
sfa=sfq[sfq["active"]==1].copy()
bfirst=sfa.groupby("brand_id")["qt"].min()
bq=(sfa.groupby(["brand_id","brand_name","qt"]).agg(
    n_loc=("storefront_id","nunique"), n_border_loc=("border_sf","sum"),
    revenue=("revenue","sum"), n_zip=("zip5","nunique"), n_city=("city","nunique")).reset_index())
bq["brand_first_qt"]=bq["brand_id"].map(bfirst); bq["brand_age_q"]=bq["qt"]-bq["brand_first_qt"]
bq["border_share"]=bq["n_border_loc"]/bq["n_loc"]; bq["multi_now"]=(bq["n_loc"]>=2).astype(int)
bq.to_csv(os.path.join(BUILD,"panel_brand_quarter.csv"),index=False)

# ---------------- ZIP x quarter (concentration/composition) ----------------
lifetime_multi=dict(zip(bm["brand_id"],(bm["n_storefronts"]>=2).astype(int)))
sfa["lt_multi"]=sfa["brand_id"].map(lifetime_multi)
bq_nloc=bq.set_index(["brand_id","qt"])["n_loc"]
rows=[]
present=set(sfa.groupby(["brand_id","zip5","qt"]).groups.keys())
zbrand={(zz,qt):set(s["brand_id"]) for (zz,qt),s in sfa.groupby(["zip5","qt"])}
for (zz,qt),s in sfa.groupby(["zip5","qt"]):
    rev=s.groupby("brand_id")["revenue"].sum(); sh=(rev/rev.sum()).values if rev.sum()>0 else np.array([1.])
    ns=s["storefront_id"].nunique(); nb=s["brand_id"].nunique(); srt=np.sort(sh)[::-1]
    nloc_now=s["brand_id"].map(lambda b: bq_nloc.get((b,qt),1))
    prev=zbrand.get((zz,qt-1),set()); cur=zbrand.get((zz,qt),set()); nxt=zbrand.get((zz,qt+1),set())
    ent_ids=cur-prev
    rows.append(dict(zip5=zz,qt=qt,n_stores=ns,n_brands=nb,total_revenue=float(s["revenue"].sum()),
        loc_per_brand=ns/nb, hhi=float((sh**2).sum()), hhi_10000=float((sh**2).sum())*1e4,
        cr1=float(srt[:1].sum()),cr3=float(srt[:3].sum()),cr5=float(srt[:5].sum()),
        entropy=float(-(sh[sh>0]*np.log(sh[sh>0])).sum()),
        n_indep_brands=int((s.groupby("brand_id")["lt_multi"].first()==0).sum()),
        n_chain_brands=int((s.groupby("brand_id")["lt_multi"].first()==1).sum()),
        indep_store_share=float((s["lt_multi"]==0).mean()),
        chain_now_store_share=float((nloc_now>=2).mean()),
        brand_entries=len(ent_ids), brand_exits=(len(cur-nxt) if qt<=QMAX-2 else np.nan)))
zq=pd.DataFrame(rows)
zq["brand_entry_rate"]=zq["brand_entries"]/zq["n_brands"]
zq["brand_exit_rate"]=zq["brand_exits"]/zq["n_brands"]
zq=zq.merge(z[["zip5","city","county_fips","county_name","zip_pop","log_zip_pop","dist_nonleg_mi",
   "log_dist_nonleg","dist_legalneigh_mi","border_z","placebo_border_z","nonleg_pop_25","nonleg_pop_50",
   "nonleg_pop_75","nonleg_pop_100","legalneigh_pop_75"]],on="zip5",how="left")
zq.to_csv(os.path.join(BUILD,"panel_zip_quarter.csv"),index=False)
p(f"panel_zip_quarter: {len(zq)} rows, {zq['zip5'].nunique()} ZIPs "
  f"({zq[zq['border_z']==1]['zip5'].nunique()} border / {zq[zq['border_z']==0]['zip5'].nunique()} interior / "
  f"{zq['border_z'].isna().sum() and zq[zq['border_z'].isna()]['zip5'].nunique()} unmatched)")

# ---------------- city x quarter + city cross-section (BR) ----------------
CC=citypop[["city","metro","city_pop","border_city","placebo_border_city","dist_nonleg_mi",
    "dist_legalneigh_mi","nonleg_pop_25","nonleg_pop_50","nonleg_pop_75","nonleg_pop_100","legalneigh_pop_75"]
    ].rename(columns={"dist_nonleg_mi":"c_dist_nonleg","dist_legalneigh_mi":"c_dist_legal",
    "nonleg_pop_25":"c_nl25","nonleg_pop_50":"c_nl50","nonleg_pop_75":"c_nl75","nonleg_pop_100":"c_nl100",
    "legalneigh_pop_75":"c_ln75"})
sfa2=sfa[["storefront_id","brand_id","qt","city","lt_multi","revenue"]].merge(CC,on="city",how="left")
cq=(sfa2.groupby(["metro","qt"]).agg(
    n_stores=("storefront_id","nunique"),n_brands=("brand_id","nunique"),revenue=("revenue","sum"),
    border_city=("border_city","max"),placebo_border_city=("placebo_border_city","max"),
    c_dist_nonleg=("c_dist_nonleg","min"),c_dist_legal=("c_dist_legal","min"),
    c_nl75=("c_nl75","max"),c_nl50=("c_nl50","max"),c_nl25=("c_nl25","max"),c_nl100=("c_nl100","max"),
    c_ln75=("c_ln75","max"),
    n_indep=("brand_id",lambda s: s[sfa2.loc[s.index,"lt_multi"]==0].nunique()),
    n_chain=("brand_id",lambda s: s[sfa2.loc[s.index,"lt_multi"]==1].nunique())).reset_index())
cq=cq.merge(metro_pop,on="metro",how="left")
cq.to_csv(os.path.join(BUILD,"panel_city_quarter.csv"),index=False)
mat=cq[cq["qt"].between(2024*4,2025*4+3)]
cs=(mat.groupby("metro").agg(
    N_stores=("n_stores","mean"),N_brands=("n_brands","mean"),
    N_indep=("n_indep","mean"),N_chain=("n_chain","mean"),revenue=("revenue","mean"),
    city_pop=("metro_pop","first"),border_city=("border_city","first"),
    placebo_border_city=("placebo_border_city","first"),
    dist_nonleg_mi=("c_dist_nonleg","first"),dist_legalneigh_mi=("c_dist_legal","first"),
    nonleg_pop_75=("c_nl75","first"),nonleg_pop_50=("c_nl50","first"),
    nonleg_pop_25=("c_nl25","first"),nonleg_pop_100=("c_nl100","first"),
    legalneigh_pop_75=("c_ln75","first")).reset_index())
cs["N_stores_r"]=cs["N_stores"].round().astype(int); cs["N_brands_r"]=cs["N_brands"].round().astype(int)
cs["log_city_pop"]=np.log(cs["city_pop"].clip(lower=1))
cs["mkt_size"]=cs["city_pop"]+cs["nonleg_pop_75"]
cs["log_mkt_size"]=np.log(cs["mkt_size"].clip(lower=1))
cs["is_abq"]=(cs["metro"]=="ALBUQUERQUE METRO").astype(int)
cs.to_csv(os.path.join(BUILD,"city_cross_section.csv"),index=False)
p(f"city_cross_section: {len(cs)} metro-markets (2024-2025 mean);  "
  f"border {int(cs['border_city'].sum())};  ABQ metro flagged")

# ---------------- brand x ZIP x quarter (presence / entry / conditional-logit input) ----------------
op_zips=sorted(sfa["zip5"].dropna().unique())
abrand_q={qt:set(s["brand_id"]) for qt,s in sfa.groupby("qt")}
nm_present_q={qt:set(s["brand_id"]) for qt,s in sfa.groupby("qt")}
gc=g[["brand_id","qt","lat","lon"]].dropna()
bcoord={(b,qt):s[["lat","lon"]].to_numpy() for (b,qt),s in gc.groupby(["brand_id","qt"])}
zll=z.set_index("zip5")[["lat","lon"]].to_dict("index")
rev_by_zq=sfa.groupby(["zip5","qt"])["revenue"].sum().to_dict()
rows=[]
for qt in range(QMIN,QMAX+1):
    for b in (abrand_q.get(qt,set())|abrand_q.get(qt-1,set())):
        pc=bcoord.get((b,qt-1))
        for zz in op_zips:
            pres=(b,zz,qt) in present; prev=(b,zz,qt-1) in present
            nn=np.nan;n50=0;n100=0
            if pc is not None and zz in zll:
                d=hav(zll[zz]["lat"],zll[zz]["lon"],pc[:,0],pc[:,1])
                nn=float(d.min());n50=int((d<=50).sum());n100=int((d<=100).sum())
            rows.append((b,zz,qt,int(pres),int(pres and not prev),
                         int(b in nm_present_q.get(qt-1,set())),nn,n50,n100))
bzq=pd.DataFrame(rows,columns=["brand_id","zip5","qt","present","entry","brand_in_nm_prevq",
                               "brand_nearest_store_mi","brand_stores_50mi","brand_stores_100mi"])
bzq["expansion_entry"]=((bzq["entry"]==1)&(bzq["brand_in_nm_prevq"]==1)).astype(int)
bzq["denovo_entry"]=((bzq["entry"]==1)&(bzq["brand_in_nm_prevq"]==0)).astype(int)
bzq=bzq.merge(z[["zip5","border_z","placebo_border_z","log_dist_nonleg","dist_nonleg_mi","log_zip_pop",
   "zip_pop","nonleg_pop_75","county_fips"]],on="zip5",how="left")
bzq=bzq.merge(bm[["brand_id","n_storefronts"]].rename(columns={"n_storefronts":"brand_lifetime_n"}),on="brand_id",how="left")
bzq=bzq.merge(bq[["brand_id","qt","n_loc","brand_age_q","border_share","multi_now"]],on=["brand_id","qt"],how="left")
bzq.to_csv(os.path.join(BUILD,"panel_brand_zip_quarter.csv"),index=False)
p(f"panel_brand_zip_quarter: {len(bzq)} rows;  present {bzq['present'].mean():.4f};  "
  f"entries {int(bzq['entry'].sum())} (exp {int(bzq['expansion_entry'].sum())}, denovo {int(bzq['denovo_entry'].sum())})")

with open(LOG,"w",encoding="utf-8") as fh: fh.write("\n".join(_l))
p("DONE")
