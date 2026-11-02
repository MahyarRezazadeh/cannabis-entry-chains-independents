"""
02_brand_crosswalk.py  --  entry game project (self-contained)

Storefront -> parent-brand crosswalk from the raw CROP `Licensee` name strings.
Unit that is unambiguous = storefront (normalized street address).  Brand = parent
operator, inferred (never assumed from mere name similarity) by union-find over:
  A. exact shared raw modal name across >1 storefront            (high)
  B. identical algorithmically-cleaned brand key (>=1 dist. tok.) (medium)
  C. 41-entry hand-curated alias map (each = one real NM operator, verified in the
     review dump)                                                (curated)
Default: singleton brand.  Fuzzy similarity only proposes review pairs.

Outputs: build/brand_crosswalk.csv  build/brand_master.csv
         output/audit_brand_review.txt  output/audit_brand_fuzzy_pairs.csv
"""
import os, re, glob, unicodedata, difflib
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW =os.path.join(ROOT,"data","raw","nm_crop")
BUILD=os.path.join(ROOT, "data", "intermediate", "build_rebuilt"); AUD=os.path.join(ROOT,"output")
LOG =os.path.join(ROOT, "output", "logs","02_brand_crosswalk.log")
_l=[]
def p(*a):
    s=" ".join(str(x) for x in a); print(s); _l.append(s)

MONTHS={m:i+1 for i,m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"])}
def ym(fn):
    b=os.path.basename(fn)
    return int(re.match(r"(\d{4})_",b).group(1)), next(MONTHS[k] for k in MONTHS if k in b)
def NM(s):
    if pd.isna(s): return ""
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode()
    return re.sub(r"\s+"," ",s.strip().upper())
def Naddr(a):
    a=NM(a); return a.rstrip(".") if a else None

sf=pd.read_csv(os.path.join(BUILD,"storefronts.csv"))
rows=[]
for fn in sorted(glob.glob(os.path.join(RAW,"*Licensee*"))):
    y,m=ym(fn); d=pd.read_excel(fn); d.columns=[str(c).strip() for c in d.columns]
    d=d[["Licensee","Address","City","Zip"]].dropna(subset=["Licensee"])
    d=d[~d["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().isin(["totals","total"])]
    d["addr"]=d["Address"].map(Naddr); d["name"]=d["Licensee"].map(NM); d["city"]=d["City"].map(NM)
    d["t"]=y*12+m; rows.append(d.dropna(subset=["addr"]))
raw=pd.concat(rows,ignore_index=True).merge(sf[["storefront_id","addr"]],on="addr",how="inner")
CITIES=set(c for c in raw["city"].dropna().unique() if c)

LEGAL=re.compile(r"\b(L\.?L\.?C|INC(ORPORATED)?|CORP(ORATION)?|LLP|PLLC|LP|LC|PC|PA|CO(MPANY)?|LTD|"
                 r"HOLDINGS?|ENTERPRISES?|MANAGEMENT|GROUP|VENTURES?|INVESTMENTS?|PARTNERS?|FARMS?|LABS?)\b\.?")
NMTAG=re.compile(r"\b(OF NEW MEXICO|NEW MEXICO|N\.? ?M\.?)\b")
TRAIL={"RETAILER","RETAIL","RETAI","RTLR","DISPENSARIES","DISPENSARY","CONSUMPTION","MICROBUSINESS",
       "MANUFACTURER","MANUFACTURING","MFG","PRODUCERS","PRODUCER","PRODUCTION","PRODUCTIONS",
       "CULTIVATION","STORE","LOCATION","OUTLET","SHOP","DISP","LLC","INC","CO","THE"}
STREET=re.compile(r"\b(ST|STREET|RD|ROAD|BLVD|BOULEVARD|AVE|AVENUE|DR|DRIVE|HWY|HIGHWAY|LN|LANE|CT|"
                  r"PKWY|PARKWAY|WAY|LOOP|TRAIL|TRL|CIR|CIRCLE|PL|PLACE|NE|NW|SE|SW|NORTH|SOUTH|EAST|"
                  r"WEST|SUITE|STE|UNIT|BLDG|US)\b")
STOP1={"GREEN","GREENS","HIGH","THE","NEW","SMOKE","HERB","HERBS","LEAF","BUD","BUDS","FARM","FARMS",
       "CANNABIS","WEED","GRASS","TREE","TREES","GARDEN","GARDENS","FLOWER","FLOWERS","HOUSE","GOLD",
       "SUN","SKY","ROOTS","BLOOM","HAZE","KUSH","HEMP","JOINT","DANK","LOUD","LIT","STASH","STONE",
       "PUFF","TOKE","BLAZE","RELIEF","REMEDY","WELLNESS","HOLISTIC","ORGANIC","ORGANICS","NATURAL",
       "NATURALS","MEDS","MED"}
def bkey(name):
    s=NM(name)
    m=re.search(r"\bD ?/? ?B ?/? ?A ?/?\b\s*(.+)$",s)
    if m: s=m.group(1)
    s=re.sub(r"^\s*NMTO\s*[-:]\s*","",s); s=re.sub(r"^\s*THE\s+","",s)
    s=LEGAL.sub(" ",s); s=re.split(r"[-/\\]",s,maxsplit=1)[0]
    s=re.sub(r"[.,'\"()&]"," ",s); s=NMTAG.sub(" ",s)
    toks=[t for t in s.split() if t]
    while toks and toks[-1] in TRAIL: toks.pop()
    while len(toks)>1 and toks[-1] in CITIES: toks.pop()
    s=" ".join(toks); s=re.sub(r"^\s*\d{2,5}\b"," ",s); s=STREET.sub(" ",s)
    s=re.sub(r"\bSTE\b|\bSUITE\b"," ",s)
    return re.sub(r"\s+"," ",s).strip()

ALIAS=[
 (r"\bULTRA HEALTH\b|\bNMTO\b|NEW MEXICO TOP ORGANICS|ULTRA HEALTHALAMOGORDO","ULTRA HEALTH"),
 (r"\bR\.? ?GREEN ?LEAF\b","R GREENLEAF ORGANICS"),(r"PECOS VALLEY","PECOS VALLEY PRODUCTION"),
 (r"EVEREST APOTHECARY|EVEREST CANNABIS","EVEREST CANNABIS CO"),(r"\bVERDES\b","VERDES FOUNDATION"),
 (r"\bURBAN WELLNESS\b","URBAN WELLNESS"),(r"\bPUR ?LIFE\b","PURLIFE"),(r"\bSACRED GARDEN\b","SACRED GARDEN"),
 (r"\bSAWMILL\b","SAWMILL CANNABIS"),(r"SOUTHWEST ORGANIC","SOUTHWEST ORGANIC PRODUCERS"),
 (r"\bPROHIBITION 37\b","PROHIBITION 37"),(r"LOVE'?S CANNABIS|\bLOVES CANNABIS\b","LOVES CANNABIS"),
 (r"CARVER FAMILY FARM|\bCARVER'?S\b","CARVER FAMILY FARM"),(r"\bMINERVA\b","MINERVA CANNA"),
 (r"\bNATURAL RX\b","NATURAL RX"),(r"BRYAN'?S GREEN ?CARE","BRYANS GREENCARE"),
 (r"COMPASSIONATE DISTRIBUTORS","COMPASSIONATE DISTRIBUTORS"),(r"FRUIT ?OF ?THE ?EARTH","FRUIT OF THE EARTH ORGANICS"),
 (r"\bSEVEN CLOVER\b|\b7 CLOVER\b","SEVEN CLOVER"),(r"\bENCHANTED BOTANICALS\b","ENCHANTED BOTANICALS"),
 (r"\bDUKE CITY\b","DUKE CITY CANNABIS"),(r"VIREO HEALTH|VIREO OF SANTA FE|\bGREEN GOODS\b","GREEN GOODS"),
 (r"\bHIGH DESERT RELIEF\b","HIGH DESERT RELIEF"),(r"\bBEST DAZE\b","BEST DAZE"),
 (r"SANTA FE FARMS","SANTA FE FARMS"),(r"MARIJUANA MEDICINE MAN|\bMEDICINE MAN\b","MEDICINE MAN"),
 (r"\bRIO GRANDE CANNABIS\b","RIO GRANDE CANNABIS"),(r"\bKURE\b","KURE"),
 (r"GRASSROOTS GRANTS","GRASSROOTS GRANTS"),(r"GRASS ?ROOTS RX","GRASSROOTS RX"),
 (r"BUDDIES FARMS? L","BUDDIES FARMS"),(r"\bLEAF AND FLOWER\b|\bLEAF & FLOWER\b","LEAF AND FLOWER"),
 (r"\bSCORE 420\b","SCORE 420"),(r"\bELKHORN\b","ELKHORN"),(r"\bOCC ABQ\b","OCC ABQ"),
 (r"\bFCI\b.*NEW MEXICO|FCI OF NM","FCI OF NEW MEXICO"),(r"\bRMC NM\b","RMC NM"),(r"\bACCENDO\b","ACCENDO"),
 (r"SOUTHWEST WELLNESS","SOUTHWEST WELLNESS CENTER"),(r"\b285 SOUTH\b","285 SOUTH DISPENSARY"),
]
def alias_of(nr,k):
    for pat,c in ALIAS:
        if re.search(pat,nr) or (k and re.search(pat,k)): return c
    return None

sf["bkey"]=sf["name_modal"].map(bkey)
sf["alias"]=[alias_of(nr,bk) for nr,bk in zip(sf["name_modal"],sf["bkey"])]
def keydiff(all_names):
    ks={bkey(x) for x in str(all_names).split(" || ")}
    return len({k for k in ks if k})
sf["ownership_change_flag"]=sf["all_names"].map(keydiff)>1
def usable(k):
    if not k: return False
    t=k.split()
    return len(t)>=2 or (len(t)==1 and len(t[0])>=4 and t[0] not in STOP1)
sf["bkey_ok"]=sf["bkey"].map(usable)

par={i:i for i in sf["storefront_id"]}
def find(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
def uni(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: par[max(ra,rb)]=min(ra,rb)
GEN={"","CANNABIS","DISPENSARY","RETAIL","THE DISPENSARY","DISPENSARY LLC"}
for nm_,g in sf.groupby("name_modal"):
    if nm_ in GEN or len(g)<2: continue
    ids=g["storefront_id"].tolist()
    for j in ids[1:]: uni(ids[0],j)
for k,g in sf[sf["bkey_ok"]].groupby("bkey"):
    ids=g["storefront_id"].tolist()
    for j in ids[1:]: uni(ids[0],j)
for k,g in sf[sf["alias"].notna()].groupby("alias"):
    ids=g["storefront_id"].tolist()
    for j in ids[1:]: uni(ids[0],j)
sf["comp"]=sf["storefront_id"].map(find)

def disp_name(name):
    s=NM(name)
    m=re.search(r"\bD ?/? ?B ?/? ?A\b\s*(.+)$",s)
    if m: s=m.group(1)
    s=re.sub(r"^\s*NMTO\s*[-:]\s*","",s); s=re.split(r"[-/\\]",s,maxsplit=1)[0]
    s=LEGAL.sub(" ",s); s=NMTAG.sub(" ",s); s=re.sub(r"[.,]"," ",s)
    toks=[t for t in s.split() if t]
    while toks and toks[-1] in (TRAIL-{"CO","THE"}): toks.pop()
    while len(toks)>1 and toks[-1] in CITIES: toks.pop()
    return re.sub(r"\s+"," "," ".join(toks)).strip() or NM(name)

R=[]
for comp,g in sf.groupby("comp"):
    if g["alias"].notna().any():
        d=g["alias"].dropna().mode().iloc[0]; rule="C_alias"
    elif (g["name_modal"].map(lambda x:(sf["name_modal"]==x).sum()).max()>=2 and g["name_modal"].nunique()==1):
        d=disp_name(g["name_modal"].iloc[0]); rule="A_sharedname"
    elif g["bkey_ok"].any():
        d=disp_name(g.loc[g["bkey_ok"],"name_modal"].value_counts().index[0]); rule="B_cleankey"
    else:
        d=disp_name(g["name_modal"].iloc[0]); rule="D_singleton"
    R.append((comp,d,rule))
cd=pd.DataFrame(R,columns=["comp","brand_name","link_rule"])
dup=cd["brand_name"].duplicated(keep=False)
cd.loc[dup,"brand_name"]=cd.loc[dup,"brand_name"]+" ["+cd.loc[dup,"comp"].astype(str)+"]"
cd=cd.sort_values("brand_name").reset_index(drop=True); cd["brand_id"]=np.arange(len(cd))
sf=sf.merge(cd[["comp","brand_id","brand_name","link_rule"]],on="comp",how="left")

bm=sf.groupby(["brand_id","brand_name","link_rule"]).agg(
    n_storefronts=("storefront_id","nunique"), n_cities=("city","nunique"),
    n_zips=("zip5","nunique"), first_t=("first_t","min"),
    lifetime_sales=("lifetime_sales","sum")).reset_index()
bm["multi_location"]=(bm["n_storefronts"]>=2).astype(int)

p(f"storefronts {sf['storefront_id'].nunique()};  brands {sf['brand_id'].nunique()};  "
  f"multi-location {(bm['n_storefronts']>=2).sum()};  singleton {(bm['n_storefronts']==1).sum()};  "
  f"ownership-change storefronts {int(sf['ownership_change_flag'].sum())}")
p("link_rule (storefronts):\n"+sf["link_rule"].value_counts().to_string())
p("\ntop 20 brands by #storefronts:")
p(bm.sort_values("n_storefronts",ascending=False).head(20)[
    ["brand_name","n_storefronts","n_cities","link_rule"]].to_string(index=False))

sf[["storefront_id","brand_id","brand_name","link_rule","name_modal","bkey","alias",
    "n_name_variants","ownership_change_flag","all_names"]].to_csv(
    os.path.join(BUILD,"brand_crosswalk.csv"),index=False)
bm.to_csv(os.path.join(BUILD,"brand_master.csv"),index=False)

with open(os.path.join(AUD,"audit_brand_review.txt"),"w",encoding="utf-8") as fh:
    fh.write("BRAND REVIEW DUMP\n"+"="*90+"\n\n")
    for multi in (1,0):
        for _,br in bm[bm["multi_location"]==multi].sort_values(
                ["n_storefronts","brand_name"],ascending=[False,True]).iterrows():
            sub=sf[sf["brand_id"]==br["brand_id"]]
            fh.write(f"[{int(br['brand_id'])}] {br['brand_name']} (n={int(br['n_storefronts'])}, rule={br['link_rule']})\n")
            for _,r in sub.sort_values("city").iterrows():
                fh.write(f"   id{r['storefront_id']:>4} {str(r['city'])[:16]:<16} {r['zip5']} | {r['name_modal'][:70]}\n")
            fh.write("\n")
sg=bm[bm["n_storefronts"]==1]["brand_name"].tolist()
cds=[]
for i in range(len(sg)):
    for j in range(i+1,len(sg)):
        rr=difflib.SequenceMatcher(None,sg[i],sg[j]).ratio()
        if rr>=0.86: cds.append((sg[i],sg[j],round(rr,3)))
pd.DataFrame(cds,columns=["a","b","ratio"]).to_csv(os.path.join(AUD,"audit_brand_fuzzy_pairs.csv"),index=False)
with open(LOG,"w",encoding="utf-8") as fh: fh.write("\n".join(_l))
p(f"\nfuzzy pairs (not merged): {len(cds)}   WROTE build/brand_crosswalk.csv, build/brand_master.csv")
