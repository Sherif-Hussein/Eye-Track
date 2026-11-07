import sys, os, json, hashlib, urllib.request, datetime
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "."
RAW = ROOT + "/data/external/vdc2024/raw"
OUT = ROOT + "/results/phase9/vdc_heldout/step1"
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def jget(url):
    return json.loads(get(url).decode("utf-8"))

files = []
def walk(url, rel_prefix, node_label):
    while url:
        d = jget(url)
        for it in d["data"]:
            a = it["attributes"]
            if a["kind"] == "folder":
                walk(it["relationships"]["files"]["links"]["related"]["href"], rel_prefix + a["name"] + "/", node_label)
            else:
                files.append(dict(node=node_label, rel=rel_prefix + a["name"], size=a.get("size"),
                                  url=it["links"]["download"], modified=a.get("date_modified")))
        url = d["links"].get("next")

def do_node(node_id, label):
    n = jget(f"https://api.osf.io/v2/nodes/{node_id}/")
    at = n["data"]["attributes"]
    print("NODE", node_id, at["title"], "| public:", at["public"], "| modified:", at["date_modified"])
    lic = None
    try:
        lic = jget(f"https://api.osf.io/v2/nodes/{node_id}/license/")["data"]["attributes"]["name"]
    except Exception:
        try:
            lr = n["data"]["relationships"].get("license", {})
            lic = lr.get("links", {}).get("related", {}).get("title")
        except Exception:
            pass
    provs = jget(f"https://api.osf.io/v2/nodes/{node_id}/files/")
    for p in provs["data"]:
        print("  provider:", p["attributes"]["name"])
        walk(p["relationships"]["files"]["links"]["related"]["href"], p["attributes"]["name"] + "/", label)
    # child components
    url = f"https://api.osf.io/v2/nodes/{node_id}/children/"
    kids = []
    while url:
        d = jget(url)
        kids += d["data"]
        url = d["links"].get("next")
    return at, lic, kids

meta = {}
at, lic, kids = do_node("p562n", "p562n")
meta["p562n"] = dict(title=at["title"], license=lic, description=at.get("description"), tags=at.get("tags"),
                     doi=at.get("doi"), date_created=at["date_created"], date_modified=at["date_modified"])
stack = list(kids)
while stack:
    k = stack.pop()
    kid = k["id"]
    at2, lic2, kids2 = do_node(kid, kid)
    meta[kid] = dict(title=at2["title"], license=lic2, description=at2.get("description"))
    stack += kids2

print("N files:", len(files))
for f in files:
    print(f["node"], f["rel"], f["size"])
json.dump(dict(meta=meta, files=files), open(OUT + "/osf_listing.json", "w", encoding="utf-8"), indent=1)

rows = []
for f in files:
    dest = os.path.join(RAW, f["node"], f["rel"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t0 = datetime.datetime.now(datetime.timezone.utc)
    data = get(f["url"])
    open(dest, "wb").write(data)
    t1 = datetime.datetime.now(datetime.timezone.utc)
    h = hashlib.sha256(data).hexdigest()
    rows.append(dict(node=f["node"], rel=f["rel"], size=len(data), sha256=h, osf_modified=f["modified"],
                     utc_downloaded=t1.strftime("%Y-%m-%d %H:%M:%S UTC")))
    print("downloaded", f["rel"], len(data), h)
json.dump(rows, open(OUT + "/download_manifest.json", "w", encoding="utf-8"), indent=1)
