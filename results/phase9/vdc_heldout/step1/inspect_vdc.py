# Descriptive inventory only: sheets, shapes, columns, dtypes, 5 example rows.
# No condition-based statistics of any kind. Raw files are read, never modified.
# Some files are "strict" OOXML (purl.oclc.org namespaces) that openpyxl does not list;
# they are converted to the transitional namespaces IN MEMORY before reading.
# Usage: inspect_vdc.py [file.xlsx ...]   (default: all files in Data/)
import sys, os, glob, warnings, pickle, io, zipfile
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 30)
RAW = "./data/external/vdc2024/raw/p562n/osfstorage/Data"
CACHE = "<local path>"
os.makedirs(CACHE, exist_ok=True)
REP = [(b"http://purl.oclc.org/ooxml/spreadsheetml/main", b"http://schemas.openxmlformats.org/spreadsheetml/2006/main"),
       (b"http://purl.oclc.org/ooxml/officeDocument/relationships", b"http://schemas.openxmlformats.org/officeDocument/2006/relationships")]


def open_xl(p):
    zin = zipfile.ZipFile(p)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zout:
        for it in zin.infolist():
            d = zin.read(it.filename)
            if it.filename.endswith((".xml", ".rels")):
                for a, b in REP:
                    d = d.replace(a, b)
            zout.writestr(it.filename, d)
    buf.seek(0)
    return pd.ExcelFile(buf, engine="openpyxl")


if __name__ == "__main__":
    only = sys.argv[1:]
    for p in sorted(glob.glob(RAW + "/*.xlsx")):
        name = os.path.basename(p)
        if only and name not in only:
            continue
        print("=" * 110)
        print("FILE", name, os.path.getsize(p), "bytes", flush=True)
        xl = open_xl(p)
        print("SHEETS:", xl.sheet_names)
        for sh in xl.sheet_names:
            df = xl.parse(sh, header=0)
            pickle.dump(df, open(f"{CACHE}/{name}__{sh}.pkl", "wb"))
            print(f"-- SHEET {sh!r}: {df.shape[0]} data rows x {df.shape[1]} columns")
            print("COLUMNS/DTYPES:")
            for c in df.columns:
                print(f"   {str(c)!r:45} {str(df[c].dtype):10} n_unique={df[c].nunique(dropna=True):<8} n_null={int(df[c].isna().sum())}")
            print("FIRST 5 ROWS:")
            print(df.head(5).to_string())
