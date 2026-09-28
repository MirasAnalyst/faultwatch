"""Download the public datasets into data/.

    python scripts/download_data.py                 # all datasets (~330 MB)
    python scripts/download_data.py naval cmapss    # just some
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(ROOT.parent))

NAVAL = "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip"
CMAPSS = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CWRU = "https://engineering.case.edu/sites/default/files/{}.mat"
TEP = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/{}"
CARE = "https://zenodo.org/api/records/15846963/files/CARE_To_Compare.zip/content"


def get(url) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def fetch_zip(url):
    print(f"downloading {url}")
    return zipfile.ZipFile(io.BytesIO(get(url)))


def naval():
    (ROOT / "naval").mkdir(parents=True, exist_ok=True)
    z = fetch_zip(NAVAL)
    name = next(n for n in z.namelist() if n.endswith("data.txt") and "MACOSX" not in n)
    (ROOT / "naval" / "data.txt").write_bytes(z.read(name))


def cmapss():
    (ROOT / "cmapss").mkdir(parents=True, exist_ok=True)
    outer = fetch_zip(CMAPSS)
    inner = zipfile.ZipFile(io.BytesIO(outer.read(next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip")))))
    for n in inner.namelist():
        if n.endswith(".txt") and not n.startswith("readme"):
            (ROOT / "cmapss" / Path(n).name).write_bytes(inner.read(n))


def cwru():
    from faultwatch.data import CWRU_FILES
    d = ROOT / "cwru"
    d.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(CWRU_FILES)} CWRU recordings")
    for fid in CWRU_FILES:
        if not (d / f"{fid}.mat").exists():
            (d / f"{fid}.mat").write_bytes(get(CWRU.format(fid)))


def tep():
    d = ROOT / "tep"
    d.mkdir(parents=True, exist_ok=True)
    print("downloading Tennessee Eastman runs d00-d21")
    for i in range(22):
        for suffix in ("", "_te"):
            n = f"d{i:02d}{suffix}.dat"
            if not (d / n).exists():
                (d / n).write_bytes(get(TEP.format(n)))


def wind():
    """Only Wind Farm A (~160 MB) is read out of the 5.5 GB CARE archive,
    using HTTP range requests, so the whole archive is never downloaded."""
    from faultwatch.remote_zip import RemoteZip
    d = ROOT / "care" / "wind_farm_a"
    (d / "datasets").mkdir(parents=True, exist_ok=True)
    print("reading Wind Farm A from the CARE to Compare archive (range requests)")
    z = RemoteZip(CARE)
    prefix = "CARE_To_Compare/Wind Farm A/"
    for info in z.infolist():
        if info.filename.startswith(prefix) and not info.is_dir():
            target = d / info.filename[len(prefix):]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))


DATASETS = {"naval": naval, "cmapss": cmapss, "cwru": cwru, "tep": tep, "wind": wind}


def main():
    for name in sys.argv[1:] or DATASETS:
        DATASETS[name]()
    print(f"done -> {ROOT}")


if __name__ == "__main__":
    main()
