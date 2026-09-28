"""Download the public datasets into data/.

    python scripts/download_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"

NAVAL = "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip"
CMAPSS = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"


def fetch(url):
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r:
        return zipfile.ZipFile(io.BytesIO(r.read()))


def main():
    (ROOT / "naval").mkdir(parents=True, exist_ok=True)
    z = fetch(NAVAL)
    name = next(n for n in z.namelist() if n.endswith("data.txt") and "MACOSX" not in n)
    (ROOT / "naval" / "data.txt").write_bytes(z.read(name))

    (ROOT / "cmapss").mkdir(parents=True, exist_ok=True)
    outer = fetch(CMAPSS)
    inner = zipfile.ZipFile(io.BytesIO(outer.read(next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip")))))
    for n in inner.namelist():
        if n.endswith(".txt") and not n.startswith("readme"):
            (ROOT / "cmapss" / Path(n).name).write_bytes(inner.read(n))
    print(f"done -> {ROOT}")


if __name__ == "__main__":
    main()
