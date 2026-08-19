#!/bin/bash
# Downloads IWSLT15 en-vi into data/raw/{train,tst2012,tst2013}.{en,vi}.
# Primary source: GitHub mirror (stdlib only, no extra dependencies).
# The official Stanford host (nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/)
# returns 403 as of this writing
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw

python3 - <<'PY'
import os
import tarfile
import urllib.request

OUT = "data/raw"
MIRROR = "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data"
ARCHIVES = ["train-en-vi.tgz", "dev-2012-en-vi.tgz", "test-2013-en-vi.tgz"]


def from_mirror():
    for name in ARCHIVES:
        path = os.path.join(OUT, name)
        urllib.request.urlretrieve(f"{MIRROR}/{name}", path)
        with tarfile.open(path) as tar:
            tar.extractall(OUT)  # writes train.{en,vi}, tst2012.{en,vi}, tst2013.{en,vi}
        os.remove(path)


def from_hf_fallback():
    # Needs pandas + pyarrow (not in requirements.txt - only used if the
    # mirror above is down). On this source, "validation" and "test" are the
    # same underlying file, unlike the mirror's distinct tst2012/tst2013, so
    # treat this path as a last resort, not an equal substitute.
    import pandas as pd
    base = ("https://huggingface.co/datasets/IWSLT/mt_eng_vietnamese/resolve/"
            "refs%2Fconvert%2Fparquet/iwslt2015-vi-en")
    for hf_split, out_name in [("train", "train"), ("validation", "tst2012"), ("test", "tst2013")]:
        df = pd.read_parquet(f"{base}/{hf_split}/0000.parquet")
        with open(f"{OUT}/{out_name}.vi", "w") as f_vi, open(f"{OUT}/{out_name}.en", "w") as f_en:
            for pair in df["translation"]:
                f_vi.write(pair["vi"] + "\n")
                f_en.write(pair["en"] + "\n")


try:
    from_mirror()
    print("downloaded from GitHub mirror (stefan-it/nmt-en-vi)")
except Exception as e:
    print(f"mirror failed ({e}); falling back to HuggingFace parquet")
    print("needs pandas + pyarrow: pip install pandas pyarrow")
    from_hf_fallback()
PY

echo
wc -l data/raw/*.en data/raw/*.vi
