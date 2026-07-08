#!/usr/bin/env python3
# INSTRUMENT (data tooling), never imported by artifact code. Uses only the
# Python stdlib (urllib/hashlib/re) — no numpy/pandas/ML libs.
#
# Stage-1 corpus builder: download curated public-domain Project Gutenberg
# texts (children's / simple English prose), strip Gutenberg boilerplate,
# normalize to UTF-8 (LF), exact-dedup, and write a provenance manifest.
#   data/raw/    downloaded originals  (gitignored)
#   data/clean/  boilerplate-stripped  (gitignored)
#   data/manifest.json  provenance (tracked)
# Usage: python3 data/build_corpus.py 2026-07-07
import sys, os, re, json, hashlib, time, urllib.request

DATE = sys.argv[1] if len(sys.argv) > 1 else "unknown"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW, CLEAN = os.path.join(HERE, "raw"), os.path.join(HERE, "clean")
os.makedirs(RAW, exist_ok=True); os.makedirs(CLEAN, exist_ok=True)
LICENSE = "Public domain in the US; distributed under the Project Gutenberg License (gutenberg.org/license). Header/footer stripped for training; see manifest."

# (id, title) — curated PD, simple/children's-leaning. Failures are skipped.
BOOKS = [
    (11,"Alice's Adventures in Wonderland"),(12,"Through the Looking-Glass"),
    (55,"The Wonderful Wizard of Oz"),(74,"The Adventures of Tom Sawyer"),
    (76,"Adventures of Huckleberry Finn"),(113,"The Secret Garden"),
    (146,"A Little Princess"),(236,"The Jungle Book"),(271,"Black Beauty"),
    (289,"The Wind in the Willows"),(45,"Anne of Green Gables"),
    (2591,"Grimms' Fairy Tales"),(16,"Peter Pan"),(514,"Little Women"),
    (120,"Treasure Island"),(902,"The Happy Prince and Other Tales"),
    (1342,"Pride and Prejudice"),(158,"Emma"),(64317,"The Great Gatsby"),
]

def fetch(bid):
    urls = [f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
            f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt"]
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent":"loom-corpus/1.0 (research; polite)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return u, r.read().decode("utf-8", "replace")
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    return None, last

START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I|re.S)
END   = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I|re.S)

def strip_boiler(text):
    m1, m2 = START.search(text), END.search(text)
    body = text[m1.end():m2.start()] if (m1 and m2) else text
    body = body.replace("\r\n","\n").replace("\r","\n")
    # drop leading "Produced by ..." transcriber line block if present
    body = re.sub(r"^\s*Produced by .*?\n\n", "", body, flags=re.S)
    return body.strip() + "\n"

def sha256(b): return hashlib.sha256(b).hexdigest()

def main():
    manifest, seen, total_raw, total_clean = [], set(), 0, 0
    for bid, title in BOOKS:
        u, text = fetch(bid)
        if u is None:
            print(f"SKIP {bid} {title}: {text}"); continue
        raw_bytes = text.encode("utf-8")
        open(os.path.join(RAW, f"{bid}.txt"),"wb").write(raw_bytes)
        clean = strip_boiler(text)
        cb = clean.encode("utf-8")
        h = sha256(cb)
        if h in seen:
            print(f"DEDUP {bid} {title}: identical content, dropped"); continue
        seen.add(h)
        open(os.path.join(CLEAN, f"{bid}.txt"),"wb").write(cb)
        total_raw += len(raw_bytes); total_clean += len(cb)
        manifest.append({
            "id": bid, "title": title, "source_url": u,
            "license": LICENSE, "bytes_raw": len(raw_bytes), "bytes_clean": len(cb),
            "sha256_clean": h, "download_date": DATE,
            "filters": ["gutenberg-boilerplate-strip(START/END markers)","crlf->lf","strip-produced-by","utf-8"],
        })
        print(f"OK   {bid} {title}: raw {len(raw_bytes)}B clean {len(cb)}B")
        time.sleep(1.0)  # be polite to Gutenberg
    out = {
        "_meta": {"stage":"1","purpose":"Day-5 milestone pretrain corpus (public-domain English, simple/children's-leaning).",
                  "built": DATE, "builder":"data/build_corpus.py (instrument, stdlib only)",
                  "note":"data/raw and data/clean are gitignored (bytes not tracked); this manifest is the tracked, reproducible record. chars-per-token reported after the tokenizer (Day 5d)."},
        "totals": {"docs": len(manifest), "bytes_raw": total_raw, "bytes_clean": total_clean,
                   "mib_clean": round(total_clean/1048576,2)},
        "documents": manifest,
    }
    json.dump(out, open(os.path.join(HERE,"manifest.json"),"w"), indent=2)
    print(f"\nSTAGE1 docs={len(manifest)} clean_bytes={total_clean} ({out['totals']['mib_clean']} MiB)")

if __name__ == "__main__":
    main()
