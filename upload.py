#!/usr/bin/env python3
"""File a finished score in the band's sheet-music folder on the Koala Cloud drive.

    python upload.py "괴수의 하나우타 기타 악보.pdf" --artist "Vaundy" --song "괴수의 하나우타"
    python upload.py <pdf> --artist <artist> --song <song> --dry-run

It lands at 악보모음/<artist>/<song>.pdf, which is
https://drive.koalarepublic.top/?path=band_stuff%2F%EC%95%85%EB%B3%B4%EB%AA%A8%EC%9D%8C
from the drive's side. The artist folder is created if it does not exist yet.

Two details that are easy to get wrong:

* It goes through the folder's public /bandscore endpoint, which needs no login -- so
  nothing secret lives in this repo. The flip side is that the endpoint never overwrites
  or deletes: uploading the same name twice files the second one as "<song> (1).pdf".
  Check the folder before re-uploading a fixed score, and ask the owner to remove the
  old one rather than leaving two.
* Names are NFC-normalised before sending. yt-dlp writes some Korean titles in NFD, and a
  name copied out of one of those filenames would otherwise create a second artist
  folder that looks identical to the existing one.

Standard library only, so it runs on a bare machine like download.py does.
"""
import argparse
import json
import os
import sys
import unicodedata
import urllib.error
import urllib.request
import uuid

BASE = "https://drive.koalarepublic.top/bandscore"


def clean(name, what):
    name = unicodedata.normalize("NFC", name).strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        sys.exit(f"bad {what} name: {name!r} (no slashes, not empty)")
    return name


def post(pdf, relpath):
    boundary = uuid.uuid4().hex
    with open(pdf, "rb") as f:
        data = f.read()
    # The server takes the stored path from `relpath`, so the file part's own filename
    # can stay ASCII and sidestep how each client encodes non-ASCII there.
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="relpath"\r\n\r\n'
        f"{relpath}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="score.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/upload?p=", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf")
    ap.add_argument("--artist", required=True)
    ap.add_argument("--song", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the destination only")
    a = ap.parse_args()

    if not os.path.isfile(a.pdf):
        sys.exit(f"no such file: {a.pdf}")
    relpath = f"{clean(a.artist, 'artist')}/{clean(a.song, 'song')}.pdf"
    print(f"-> 악보모음/{relpath}")
    if a.dry_run:
        return

    res = post(a.pdf, relpath)
    if not res.get("ok"):
        sys.exit(f"upload failed: {res.get('error', res)}")
    saved = res["path"]
    print(f"uploaded {os.path.getsize(a.pdf) // 1024} KB -> 악보모음/{saved}")
    if saved != relpath:
        print("  (a file with that name was already there, so it was filed under a new name)")


if __name__ == "__main__":
    main()
