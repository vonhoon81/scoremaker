#!/usr/bin/env python3
"""Download a lesson video from YouTube into the repo root, ready for a pipeline.

    python download.py https://www.youtube.com/watch?v=...
    python download.py <url> --with-audio      # mux audio in, to play along with
    python download.py <url> --height 720      # smaller, if 1080p is overkill

Video only by default: the pipelines never read the audio track, only pixels, so pulling
the audio doubles the download for nothing.

Two details that are easy to get wrong:

* YouTube serves the first 20 MiB and then answers 403 unless yt-dlp impersonates a
  browser, which needs `curl_cffi` installed alongside it. Retries, smaller chunks and
  fresh URLs all hit the same wall, and hammering it gets the IP blocked for ten minutes.
  So `--impersonate Chrome` is passed whenever curl_cffi can be imported, and if it
  cannot and the download dies early, the error message says what to install.
* `--windows-filenames` is passed on every platform, not just Windows. yt-dlp otherwise
  sanitises titles differently per OS, and the pipelines refer to their video by its
  exact filename -- so without it the same URL yields a name that works on Linux and a
  different one on Windows.
"""

import argparse
import os
import subprocess
import sys

import common

TEMPLATE = "%(title)s [%(id)s].%(ext)s"
VIDEO_EXTS = (".mp4", ".mkv", ".webm")


def yt_dlp_command():
    """How to invoke yt-dlp, and whether browser impersonation is available.

    Running it as a module of *this* interpreter is what makes the curl_cffi check
    meaningful -- a `yt-dlp` binary elsewhere on PATH may not share our site-packages.
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        from shutil import which
        exe = which("yt-dlp") or which("yt-dlp.exe")
        if not exe:
            raise SystemExit("yt-dlp not found -- pip install yt-dlp curl_cffi")
        return [exe], _has_curl_cffi_for(exe)
    return [sys.executable, "-m", "yt_dlp"], _has_curl_cffi()


def _has_curl_cffi():
    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        return False
    return True


def _has_curl_cffi_for(exe):
    """Ask a standalone yt-dlp whether it has any impersonation target available."""
    try:
        out = subprocess.run([exe, "--list-impersonate-targets"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "chrome" in out.lower()


def run(cmd, capture=False):
    return subprocess.run(cmd, capture_output=True, text=True, check=False) if capture \
        else subprocess.run(cmd, check=False)


def main():
    common.utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url")
    ap.add_argument("--height", type=int, default=1080,
                    help="tallest acceptable video (default 1080)")
    ap.add_argument("--with-audio", action="store_true",
                    help="mux the audio track in as well (needs ffmpeg)")
    ap.add_argument("--force", action="store_true", help="re-download if it already exists")
    args = ap.parse_args()

    base, impersonate = yt_dlp_command()
    fmt = (f"bv*[height<={args.height}]+ba/b[height<={args.height}]" if args.with_audio
           else f"bv*[height<={args.height}][ext=mp4]/bv*[height<={args.height}]"
                f"/b[height<={args.height}]")
    common_args = ["--windows-filenames", "--no-playlist", "-f", fmt]
    if impersonate:
        common_args += ["--impersonate", "Chrome"]
    else:
        print("note: curl_cffi is not installed, so yt-dlp cannot impersonate a browser.\n"
              "      YouTube often cuts these downloads off at 20 MiB with HTTP 403.\n"
              "      If that happens: pip install curl_cffi\n")

    # Ask yt-dlp for the filename first. It costs one metadata request, tells the caller
    # exactly what to put in a pipeline's VIDEO constant, and lets an already-downloaded
    # file be skipped. It has to be `--print filename` against the real -o template:
    # `--print "%(title)s..."` hands back the *unsanitised* title, and a title containing
    # a slash then reads as a directory separator and buries the download in a new
    # subdirectory named after half the title.
    template = os.path.join(common.ROOT, TEMPLATE)
    probe = run(base + common_args
                + ["-o", template, "--print", "filename", "--print", "id",
                   "--simulate", args.url], capture=True)
    if probe.returncode != 0:
        sys.stderr.write(probe.stderr)
        raise SystemExit("could not read the video's metadata")
    out = probe.stdout.strip().splitlines()
    path, video_id = out[-2], out[-1]
    name = os.path.basename(path)

    if not args.force:
        # match on the id, not the name: files fetched with older yt-dlp settings can
        # have a slightly different title, and re-downloading 200 MB to find that out
        # would be rude
        have = [f for f in os.listdir(common.ROOT)
                if f"[{video_id}]" in f and f.lower().endswith(VIDEO_EXTS)]
        if have:
            print(f"already downloaded: {have[0]}\n  {common.video(have[0])}\n\n"
                  f'In the pipeline\'s extract.py:\n  VIDEO = common.video("{have[0]}")')
            return
    sys.stdout.flush()
    if run(base + common_args + ["-o", template, args.url]).returncode != 0:
        raise SystemExit("download failed")

    size = os.path.getsize(path) / 1e6
    print(f"\ndownloaded {size:.0f} MB\n  {path}\n\n"
          f'In the new pipeline\'s extract.py:\n  VIDEO = common.video("{name}")\n\n'
          "Next: python probe.py to work out the layout.")


if __name__ == "__main__":
    main()
