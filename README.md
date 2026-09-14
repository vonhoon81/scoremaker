# scoremaker

Turns a guitar-lesson video into a printable A4 tab score. The lesson videos show the tab
as an overlay that holds still for a few seconds, flips to the next few bars, and repeats.
This finds those still frames, composites each one into a clean image, works out how
consecutive pages overlap, and re-engraves the whole thing into justified systems.

Ten videos have been through it so far. Every one needed different constants, and four
needed a different *method*. The point of this README is that the eleventh should be
faster.

---

## Dependencies

```
pip install numpy pillow yt-dlp curl_cffi
```

plus **ffmpeg** and **ffprobe** on `PATH` — every stage shells out to them.

Keep **yt-dlp** current. A build more than a few months old loses whichever YouTube
player client still serves unsigned URLs and dies at `HTTP Error 403` on every format,
which looks exactly like the `curl_cffi` failure described below but is not. `pip install -U
yt-dlp` fixed it. yt-dlp also now warns that it wants a JavaScript runtime (deno by
default) and that *some formats may be missing* without one; the 1080p video-only format
still came through fine without it.

`download.py` on its own needs only `yt-dlp` and `curl_cffi`: it deliberately does not
pull in numpy or Pillow, so fetching a video works on a bare machine.

| | why |
|---|---|
| `numpy` | all the image maths |
| `pillow` | reading/writing frames, drawing the title block, writing the PDF |
| `yt-dlp` | `download.py` |
| `curl_cffi` | lets yt-dlp impersonate a browser; **without it YouTube serves 20 MiB and then answers HTTP 403.** Retries, smaller chunks and fresh URLs all hit the same wall, and hammering it gets the IP blocked for about ten minutes. **Pin it below 0.16** -- yt-dlp only accepts `0.5.10` and `0.10.x`-`0.15.x`, and a plain `pip install curl_cffi` now lands 0.16, which it rejects with *"Impersonate target chrome is not available"* even though the package is installed. `pip install 'curl_cffi>=0.10,<0.16'` |
| `ffmpeg` / `ffprobe` | frame extraction and metadata |

A CJK font is needed for the title block (Korean and Japanese). Found automatically:
Noto Sans CJK or Nanum Gothic on Linux, Malgun Gothic or Yu Gothic on Windows, Apple SD
Gothic Neo on macOS. Override with `SCOREMAKER_FONT` (and optionally
`SCOREMAKER_FONT_BOLD`) if none of those is installed.

Windows, macOS and Linux all work: `common.py` holds the only platform-specific bits
(where videos live, where the scratch cache goes, which font to use). Scratch goes to the
system temp directory, so `/tmp/sm9` on Linux and `%LOCALAPPDATA%\Temp\sm9` on Windows.
Note that Windows has a 260-character path limit by default and these video filenames are
long — keep the checkout somewhere short like `C:\src\scoremaker`.

---

## Quick start

```
python download.py "https://www.youtube.com/watch?v=..."   # 1080p, video only
python probe.py "<the filename it printed>"                # what layout is this?
```

Then copy the closest existing pipeline, paste in the constants `probe.py` printed, and
run its three stages from inside its own directory:

```
cd translucent-overlay/newsong
python extract.py     # video  -> <temp>/smN/pages/*.npy   one composite per page
python stitch.py      # pages  -> the score, in order, bars located
python make_pdf.py    # bars   -> ../../<song> 악보.pdf
```

`extract.py` caches its scan and its pages, so re-running `stitch.py` or `make_pdf.py`
after a tweak is free. Delete the scratch directory to force a re-extract.

---

## Getting the video, and getting frames out of it

### Downloading

```
python download.py "https://www.youtube.com/watch?v=..."
python download.py <url> --with-audio    # mux audio in, to play along with
python download.py <url> --height 720    # smaller, if 1080p is overkill
python download.py <url> --force         # re-download over an existing file
```

It prints the filename it saved and the exact line to paste into the new pipeline:

```
downloaded 42 MB
  /home/you/scoremaker/【TAB】Some Song [abc123].mp4

In the new pipeline's extract.py:
  VIDEO = common.video("【TAB】Some Song [abc123].mp4")
```

What it does under the hood, and why:

- **Video only, no audio.** The format string is
  `bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]`. Nothing in the pipeline
  reads the audio track — every stage works on pixels — so pulling audio roughly doubles
  the download for nothing. `--with-audio` switches to `bv*+ba/b` if you want to listen
  while reading the score; that needs ffmpeg to mux.
- **1080p matters.** The tab digits are 10–15px tall at 1080p. At 720p a barline is
  sub-pixel and the barline-span test that finds bar boundaries stops working.
- **`--impersonate Chrome`, when curl_cffi is installed.** Without it YouTube serves
  exactly 20 MiB and then answers `HTTP Error 403: Forbidden`. Retries,
  `--http-chunk-size`, fresh signed URLs and DASH fragments all hit the same wall, and
  hammering it gets the IP blocked for about ten minutes. `download.py` runs yt-dlp as a
  module of the *current* interpreter, so importing `curl_cffi` here is a valid test of
  whether yt-dlp will have it; if a standalone `yt-dlp` on `PATH` is used instead it asks
  that binary via `--list-impersonate-targets`. When impersonation is unavailable it says
  so up front rather than dying at 20 MiB.
- **`--windows-filenames` on every platform.** yt-dlp sanitises titles differently per
  OS, and each pipeline refers to its video by exact filename. Without this flag the same
  URL yields one name on Linux and a different one on Windows, and the pipeline breaks
  when moved. (`/` and `|` already become `⧸` and `｜` everywhere; this pins down `:`,
  `?`, `*`, `"`, `<`, `>` too.)
- **Metadata first, and `--print filename` specifically.** It asks yt-dlp for the final
  filename before downloading: one metadata request, and it lets an existing file be
  skipped. It has to be `--print filename` against the real `-o` template --
  `--print "%(title)s [%(id)s].%(ext)s"` hands back the *unsanitised* title, and a title
  containing a slash then reads as a directory separator and buries the download in a new
  subdirectory named after half the title.
- **Already-downloaded videos are matched by id**, not by name, because a video's title
  can change upstream and re-fetching 200 MB to discover that would be rude. Pass
  `--force` to download anyway.
- **The name it prints can be in NFD.** yt-dlp wrote the 하루베이스 video's Korean title with the
  syllables decomposed, so the `VIDEO = common.video("...")` line it suggests cannot be
  retyped by hand — an NFC literal that looks identical will not open the file. Copy the
  bytes, or glob the `[id]` out of the repo root, which is what matching by id already
  does.

A cheap trick when deciding whether a video is worth downloading at all: storyboard
frames (`yt-dlp -f sb0`) come from `i.ytimg.com`, are never throttled, and show the
layout well enough to tell where the tab sits.

### Pulling frames

Every stage shells out to ffmpeg and reads **raw frames over a pipe** rather than writing
PNGs to disk. A scan of a three-minute video at 5fps is ~900 full-width frames; as PNGs
that is a gigabyte of I/O for data used once, and as `rawvideo` it is nothing on disk at
all. The shape used throughout:

```python
cmd = ["ffmpeg", "-v", "error"]
if t0 is not None:
    cmd += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}"]     # a single page's window
cmd += ["-i", VIDEO,
        "-vf", f"fps={fps:.4f},crop={W}:{PANEL_H}:0:{PANEL_Y}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
n = W * PANEL_H * 3
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
while True:
    buf = p.stdout.read(n)
    if len(buf) < n:
        break
    yield np.frombuffer(buf, np.uint8).reshape(PANEL_H, W, 3).max(axis=2)
```

Points worth keeping:

- **`crop` before anything else.** Only the tab band is ever needed, and cropping in
  ffmpeg keeps the pipe (and the numpy arrays) an eighth of the size.
- **`fps=` picks the sample count.** For a page composite, ask for exactly the number of
  samples wanted across the page's span: `fps = SAMPLES / (b - a)`. For a page scan, 5fps
  is plenty — pages last seconds.
- **`-ss` before `-i`** so ffmpeg seeks rather than decoding from the start. It seeks to a
  keyframe, so a window shorter than about 1s can come back with fewer frames than asked
  for; ask for a generous window and index into it instead of assuming a count.
- **Reduce to one channel early** (`.max(axis=2)` for white ink, `.min(axis=2)` for dark
  ink on white) unless colour is actually needed. Kaiju2 keeps colour because its bar
  numbers are red and they are what you navigate a 137-bar score by.
- **Do not downscale to save time.** A playhead is often a 1px line and a barline 1–2px;
  halving the resolution blends them into the panel and they stop being detectable. This
  was learned by trying: at half resolution Kaiju2's highlight box vanished entirely.

---

## Which parts are automated, and which need the agent

This matters more than it might seem. The mechanical parts are genuinely hands-off, but
three or four judgement calls sit in the middle of the pipeline, and every one of them
was originally got wrong by trusting a number instead of looking at the picture.

### Automated — trust these

| step | tool | how reliable |
|---|---|---|
| Download at 1080p, name the file deterministically across OSes | `download.py` | reliable |
| Video metadata, sample frames, temporal-median image | `probe.py` | reliable |
| **Staff geometry** — rows, spacing, line count, top-or-bottom | `probe.py` | exact on 3 of the 5 videos tested; on Creep it locked onto every *other* line and reported half the count at double the spacing, and on the `harrubass` bass tab it found **one** line and gave up. Always check it against `static.png` — see *when the staff finder gives up* below |
| **Panel polarity** — dark ink on light, or light on dark | `probe.py` | exact on all four |
| **Smooth scroll vs page flips** | `probe.py` | decisive. Correlates the band with itself 0.2s later, well inside a page: a shift of 0 means static pages |
| Page compositing, once the constants are right | `extract.py` | reliable |
| Barline detection, bar segmentation | `stitch.py` | reliable, but see *barline nicks* below |
| Overlap between consecutive pages; panorama registration | `stitch.py` | reliable, with the dilated metric below |
| Line breaking, justification, pagination, PDF | `make_pdf.py` | reliable |

### The agent's job — do not automate these

1. **Decide translucent vs opaque by looking.** `probe.py` reports how much moves inside
   the panel, but that number cannot tell a translucent overlay from an opaque panel with
   a big moving highlight: Nanmonee (translucent) reads 9.4 and Kaiju2 (opaque white)
   reads 6.2. Open a sample frame and answer one question — *can you see the guitar
   through the tab?* That single look picks the family, and the two families need
   different extraction code.

2. **Choose the page-segmentation signal when the sweep fails.** `probe.py` sweeps ink
   thresholds and reports the separation between a page flip and a quiet frame. When it
   finds one (Creep: 2499:1) the flip times are trustworthy. When it does not (Nanmonee
   4.2:1, Kaiju2 4.3:1) something moves within a page as much as a flip does, and the
   agent has to find a cleaner signal — subtract the median background first, restrict to
   the staff rows, or diff a mask of the engraving only. This is where most of the
   per-video work actually goes.

3. **Read the bar numbers to verify the assembly.** This is the single most valuable check
   and there is no OCR here. Dump the row just above the staff across the whole assembled
   score, look at it, and confirm the numbers run 1..N with no gaps and no repeats. It
   caught real bugs every time: a panorama that had OR'd two different stretches of score
   together, a missed barline, a page placed at the wrong offset. Independent
   cross-checks: bar count against the last printed number, and total played bars × tempo
   against the video's runtime (Pretender came to 294s of music against a 295s video).

4. **Work out the score's structure and the play order.** Repeat signs, first/second-time
   voltas, a segno, a D.S. al Coda, "Da Coda", multi-bar rests, practice repeat counts
   like "12x". These change what the video's page order *means*: Horizon plays bars 26-56
   twice, Pretender jumps segno→coda→back. The video's page order plus the printed marks
   together give the play order, and only an agent can put those together. Say what was
   inferred and what was read off the page.

5. **Trim the band and judge the output.** How far above the staff to keep (bar numbers?
   chord symbols? section labels? a tempo mark?), whether leftover smears are acceptable,
   whether a label came out too faint, and how big to print it (`LINES_PER_PAGE` trades
   readability against page count). All visual calls.

6. **Name the score and file it.** Also the agent's call.

---

## The two families

The question that decides how much work extraction is: **can you see the video through
the tab overlay?**

### `opaque-panel/` — the panel hides the video

A threshold alone separates the ink. The only moving things are the playhead and any
current-bar highlight, and a temporal statistic over the page's frames removes them: take
the **max** where the ink is dark on a light panel, the **min** where it is light on a
dark one, and the **median** when neither works.

| | tab | panel | how it advances | notes |
|---|---|---|---|---|
| `kaiju/` | bottom | white, dark ink | flips to a **fresh** set of bars, no overlap | temporal max kills the red playhead and the bar highlight for free; pages just concatenate. *Source video deleted — superseded by `kaiju2`.* |
| `kaiju2/` | bottom | white, dark ink | flips at the **second-to-last** bar, so the trailing partial bar is the next page's first — a **1-bar overlap** | temporal **median**: max would turn every played digit red, min would keep every highlight box. Keep only bars with a barline on both sides and the duplicate drops out. 137 bars. |
| `betelgeuse/` | bottom | white, dark ink | flips a system at a time, **no overlap** | the song is played three times, once per guitar part; the passes flip at the same offsets, which is what lets them be stacked into one score. |
| `creep/` | bottom | black, white ink | steps ~2 bars at a time, **overlapping** | the layout *switches*: one full-width panel for most of it, two side-by-side panels through the both-guitars chorus, each its own stream of pages. |
| `horizon/` | **top** | light, dark ink | flips with a **variable** overlap (0-2 bars) | notation staff + Korean lyrics + chords + tab, 438px tall. Plays a **repeat** — bars 26-56 twice — so pages split into two passes that have to be merged. 89 bars. |

### `translucent-overlay/` — the video shows through

The ink has to be separated from a moving performance behind it. Which approach works
depends on whether the room moves:

- **temporal min over the page** (`hongyeon`, `jjanggu`) — the tab is static and the
  player is not, so the per-pixel minimum over a page pushes the bleed-through toward its
  darkest, and a top-hat removes what is left.
- **median background subtraction** (`pretender`, `nanmonee`) — when the room barely
  moves (frames a second apart differing over ~1.5% of pixels), a temporal median across
  the *whole video* is an excellent estimate of everything that is not score, because the
  score is the only thing that changes every page. Subtracting it isolates the ink almost
  perfectly.

| | tab | overlay | how it advances | notes |
|---|---|---|---|---|
| `hongyeon/` | bottom half | light, ~50% white | static pages | white top-hat recovers the half-opacity staff lines and chord grids. |
| `jjanggu/` | bottom strip | dark | flips every few seconds | where the guitar body glows through, the overlay's contrast is scaled down, so every mark's top-hat response is divided by a per-column staff-line reference that measures that attenuation exactly. *Source video is outside the repo.* |
| `pretender/` | bottom | dark | flips, and **jumps**: a segno, a D.S. al Coda and practice loops (12x, 8x, 9x) | pages are grouped into **runs** — stretches flipped through without a jump — and the runs are registered into one **panorama**. Single pages cannot be placed: the riff repeats enough that one page matches several positions. 62 bars written, 58 bar boxes (a five-bar multi-rest). |
| `nanmonee/` | bottom | dark | flips **straight through**, ~4 of the 7 bars a page shows | chord symbols sit *above* the panel over live video. 115 bars. |
| `harrubass/` | bottom | none -- ink straight over the video | flips to a **fresh** page, **no overlap at all** | a four-line **bass** tab, and the only translucent chart here that *concatenates*. The studio is black, so the ink separates almost perfectly and the page split reaches 208:1. No bar numbers and no repeat signs anywhere. 125 bars. **Named for a channel, not a song** — see below. |

---

## What was learned the hard way

### Compositing a page: percentile, not median

For a translucent overlay, what you combine the page's frames *with* matters more than it
looks. A median leaves a grey smear wherever the guitar drifted. A **low percentile**
(5th–10th of ~30 frames) does not: the engraving is in *every* frame of a page, while the
fretting hand, the neck under vibrato and the playhead are each in only some. The same
trick removes the playhead for free, so it never has to be detected at all.

For an opaque panel, pick the temporal statistic by polarity, and think about what else
moves. On Kaiju2 the obvious max would have turned every digit the playhead had passed
over permanently red, because the video colours the currently-sounding digits; the median
was right.

### Staff lines belong to the background

They never move, so a median background subtraction removes them along with the room.
Redraw them from their measured rows — which also gives a crisper printed line. Two
consequences:

- **Barline nicks.** A barline crossing a staff line gets its ink cancelled there too, so
  every barline comes out with 1px gaps and a run-length test fails to see it. Close
  gaps of 1–2px before measuring runs.
- Redraw at `int(y)`, not `round(y)`, so the drawn line sits on the rows the source drew
  it on and covers those gaps.

### White ink over a bright background

How much ink a pixel can possibly show is `255 - background`. Over a dark panel that is
nearly the full range; a chord symbol printed over the guitar's white body has about 45
levels to work with and comes out pale grey at any fixed gain. Dividing the ink by that
headroom recovers those labels without touching the ones over the dark panel. Where the
background is genuinely white the contrast is ~2% and nothing can be recovered — say so
rather than chasing it.

### Joining pages, in order of how much machinery it takes

1. **Concatenate** — pages do not overlap at all (`kaiju`, `betelgeuse`, `harrubass`).
   Not only the opaque family: check every chart for this before assuming otherwise.
2. **Complete bars only** — pages overlap by exactly the partial bar at the edge, so
   keeping bars with a barline on both sides is enough (`kaiju2`). This is what makes the
   apparent "repeat of the last bar at every scroll" disappear.
3. **Measure the junction** — overlay consecutive pages at the shifts that align their
   barlines and take the best; the overlap in bars falls out (`horizon`, `creep`). Do not
   assume a fixed advance: bar widths are content-driven (149–559px on Horizon) so a page
   holds three or four bars and the same bar is sometimes complete on two pages.
4. **Panorama** — register every page into one absolute coordinate system and take the
   per-pixel median of everything covering a column. Needed when the page order is not
   monotonic, and worth it anyway: each stretch of score is seen by several pages, and
   residue that survived extraction sits somewhere different on each (`pretender`,
   `nanmonee`).

Panorama gotchas, both of which produced visibly wrong output first:

- Write **first-write-wins, never OR'd**. A union of two different stretches of score
  poisons every later registration against that region.
- Composite over intervals where the set of contributing pages is **constant**. Padding a
  block with white for the pages that only half cover it lets those white pixels vote in
  the median and greys out everything they touch.
- Inset each page's own edges (~12px) before it contributes: a glyph the page edge cut in
  half would otherwise outvote the pages that show it whole.

### One directory is a channel, not a song

`harrubass/` is the exception to the one-directory-per-song rule. 하루베이스
(`@harrubass`, youtube.com/channel/UCzGjV-2jL1iLZLojqubPygw) numbers its uploads
("405. ...") and uses a single layout across all of them, so the pipeline is the
channel's, and a new video from it changes only a `PER-SONG` block of three constants in
`extract.py` (`VIDEO`, `WORK`, `MUSIC`) plus the title block in `make_pdf.py`. Everything
below that line — band rows, staff geometry, the barline and page-junction rules — is the
channel's template and was measured once.

Worth copying the idea if another source turns out to be a series: the per-video work
collapses to reading one timestamp pair, and the bar-timing table in `stitch.py` is
enough to tell you when a video has broken the template and needs re-measuring.

### Not every translucent chart overlaps — check before reaching for the panorama

The README's own framing invites the mistake: the opaque family concatenates, the
translucent family registers. `harrubass` is translucent and concatenates. Each of its
pages is a self-contained system of complete bars justified to fill the staff, sharing
nothing with its neighbours, and copying Nanmonee's panorama onto it produced garbage.

What makes this worth a section is that **registration does not fail loudly**. It
returned an offset for every consecutive pair and a plausible-looking run structure; the
offsets were just nonsense, alternating between ~0 and a full page width. The tells:

- **every** pair scores badly. Ordinary flips come in under 0.05 and real jumps at 0.14;
  a chart with nothing to match on scores 0.25–0.5 across the board, with no clean
  split between the two populations.
- the non-zero offsets cluster at the page width rather than at a fraction of it.
- consecutive pages carry **no shared bar**. This is the check to run first, and it is
  a look, not a number: put two consecutive pages one above the other and see whether
  any bar appears in both. Here a tie settled it in one glance — the page ends on a tied
  note and the next page opens with that note bracketed as a continuation, *at its first
  bar*, which is exactly what no overlap looks like.

If the pages concatenate, throw the whole `offsets`/`panorama` pair away rather than
special-casing it. What replaces them is `np.hstack` of each page's staff span.

### When the staff finder gives up

`probe.py` wants rows that are uniform full-width lines, which is a fair description of
a staff drawn on a panel and a poor one of a staff drawn straight over a moving video.
On the `harrubass` tab it reported a single line out of four and refused to go on. Two
causes,
both worth recognising:

- **a translucent staff is only uniform where the background is.** The temporal median
  still shows the lines clearly; they just fail `UNIFORM`.
- **`staff_lines` needs four in an even run, and a bass tab only has four**, so one
  missed line takes it below the floor that a six-line guitar staff has slack for.

Measuring them by hand takes a minute and is exact: take the row profile of
`static.png` across the columns the staff spans, and the lines are the rows that spike.
Weight each spike's rows by brightness for a sub-pixel centre. Four spikes came out at
852.4 / 879.0 / 905.6 / 932.2 — dead even at 26.59px, which is itself the confirmation
that the right rows were found. Then hand `PANEL_Y`, `PANEL_H`, `STAFF_*` to the
pipeline and let `probe.py`'s *other* answers (polarity, scroll-vs-flip, the ink sweep)
run off the band you measured, which is what `/tmp/probe2.py`-style throwaway does.

Also worth knowing: the ink sweep can be *much* better than the worked examples suggest.
Against a black studio, `harrubass` separated a page flip from a quiet frame at 208:1,
where Nanmonee managed 4.3:1. A chart that looks hard because it is translucent can be
the easiest one yet.

### Barlines: measure the span as a fraction, not as a run

`barline_cols` originally asked for one unbroken dark run from the top staff line to the
bottom, bridging 1–2px nicks first. On a four-line bass staff the gaps between lines are
26px rather than 18, the engraving is thin, and background subtraction left barlines
broken in two or three places at once — `BRIDGE` closes a 1px gap per pass and cannot
close a 2px one at all, so two thirds of the barlines went undetected and pages came back
with one barline where they plainly had three.

Asking instead that a column be dark over **90% of the staff's rows** found every one,
on every page, with no bridging step. It needs one guard, because a note stem passing
through the staff also darkens most of it:

- **a real barline is 2–9px wide; a stem is exactly 1.** Rejecting single-column groups
  removed every false positive and kept every true one. Measure the group's width, not
  its darkness — the spurious ones were just as dark.

The other half of the guard is free: a stem reaches far below the staff (to row 203+ here)
while a barline stops at it, so extent is a second signal if width is ever not enough.

### Verifying an assembly with no bar numbers on it

The README's best check — read the bar numbers off the assembled score and confirm they
run 1..N — needs the chart to print bar numbers. The 하루베이스 charts print none. The
substitute,
which caught two real findings:

**A page's time on screen is proportional to how many bars it shows.** The display
advances when the bars it is showing have been played, so seconds-per-bar is a constant
of the video, and every page has to agree on it. Print the table and read down the last
column. Here 28 of 32 pages agreed at 1.20–1.35s, which
- fixes the tempo as a by-product: 1.30s/bar is ♩=185 in 4/4, and
- makes the four that disagree mean something specific.

Two pages read 2.55 and 2.60s/bar — exactly double. Those had been held for two page
lengths, so their four bars are *played twice* and the flip between the two showings went
undetected because the two renderings are pixel-identical. That is the play-order finding
for this chart, and nothing on the page says it; only the timing does. One page read 1.85
because the tab appears before the music starts, and one page genuinely shows two bars
rather than four and read a correct 1.30.

Two cross-checks on top, both of which have to agree: total bars written (125) against
seconds of score on screen over seconds per bar (133 — the difference is exactly the two
four-bar stretches played twice), and the last page ending on a real closing double bar
rather than running out of video.

Beware one trap in the table itself: attribute a bar to a page by its **midpoint**, not
its left edge. A page's closing barline can sit a few px short of the page's right edge,
and counting on left edges lends that bar to the wrong page — which showed up as pairs of
pages reading 5 bars and 3 bars where both had four.

### Junction barlines, when the source draws them inconsistently

A chart whose pages concatenate has a barline at every page junction by definition, but
this source draws one at a page's right edge only about half the time and never at the
left. Drawing one in unconditionally is wrong in the other direction: where the source
*did* close the page, its line and the redrawn one sit ~7px apart and print as a double
bar the music does not have. Draw the line only where a search either side of the
junction finds none — and do it in `stitch.py`, which can see both sides, not in
`extract.py`, which renders each page alone.

### Compare masks against each other's dilation

For any page-to-page matching, compare each ink mask against the *other's* 1px dilation so
registration jitter does not count. Without it a true flip and a jump to a repeat sign
score close enough together to be confused; with it, flips land under 0.05 and jumps above
0.14. This is what made Pretender's D.S. detectable at all.

### Register runs, not pages

When a video jumps around the score, a single page cannot be placed reliably — a riff-based
arrangement repeats enough that one page matches several positions. A *run* (a maximal
stretch flipped through without a jump) is 4000–9000px of score and places with no contest.

### Layout

`make_pdf.py` re-engraves the line breaks rather than reusing the video's: a DP picks the
breaks that minimise the squared deviation from a target line width, then each line is
stretched horizontally to the exact content width. Because the breaks are balanced, that
stretch stays within a few percent of the fixed vertical scale, so nothing visibly
distorts. The scale itself comes from `LINES_PER_PAGE`.

- Use **floor, not round**, for the line height. One pixel too tall costs a whole page of
  capacity.
- **Force a system break at repeat signs**, so a repeat opens a system instead of being
  buried mid-line where a reader will miss it. Thin them out if the chart has many
  (Pretender has eight, several a bar apart, and honouring all of them left single-bar
  lines).
- **Keep breaks out of marks drawn above the staff.** Anything horizontal reaching across
  a bar boundary gets sliced: "Da Coda" printed as "Da Cod" with a stray "a" opening the
  next system. Detect marks as gap-tolerant column runs (letters have gaps) and charge the
  DP for breaking inside one. When a repeat sign and a mark conflict, the mark wins.
- **Bar numbers straddle their barline** (roughly −8 to +10 px). Cut bars ~10px to the
  left of the barline or every bar carries half of the *next* bar's number.
- Give the last bar of each line the barline that closes it, and blank the next bar's
  number out of that padding.
- Merge a section's closing double bar and the next system's opening barline (~57px apart)
  into the following bar rather than dropping the gap — it holds the new system's clef.
- Do not print a per-line bar range if the score has multi-bar rests: those make bar
  *boxes* disagree with the numbers printed on the staff, and a wrong label is worse than
  none.

### A note on scrolling

None of the nine videos actually scrolls smoothly — all of them flip between static pages,
including ones that look like a continuous scroll at first. Pretender was checked
frame-by-frame: frames a second apart are identical. `probe.py` settles it with the
correlation test. If a video really does scroll, none of the stitching here applies; you
would register every *frame* into a panorama instead of every page.

---

## Scratch directories

Under the system temp directory: `sm` hongyeon · `sm2` jjanggu · `sm3` kaiju ·
`sm4` betelgeuse · `sm5` creep · `sm6` kaiju2 · `sm7` horizon · `sm8` pretender ·
`sm9` nanmonee · `sm10` harrubass · `probe` probe.py.

## What is not in the repo

Source videos and the generated PDFs. Neither is ours to redistribute, and the largest
video is past GitHub's 100 MB file limit. `download.py` fetches a video back; the scores
are rebuilt by running the three stages.
