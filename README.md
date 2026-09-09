# scoremaker

Turns a guitar-lesson video into a printable A4 tab score: find the frames where the tab
overlay holds still, composite each of those "pages" into a clean image, work out how the
pages overlap, and re-flow the whole score into justified systems.

Every song has its own directory because every video needs its own constants, but they
are all the same three stages, run in this order from inside the song's directory:

    extract.py    video  -> /tmp/smN/pages/*.npy   one composited page per flip
    stitch.py     pages  -> the score in order     bar boundaries, overlaps resolved
    make_pdf.py   bars   -> ../../<song> 악보.pdf   line breaking and layout

`extract.py` caches its scan and its pages under `/tmp/smN`, so re-running `stitch.py` or
`make_pdf.py` after a tweak costs nothing. Delete the cache to force a re-extract. The
source videos live in the repo root; `ROOT` in each `extract.py` points at it.

## The two families

The thing that decides how much work extraction is: **can you see the video through the
tab overlay?**

### `opaque-panel/` — the panel hides the video

A threshold alone separates the ink. The only moving things are the playhead and any
current-bar highlight, and a temporal statistic over the page's frames removes them:
take the **max** where the ink is dark on a light panel, the **min** where it is light on
a dark one, and the **median** when neither works.

| | tab | panel | how it advances | notes |
|---|---|---|---|---|
| `kaiju/` | bottom | white, dark ink | flips to a **fresh** set of bars, no overlap | temporal max kills the red playhead and the bar highlight for free. Pages just concatenate. *Source video deleted — superseded by `kaiju2`.* |
| `kaiju2/` | bottom | white, dark ink | flips at the **second-to-last** bar, so the trailing partial bar is the next page's first — a **1-bar overlap** | temporal **median**: max would turn every played digit red, min would keep every highlight box. Keep only bars with a barline on both sides and the duplicate drops out. |
| `betelgeuse/` | bottom | white, dark ink | flips a system at a time, **no overlap** | the song is played three times, once per guitar part; the three passes flip at the same offsets, which is what lets them be stacked into one score. |
| `creep/` | bottom | black, white ink | steps ~2 bars at a time, **overlapping** | the layout *switches*: one full-width panel for most of it, two side-by-side panels through the both-guitars chorus, each its own stream of pages. |
| `horizon/` | **top** | light, dark ink | flips with a **variable** overlap (0-2 bars) | the video plays a **repeat** — bars 26-56 twice — so pages split into two passes that have to be merged. Junctions are resolved by overlaying pages at the shifts that line their barlines up. |

### `translucent-overlay/` — the video shows through

The ink has to be separated from a moving performance behind it. Two approaches, and
which one works depends on whether the room moves:

- **temporal min over the page** (`hongyeon`, `jjanggu`) — the tab is static and the
  player is not, so the per-pixel minimum over a page pushes the bleed-through toward its
  darkest, and a top-hat removes what is left.
- **median background subtraction** (`pretender`, `nanmonee`) — when the room barely
  moves, a temporal median across the *whole video* is an excellent estimate of
  everything that is not score, because the score is the only thing that changes every
  page. Subtracting it isolates the ink almost perfectly. The static staff lines go with
  the background and are redrawn from their measured rows.

| | tab | overlay | how it advances | notes |
|---|---|---|---|---|
| `hongyeon/` | bottom half | light, ~50% white | static pages | white top-hat recovers the half-opacity staff lines and chord grids. |
| `jjanggu/` | bottom strip | dark | flips every few seconds | where the guitar body glows through, the overlay's contrast is scaled down, so every mark's top-hat response is divided by a per-column staff-line reference that measures that attenuation exactly. *Source video is outside the repo.* |
| `pretender/` | bottom | dark | flips, and **jumps**: a segno, a D.S. al Coda and practice loops (12x, 8x, 9x) | pages are grouped into **runs** — stretches flipped through without a jump — and the runs are registered into one **panorama**. Single pages cannot be placed: the riff repeats enough that one page matches several positions. |
| `nanmonee/` | bottom | dark | flips **straight through**, ~4 of the 7 bars a page shows | chord symbols sit *above* the panel over live video, where white text on the guitar's white body has almost no contrast; ink is divided by the headroom the background left it (255 − background) to recover them. |

## Composites: percentile, not median

For a translucent overlay, what to combine the page's frames with matters more than it
looks. A median leaves a grey smear wherever the guitar drifted. A **low percentile**
(5th–10th of ~30 frames) does not: the engraving is in *every* frame of a page, while the
fretting hand, the neck under vibrato and the playhead are each in only some. The same
trick removes the playhead for free, so it never has to be detected.

## Joining pages

Whatever the overlay, the score has to be put back together without duplicating or
dropping a bar. In rough order of how much machinery it takes:

1. **Concatenate** — pages do not overlap at all (`kaiju`, `betelgeuse`).
2. **Complete bars only** — pages overlap by exactly the partial bar at the edge, so
   keeping bars that have a barline on both sides is enough (`kaiju2`).
3. **Measure the junction** — overlay consecutive pages at the shifts that align their
   barlines and take the best; the overlap in bars falls out (`horizon`, `creep`).
4. **Panorama** — register every page into one absolute coordinate system and take the
   per-pixel median of everything covering a column. Needed when the page order is not
   monotonic, and worth it anyway: each stretch of score is seen by several pages, and
   residue that survived extraction sits somewhere different on each (`pretender`,
   `nanmonee`).

For (3) and (4), compare masks **against each other's dilation** so 1px registration
jitter does not count. Without that, a true flip and a jump to a repeat sign score close
enough together to be confused; with it, flips land under 0.05 and jumps above 0.14.

## Layout

`make_pdf.py` re-engraves the line breaks rather than reusing the video's: a DP picks the
breaks that minimise the squared deviation from a target line width, then each line is
stretched horizontally to the exact content width. Because the breaks are balanced, that
stretch stays within a few percent of the fixed vertical scale, so nothing visibly
distorts. The scale itself comes from `LINES_PER_PAGE` — choose how many systems a page
should hold and the target width follows.

Two things worth keeping when adapting it:

- **Force a system break at repeat signs** so a repeat opens a system instead of being
  buried mid-line. Thin them out if the chart has many (`pretender` has eight).
- **Keep breaks out of marks drawn above the staff.** Anything horizontal that reaches
  across a bar boundary gets sliced: "Da Coda" printed as "Da Cod" with a stray "a"
  opening the next system. Detect the marks as gap-tolerant column runs and charge the DP
  for breaking inside one.

## A note on scrolling

None of these videos actually scrolls smoothly — all nine flip between static pages, even
the ones that look like a continuous scroll at first (`pretender` was checked: frames a
second apart are identical). Verify before assuming: sample the tab band a second apart
and diff it. If it really does scroll, none of the stitching here applies; you would
register every frame into a panorama instead of every page.

## `/tmp` scratch directories

`sm` hongyeon · `sm2` jjanggu · `sm3` kaiju · `sm4` betelgeuse · `sm5` creep ·
`sm6` kaiju2 · `sm7` horizon · `sm8` pretender · `sm9` nanmonee
