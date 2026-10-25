# scoremaker — agent notes

Read README.md first: it is the method, including which steps are yours to judge by
looking rather than by trusting a number.

## When a score is finished, file it on the drive

Every finished PDF goes to the band's sheet-music folder on Koala Cloud:

    악보모음/<artist>/<song>.pdf

(browse it at https://drive.koalarepublic.top/?path=band_stuff%2F%EC%95%85%EB%B3%B4%EB%AA%A8%EC%9D%8C)

Upload with:

    python upload.py "<the pdf make_pdf.py wrote>" --artist "<artist>" --song "<song>"

- **Only upload after verifying the score** (bar numbers run 1..N, play order worked
  out, output looked at). The drive is what the band prints from.
- **Names:** use the artist and song as the band would know them, normally in the
  original script (Korean/Japanese stays as is), not the video title. Drop channel
  tags like 【TAB】, "기타 커버", "tab", lesson numbers. If the score is one part of
  several (guitar 1/2, bass, drums), say so in the song name, e.g. `혜성 (베이스)`.
  If the artist or song is genuinely unclear from the video, ask instead of guessing.
- **Reuse existing artist folders.** Run `--dry-run` first, and check the folder listing
  for an artist that is already there under a slightly different spelling.
- **It never overwrites.** A second upload with the same name is saved as
  `<song> (1).pdf`. If you are replacing a score, tell the user so they can delete the
  old one; there is no delete over this endpoint.
- No credentials are needed and none belong in this repo; it is public.
