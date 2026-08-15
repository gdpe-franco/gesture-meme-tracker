# Cat Gesture Meme Tracker

A browser app that uses MediaPipe to display a cat reaction for each gesture.

## Run

```bash
./run.sh
```

Open `http://localhost:8000`, allow camera access, and show a gesture.

## Reactions

All media lives in `images/cat-reactions/`; gesture detection runs locally in
the browser.

| Pose or expression | Reaction |
| --- | --- |
| Call-me / shaka | Chill |
| Thumbs up | Thumbs-up |
| Point up or at the camera | Pointing-up / pointing-front |
| Closed fist / two closed fists | Punch / Mad |
| Thumb-and-index heart | Heart |
| Two raised open hands / two open hands apart | Desperate / Scared |
| One-sided smile / frown | Confident / Sad |
| Open mouth / mouth covered plus raised moving hand | Tongue-out / Scuba |

No Python packages are required for the web app.

## Run with Docker (Linux)

Build and run the web app:

```bash
make up
```

Or, without Make:

```bash
docker build -t cat-gesture-meme-tracker .
docker run --rm -p 8000:80 cat-gesture-meme-tracker
```

Then open `http://localhost:8000` and allow camera access. Docker only serves
the static files; MediaPipe and webcam access run in the browser.
