# stam-checker

Minimal offline Torah text anomaly highlighter for handwritten STaM-style images.

## What it does

The tool accepts:

- a phone photo or scan of Torah text (`.jpg` / `.png`)
- a plain-text Hebrew reference file

It produces:

- an annotated image with likely anomalies highlighted
- a console summary with the number and type of suspected issues

The pipeline is intentionally simple and deterministic:

1. load the image
2. preprocess with grayscale + blur + adaptive thresholding
3. detect text lines with a horizontal projection
4. segment character-like regions with connected components
5. build simple templates from the observed glyphs and compare them to reference-backed templates
6. compare the recognized sequence to the reference text
7. draw highlights on the original image

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

```bash
python -m src.main --image input.jpg --ref text.txt --out output.jpg
```

Optional debug output:

```bash
python -m src.main --image input.jpg --ref text.txt --out output.jpg --debug
```

When `--debug` is provided, intermediate images are saved next to the output file.

## Output categories

- `missing_character`
- `extra_character`
- `substitution_suspicious`

Color hints in the annotated image:

- red: likely missing character
- orange: likely extra character
- blue: suspicious substitution
- yellow: spacing-driven missing-character suspicion

## Limitations

- This is **not** a halachic validation tool.
- This is **not** a reliable OCR system.
- It only highlights likely anomalies for human review.
- All output requires human interpretation.

## Project layout

```text
stam-checker/
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── segment.py
│   ├── match.py
│   ├── compare.py
│   ├── draw.py
│   └── utils.py
├── data/
│   ├── samples/
│   └── reference/
├── tests/
├── requirements.txt
└── README.md
```
