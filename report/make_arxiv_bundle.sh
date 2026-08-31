#!/usr/bin/env bash
# Build a minimal, self-contained arXiv submission bundle from report/paper.tex.
#
# Includes only paper.tex, its precompiled bibliography (paper.bbl, so arXiv
# doesn't need to run BibTeX), biblio.bib (for completeness), and the figures
# actually referenced via \includegraphics (auto-discovered, so stale or
# unused figure files in report/figures/ are never picked up). Verifies the
# bundle by extracting it into a fresh directory and compiling from scratch.
#
# Usage: ./make_arxiv_bundle.sh [output.tar.gz]
# Default output: report/bees_arxiv.tar.gz

set -euo pipefail

REPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$REPORT_DIR/bees_arxiv.tar.gz}"
case "$OUT" in
  /*) ;;
  *) OUT="$(pwd)/$OUT" ;;
esac

cd "$REPORT_DIR"

echo "Rebuilding paper.bbl from current paper.tex/biblio.bib..."
latexmk -pdf -interaction=nonstopmode paper.tex >/dev/null

STAGE="$(mktemp -d)/bees_arxiv"
mkdir -p "$STAGE/figures"
trap 'rm -rf "$(dirname "$STAGE")"' EXIT

cp paper.tex paper.bbl biblio.bib "$STAGE/"

echo "Discovering figures referenced by \\includegraphics..."
while IFS= read -r path; do
  if [[ -f "$path" ]]; then
    found="$path"
  else
    found=""
    for ext in pdf png jpg jpeg; do
      if [[ -f "$path.$ext" ]]; then
        found="$path.$ext"
        break
      fi
    done
  fi
  if [[ -z "$found" ]]; then
    echo "error: no file found for \\includegraphics{$path}" >&2
    exit 1
  fi
  echo "  $found"
  cp "$found" "$STAGE/figures/"
done < <(grep -oP '\\includegraphics(\[[^]]*\])?\{\K[^}]+' paper.tex)

echo "Verifying the bundle compiles standalone..."
VERIFY="$(mktemp -d)"
trap 'rm -rf "$(dirname "$STAGE")" "$VERIFY"' EXIT
cp -r "$STAGE"/. "$VERIFY/"
(
  cd "$VERIFY"
  for pass in 1 2 3; do
    if ! pdflatex -interaction=nonstopmode paper.tex > "pass$pass.log" 2>&1; then
      echo "error: verification compile failed on pass $pass; see log below" >&2
      tail -40 "pass$pass.log" >&2
      exit 1
    fi
  done
  if grep -q "Label(s) may have changed" pass3.log; then
    echo "error: references did not stabilize after 3 passes" >&2
    exit 1
  fi
)

mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$STAGE" .
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
