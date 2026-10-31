#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$package_root/report"
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
