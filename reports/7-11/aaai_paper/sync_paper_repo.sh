#!/bin/bash
# Sync the LaTeX source from this monorepo into the standalone paper repo.
#
#   monorepo : wm/reports/7-11/aaai_paper/paper/{main.tex,sections/,figures/,...}
#   paper repo: <root>/{main.tex,sections/,figures/,...}   (flat -- no wrapper dir)
#
# Two sessions edit this paper in parallel, so the script never force-pushes and
# never deletes on the far side: it pulls first, copies, requires a clean compile
# in the paper repo, shows the diff, and only then commits. Figures are copied
# only if the .tex actually references them, so superseded renders and the .png
# copies used for slides stay out of the paper repo.
#
# Usage: ./sync_paper_repo.sh [-y]      (-y = skip the confirmation prompt)
set -euo pipefail
YES=${1:-}

SRC=$(cd "$(dirname "$0")/paper" && pwd)
REPO_URL=git@github.com-xuustc99:XuUSTC99/Physics-Is-Already-There-Rethinking-Physical-Injection-in-Latent-World-Models-1-.git
WORK=${TMPDIR:-/tmp}/paperrepo_sync
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null

rm -rf "$WORK"
git clone -q "$REPO_URL" "$WORK"
cd "$WORK"

# --- copy sources ----------------------------------------------------------
cp "$SRC"/main.tex "$SRC"/math_commands.tex "$SRC"/references.bib .
[ -f "$SRC/supplementary.tex" ] && cp "$SRC/supplementary.tex" .
cp "$SRC"/aaai2026.sty "$SRC"/aaai2026.bst . 2>/dev/null || true
mkdir -p sections figures
cp "$SRC"/sections/*.tex sections/

# only figures the .tex references, and only .pdf (the .png are slide renders)
for f in "$SRC"/figures/*.pdf; do
  stem=$(basename "$f" .pdf)
  if grep -qF "$stem" sections/*.tex main.tex 2>/dev/null; then cp "$f" figures/; fi
done

# --- must compile before it may be committed -------------------------------
for i in 1 2; do pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true; done
bibtex main >/dev/null 2>&1 || true
for i in 1 2; do pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true; done
if ! grep -q "Output written on main.pdf" main.log; then
  echo "REFUSING TO PUSH: paper repo does not compile after sync"; tail -25 main.log; exit 1
fi
python3 - <<'PY'
import re
log = open("main.log", errors="ignore").read()
pages = re.search(r"Output written on main\.pdf \((\d+) pages", log).group(1)
und = log.lower().count("undefined")
print(f"  compile ok: {pages} pages, {und} undefined refs")
raise SystemExit(1 if und else 0)
PY

git add -A
if git diff --cached --quiet; then echo "  nothing to sync"; exit 0; fi
echo; git diff --cached --stat; echo
if [ "$YES" != "-y" ]; then
  read -r -p "commit and push to the paper repo? [y/N] " a
  [ "$a" = "y" ] || { echo "aborted"; exit 1; }
fi
# Default the paper-repo commit message to the monorepo's latest commit
# subject, so the two repos read the same. Override with MSG=... if needed.
: "${MSG:=$(git -C "$SRC" log -1 --format='%s' 2>/dev/null)}"
: "${MSG:=Sync paper source from the monorepo}"
git -c user.name="XuUSTC99" -c user.email="noreply@github.com" \
    commit -q -m "$MSG"
git push -q origin main
echo "  pushed: $(git log -1 --oneline)"
