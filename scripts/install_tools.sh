#!/usr/bin/env bash
# Install the two optional command-line tools PhiEx uses for its real
# (non-stub) pipeline, INTO THE ACTIVE CONDA ENV ONLY:
#
#   foldseek   structural-homology search (stage 2)       → bioconda, or the
#              official static binary if bioconda has no build for this OS/arch
#   prank      P2Rank pocket prediction (stage 3), Java   → GitHub release tarball
#              + openjdk from conda-forge
#
# Works on macOS (Apple Silicon or Intel) and x86_64 Linux (e.g. MSI).
# Nothing is written outside ./tools and $CONDA_PREFIX.
#
#   cd ~/PhiEx && conda activate PhiEx && bash scripts/install_tools.sh
#   make weights            # then downloads the Foldseek PDB database (~few GB)
#
# Override the P2Rank version with  P2RANK_VERSION=2.5 bash scripts/install_tools.sh
set -euo pipefail

P2RANK_VERSION="${P2RANK_VERSION:-2.4.2}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$HERE/tools"

if [[ -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "PhiEx" ]]; then
  echo "!! activate the PhiEx env first:  conda activate PhiEx" >&2
  exit 1
fi
CONDA=$(command -v mamba || command -v conda)
mkdir -p "$TOOLS"
echo "==> env: $CONDA_PREFIX   tools dir: $TOOLS   OS: $(uname -s) $(uname -m)"

# ── Foldseek ─────────────────────────────────────────────────────────────
if command -v foldseek >/dev/null; then
  echo "==> foldseek already on PATH: $(command -v foldseek)"
else
  echo "==> installing foldseek from bioconda"
  if ! "$CONDA" install -y -n PhiEx -c conda-forge -c bioconda foldseek; then
    echo "   bioconda failed — using the official static build"
    case "$(uname -s)-$(uname -m)" in
      Darwin-*)        url=https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz ;;
      Linux-x86_64)    url=https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz ;;
      Linux-aarch64)   url=https://mmseqs.com/foldseek/foldseek-linux-arm64.tar.gz ;;
      *) echo "!! no prebuilt foldseek for this platform" >&2; exit 1 ;;
    esac
    curl -fL "$url" -o "$TOOLS/foldseek.tar.gz"
    tar xzf "$TOOLS/foldseek.tar.gz" -C "$TOOLS"
    ln -sf "$TOOLS/foldseek/bin/foldseek" "$CONDA_PREFIX/bin/foldseek"
  fi
fi
foldseek version || true

# ── P2Rank (needs Java) ──────────────────────────────────────────────────
if ! command -v java >/dev/null; then
  echo "==> installing openjdk 17 into the env"
  "$CONDA" install -y -n PhiEx -c conda-forge "openjdk=17"
fi
if command -v prank >/dev/null; then
  echo "==> prank already on PATH: $(command -v prank)"
else
  echo "==> installing P2Rank $P2RANK_VERSION"
  curl -fL "https://github.com/rdk/p2rank/releases/download/${P2RANK_VERSION}/p2rank_${P2RANK_VERSION}.tar.gz" \
       -o "$TOOLS/p2rank.tar.gz"
  tar xzf "$TOOLS/p2rank.tar.gz" -C "$TOOLS"
  # a wrapper, not a symlink: the prank launcher locates its jars relative
  # to its own directory
  cat > "$CONDA_PREFIX/bin/prank" <<WRAP
#!/usr/bin/env bash
exec "$TOOLS/p2rank_${P2RANK_VERSION}/prank" "\$@"
WRAP
  chmod +x "$CONDA_PREFIX/bin/prank"
fi
prank -v 2>/dev/null | head -2 || prank help 2>/dev/null | head -2 || true

echo
echo "==> done.  Next:  make weights   (downloads the Foldseek PDB database)"
echo "    then restart  make run  — the registry panel should show foldseek/p2rank"
echo "    without the STUB tag."
