# Usage:
#   cd ~/basurin/work
#   source activate.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
echo "Activated: $(python -V)  python=$(python -c 'import sys; print(sys.executable)')"
