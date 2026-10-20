#!/usr/bin/env bash
set -euo pipefail
script=${BASH_SOURCE[0]}
while [[ -L "$script" ]]; do
    directory=$(cd "$(dirname "$script")" && pwd)
    script=$(readlink "$script")
    [[ "$script" = /* ]] || script="$directory/$script"
done
directory=$(cd "$(dirname "$script")" && pwd)
python=${DMFT_PYTHON:-"$directory/../python/bin/python"}
[[ -x "$python" ]] || python=python3
exec "$python" "$directory/abacus_step.py" "$1"
