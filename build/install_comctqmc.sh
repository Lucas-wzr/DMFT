#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd "$(dirname "$0")/.." && pwd)
solver="$root_dir/thirdparty/ComCTQMC"
mkdir -p "$root_dir/bin"
# Build independently; source the desired Intel MPI/MKL environment first.
mpiicpx -std=c++14 -O2 -DNDEBUG -DHAVE_MPI "$solver/ctqmc/host/AlgebraHost.C" "$solver/ctqmc/host/ctqmc.C" -qmkl -o "$root_dir/bin/CTQMC"
icpx -std=c++14 -O2 -DNDEBUG "$solver/evalsim/EvalSim.C" -qmkl -o "$root_dir/bin/EVALSIM"
install -m 755 "$root_dir/build/comctqmc-launch" "$root_dir/bin/comctqmc-launch"
