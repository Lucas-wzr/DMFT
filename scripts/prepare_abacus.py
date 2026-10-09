"""Prepare fixed-geometry charge reference files from a case-local configuration."""
from pathlib import Path
import json
import numpy as np
from abacus_io import input_parameters, lattice_vectors, output_directory

root = Path.cwd()
config = json.loads((root/'abacus_charge.json').read_text())
reference = (root/config.get('reference', 'dft')).resolve()
parameters = input_parameters(reference/'INPUT')
lattice, scale = lattice_vectors(reference/parameters.get('stru_file', 'STRU'))
(root/'abacus_charge_volume.dat').write_text(f'{abs(np.linalg.det(lattice * scale)):.17g}\n')
output = output_directory(reference, parameters)
for key, name, candidates in [
    ('reference_density_matrix', 'abacus_charge_reference.csr',
     ['dmrs1_nao.csr', 'data-DMR-sparse_SPIN0.csr', '1_data-DMR-sparse_SPIN0.csr']),
    ('reference_cube', 'abacus_charge_reference.cube', ['chg.cube', 'SPIN1_CHG.cube'])
]:
    target = root/name
    if target.exists():
        continue
    if key in config:
        source = (root/config[key]).resolve()
    else:
        source = next((output/file for file in candidates if (output/file).exists()), None)
    if source is None or not source.exists():
        raise FileNotFoundError(f'Set {key} in abacus_charge.json')
    target.symlink_to(source.resolve())
