"""Convert non-SOC, nspin=1 ABACUS output to the DMFT-dev interface."""
from pathlib import Path
import re
import json
import numpy as np
from abacus_io import input_parameters, lattice_vectors, output_directory, k_mesh, match_k_mesh

root = Path.cwd()
dft = root/'dft'
parameters = input_parameters(dft/'INPUT')
if int(parameters.get('nspin', '1')) != 1 or parameters.get('lspinorb', '0').lower() in ['1', 'true', 't']:
    raise ValueError('This interface supports non-SOC nspin=1 calculations')
if parameters.get('gamma_only', '0').lower() in ['1', 'true', 't']:
    raise ValueError('Set gamma_only 0 to output the complex k-space wavefunction format')
source = output_directory(dft, parameters)
target = dft/'outputs_to_DMFT'
lattice, lattice_scale = lattice_vectors(dft/parameters.get('stru_file', 'STRU'))

def read_csr(path):
    lines = [line.split('#')[0].strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    if lines[0].startswith('STEP'):
        lines = lines[1:]
    n, nr = int(lines[0].split()[-1]), int(lines[1].split()[-1])
    tokens = iter(' '.join(lines[2:]).split())
    blocks = []
    for _ in range(nr):
        r = tuple(int(next(tokens)) for _ in range(3))
        count = int(next(tokens))
        if count == 0:
            continue
        values = np.array([float(next(tokens)) for _ in range(count)])
        columns = np.array([int(next(tokens)) for _ in range(count)])
        pointers = np.array([int(next(tokens)) for _ in range(n + 1)])
        matrix = np.zeros((n, n))
        for row in range(n):
            sl = slice(pointers[row], pointers[row + 1])
            matrix[row, columns[sl]] = values[sl]
        blocks.append((r, values, columns, pointers, matrix))
    return n, blocks

# ABACUS develop renamed these output files; numerical formats are unchanged.
modern_output = (source / 'srs1_nao.csr').exists()
s_file = 'srs1_nao.csr' if modern_output else 'data-SR-sparse_SPIN0.csr'
h_file = 'hrs1_nao.csr' if modern_output else 'data-HR-sparse_SPIN0.csr'
wfc_name = 'wfk{}_nao.txt' if modern_output else 'WFC_NAO_K{}.txt'
n, sblocks = read_csr(source / s_file)
_, hblocks = read_csr(source / h_file)
log = (source / 'running_scf.log').read_text()
assert 'charge density convergence is achieved' in log or '#SCF IS CONVERGED#' in log
if modern_output:
    # E_Fermi columns are Ry and eV; convert the Ry column to Hartree.
    fermi = float(re.findall(r'E_Fermi\s+([-+\d.eE]+)', log)[-1]) / 2
else:
    fermi = float(re.findall(r'EFERMI\s*=\s*([-+\d.eE]+)', log)[-1]) / 27.211386245
lines = (source / ('KPT.info' if modern_output else 'kpoints')).read_text().splitlines()
nk = int(lines[0].split()[-1])
printed_kpoints = np.array([list(map(float, line.split()[1:4])) for line in lines[3:3 + nk]])
# Recover precise coordinates from the wavefunction headers, and exact weights
# from the input integration mesh rather than rounded KPT.info weights.
kpoints = []
for ik in range(nk):
    with (source/wfc_name.format(ik + 1)).open() as f:
        assert int(next(f).split()[0]) == ik + 1
        kpoints.append(lattice @ np.fromstring(next(f), sep=' '))
kpoints = np.array(kpoints)
difference = kpoints - printed_kpoints
assert np.max(np.abs(difference - np.rint(difference))) < 1e-7
full_points, full_weights = k_mesh(dft/parameters.get('kpoint_file', 'KPT'), lattice)
kpoints, weights = match_k_mesh(kpoints, full_points, full_weights)
(target / 'KS_eigenvector').mkdir(parents=True, exist_ok=True)
(target / 'overlap_matrix').mkdir(exist_ok=True)

with (target / 'overlap_matrix/SR.csr').open('w') as f:
    f.write(f'Matrix Dimension of S(R): {n}\nMatrix number of S(R): {len(sblocks)}\n')
    for r, values, columns, pointers, _ in sblocks:
        f.write(' '.join(map(str, (*r, len(values)))) + '\n')
        for array in (values, columns, pointers):
            f.write(' '.join(map(str, array)) + '\n')
with (target / 'overlap_matrix/k_vector.dat').open('w') as f:
    f.write(f'{nk}\n')
    for ik, k in enumerate(kpoints):
        f.write(f'{ik} ' + ' '.join(f'{v:.16g}' for v in k) + '\n')
with (target / 'k_weight.dat').open('w') as f:
    f.write(f'{nk}\n')
    for ik, weight in enumerate(weights):
        f.write(f'{ik} {weight:.16g}\n')

# Correlated atoms and harmonics are selected by the user, not inferred from species.
for name in ['correlated_atoms.info', 'symmetry.dat']:
    path = target/name
    if not path.exists():
        raise FileNotFoundError(f'Provide {path} before conversion')

max_norm = 0.0
max_residual = 0.0
electrons = 0.0
band_lines = []
for ik, k in enumerate(kpoints):
    with (source / wfc_name.format(ik + 1)).open() as f:
        assert int(next(f).split()[0]) == ik + 1
        cart = np.fromstring(next(f), sep=' ')
        # WFC coordinates are Cartesian in reciprocal units, not direct.
        difference = lattice @ cart - k
        assert np.max(np.abs(difference - np.rint(difference))) < 1e-7
        nb = int(next(f).split()[0])
        assert int(next(f).split()[0]) == n
        if ik == 0:
            nband = nb
        assert nb == nband
        energies, occs, coefficients = [], [], []
        for ib in range(nb):
            assert int(next(f).split()[0]) == ib + 1
            energies.append(float(next(f).split()[0]))
            occs.append(float(next(f).split()[0]))
            values = []
            while len(values) < 2 * n:
                values.extend(map(float, next(f).split()))
            a = np.array(values).reshape(n, 2)
            coefficients.append(a[:, 0] + 1j * a[:, 1])
    c = np.array(coefficients).T
    e = np.array(energies)
    # ABACUS uses exp(+i k.R). DMFT-dev forms its conjugate column for projection.
    sk = sum(np.exp(2j * np.pi * np.dot(k, r)) * matrix for r, _, _, _, matrix in sblocks)
    hk = sum(np.exp(2j * np.pi * np.dot(k, r)) * matrix for r, _, _, _, matrix in hblocks)
    max_norm = max(max_norm, float(np.max(np.abs(c.conj().T @ sk @ c - np.eye(nb)))))
    max_residual = max(max_residual, float(np.max(np.abs(hk @ c - (sk @ c) * e))))
    electrons += sum(occs)
    with (target / f'KS_eigenvector/eigenvector{ik}.dat').open('w') as f:
        f.write(f'0\n1 {nb} {n}\n')
        for ib in range(nb):
            band_lines.append(f'0 {ib} {ik} {e[ib] / 2:.16g} {occs[ib] / (2 * weights[ik]):.16g}\n')
            for j in range(n):
                f.write(f'0 {ib} {j} {c[j, ib].real:.16g} {c[j, ib].imag:.16g}\n')

assert max_norm < 1e-5, max_norm
# The bridge applies the KS residual stop threshold; retain the diagnostic here.
if max_residual >= 1e-5:
    print(f'Warning: max HC-SCe residual = {max_residual:.8g} Ry; continuing DMFT conversion', flush=True)
if float(parameters.get('nelec', '-1')) > 0:
    target_electrons = float(parameters['nelec'])
else:
    matches = re.findall(r'(?:AUTOSET number of electrons|number of electrons|NELEC)\s*(?:[:=]\s*)*([-+\d.eE]+)', log, re.I)
    if not matches:
        raise ValueError('Cannot find target electron number; set nelec in INPUT')
    target_electrons = float(matches[-1])
assert abs(electrons - target_electrons) < 1e-6, electrons
(target / 'bands.dat').write_text(f'0\n{target_electrons:.16g} 1 {nb} {nk} {fermi:.16g}\n' + ''.join(band_lines))
report = dict(kpoints=nk, bands=nb, basis=n, electrons=electrons,
              max_CdaggerSC_error=max_norm, max_HC_minus_SCepsilon_Ry=max_residual,
              fermi_Hartree=fermi, target_electrons=target_electrons,
              cell_volume_bohr3=float(abs(np.linalg.det(lattice * lattice_scale))))
(root / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
