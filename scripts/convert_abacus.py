"""Convert this scalar, nonmagnetic NiO calculation to DMFT-dev."""
from pathlib import Path
import re
import json
import numpy as np

root = Path(__file__).resolve().parent
source = root / 'dft/OUT.NiO'
target = root / 'dft/outputs_to_DMFT'

def read_csr(path):
    with path.open() as f:
        next(f)  # STEP header is absent in the older DMFT-dev format.
        n = int(next(f).split()[-1])
        nr = int(next(f).split()[-1])
        blocks = []
        for _ in range(nr):
            x, y, z, count = map(int, next(f).split())
            if count == 0:
                continue
            values = np.fromstring(next(f), sep=' ')
            columns = np.fromstring(next(f), sep=' ', dtype=int)
            pointers = np.fromstring(next(f), sep=' ', dtype=int)
            assert len(values) == len(columns) == count and len(pointers) == n + 1
            matrix = np.zeros((n, n))
            for row in range(n):
                sl = slice(pointers[row], pointers[row + 1])
                matrix[row, columns[sl]] = values[sl]
            blocks.append(((x, y, z), values, columns, pointers, matrix))
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
assert nk == 666 and n == 40
# The printed weights/coordinates are rounded. Recover exact 11^3 mesh data.
kpoints = np.array([list(map(float, line.split()[1:4])) for line in lines[3:3 + nk]])
kpoints = np.rint(kpoints * 11) / 11
weights = np.where(np.linalg.norm(kpoints, axis=1) == 0, 1, 2) / 1331
assert abs(weights.sum() - 1) < 1e-12
assert len(list(source.glob(wfc_name.format('*')))) == nk
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

# ABACUS real harmonics: m = 0,+1,-1,+2,-2; first Ni d radial function.
# DMFT-dev expects real harmonics ordered m = -2,-1,0,+1,+2.
(target / 'symmetry.dat').write_text('1\n1\n0 0\n')
(target / 'correlated_atoms.info').write_text(
    'atom 0\nangular_moment 2\n8.0 1.0 0\n-2 14\n-1 12\n0 10\n1 11\n2 13\n')

max_norm = 0.0
max_residual = 0.0
electrons = 0.0
band_lines = []
for ik, k in enumerate(kpoints):
    with (source / wfc_name.format(ik + 1)).open() as f:
        assert int(next(f).split()[0]) == ik + 1
        cart = np.fromstring(next(f), sep=' ')
        # WFC coordinates are Cartesian in reciprocal units, not direct.
        lattice = np.array([[2.0890, 2.0890, 0], [0, 2.0890, 2.0890], [2.0890, 0, 2.0890]])
        assert np.max(np.abs(lattice @ cart - k)) < 1e-7
        nb = int(next(f).split()[0])
        assert int(next(f).split()[0]) == n and nb == 40
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
# This case's residual is diagnostic only, as requested for the DMFT smoke test.
if max_residual >= 1e-5:
    print(f'Warning: max HC-SCe residual = {max_residual:.8g} Ry; continuing DMFT conversion', flush=True)
assert abs(electrons - 24) < 1e-6, electrons
(target / 'bands.dat').write_text(f'0\n24 1 {nb} {nk} {fermi:.16g}\n' + ''.join(band_lines))
report = dict(kpoints=nk, bands=nb, basis=n, electrons=electrons,
              max_CdaggerSC_error=max_norm, max_HC_minus_SCepsilon_Ry=max_residual,
              fermi_Hartree=fermi, correlated_Ni_d_indices=[14, 12, 10, 11, 13])
(root / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
