"""Input and k-mesh conventions shared by the scalar ABACUS interface."""
from pathlib import Path
from itertools import product
import numpy as np


def input_parameters(path):
    parameters = {}
    for line in Path(path).read_text().splitlines():
        words = line.split('#')[0].split('//')[0].split()
        if len(words) > 1:
            parameters[words[0].lower()] = ' '.join(words[1:])
    return parameters


def lattice_vectors(path):
    lines = [line.split('#')[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    scale = float(lines[lines.index('LATTICE_CONSTANT') + 1])
    start = lines.index('LATTICE_VECTORS') + 1
    lattice = np.array([list(map(float, line.split())) for line in lines[start:start + 3]])
    return lattice, scale


def output_directory(dft, parameters):
    return Path(dft) / ('OUT.' + parameters.get('suffix', 'ABACUS'))


def k_key(k):
    wrapped = np.mod(k, 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=1e-10, rtol=0)] = 0.0
    return tuple(np.round(wrapped, 8))


def k_mesh(path, lattice):
    lines = [line.split('#')[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    count, mode = int(lines[1]), lines[2].lower()
    if count == 0:
        mesh = list(map(int, lines[3].split()[:3]))
        shifts = list(map(float, lines[3].split()[3:6])) or [0.0] * 3
        axes = []
        for size, shift in zip(mesh, shifts):
            indices = np.arange(1, size + 1)
            if mode == 'gamma':
                axes.append((shift + indices - 1) / size)
            elif mode in ['mp', 'monkhorst-pack']:
                axes.append((shift + 2 * indices - size - 1) / (2 * size))
            else:
                raise ValueError('Use a Gamma or MP integration mesh')
        points = np.array(list(product(*axes)))
        weights = np.full(len(points), 1.0 / len(points))
    else:
        data = np.array([list(map(float, line.split()[:4])) for line in lines[3:3 + count]])
        points, weights = data[:, :3], data[:, 3]
        if mode == 'cartesian':
            points = points @ lattice.T
        elif mode != 'direct':
            raise ValueError('Explicit integration k points must be Direct or Cartesian')
        if np.any(weights <= 0):
            raise ValueError('Use integration k points with positive weights, not a band path')
        weights = weights / weights.sum()
    return points, weights


def match_k_mesh(kpoints, full_points, full_weights):
    # Only identity and time reversal may collapse the supplied integration mesh.
    # Other space-group reductions require rotation of the NAO matrices.
    # Wavefunction headers print Cartesian coordinates with limited precision.
    # Compare periodic fractional coordinates rather than rounded dictionary keys.
    tolerance = 1e-7
    def representative(point):
        difference = kpoints - point
        difference -= np.rint(difference)
        matches = np.flatnonzero(np.max(np.abs(difference), axis=1) < tolerance)
        if len(matches) > 1:
            raise ValueError('Duplicate k points modulo reciprocal lattice vectors')
        return int(matches[0]) if len(matches) else None
    weights = np.zeros(len(kpoints))
    exact_points = np.zeros_like(kpoints)
    assigned = np.zeros(len(kpoints), dtype=bool)
    full = {}
    for k, weight in zip(full_points, full_weights):
        key = k_key(k)
        full[key] = full.get(key, 0.0) + weight
        index = representative(k)
        reverse = False
        if index is None:
            index = representative(-k)
            reverse = True
        if index is None:
            raise ValueError('Output mesh has spatial symmetry reduction; use symmetry 0 or -1')
        weights[index] += weight
        if not assigned[index]:
            # Retain the output representative, including its reciprocal-lattice shift.
            point = -k if reverse else k
            exact_points[index] = point + np.rint(kpoints[index] - point)
            assigned[index] = True
    for key, weight in full.items():
        if abs(weight - full.get(k_key(-np.array(key)), 0.0)) > 1e-10:
            raise ValueError('Real DMR requires a time-reversal symmetric integration mesh')
    if not assigned.all():
        raise ValueError('Output k points do not match the input integration mesh')
    return exact_points, weights
