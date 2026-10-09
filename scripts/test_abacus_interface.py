"""Small file-interface tests; no ABACUS or QMC calculation is launched."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts))
from abacus_io import k_mesh, match_k_mesh, prepare_density_reference


class InterfaceTests(unittest.TestCase):
    def test_native_density_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'native.csr'
            target = Path(folder)/'reference.csr'
            source.write_text(' --- Ionic Step 1 ---\n1 # number of spin directions\n'
                              '1 # spin index\n2 # number of localized basis\n'
                              '3 # number of Bravais lattice vector R\n'
                              'user_defined_lattice\n1\n0 0 0\n'
                              '# CSR Format\n0 0 0 2\n# values\n1.25 -0.5\n'
                              '# columns\n0 1\n# pointers\n0 1 2\n')
            prepare_density_reference(source, target)
            self.assertEqual(target.read_text().splitlines()[:3],
                             ['STEP: 1', 'Matrix Dimension: 2', 'Matrix number: 1'])
            self.assertEqual(target.read_text().splitlines()[4], '1.25 -0.5 0 1 0 1 2')

    def test_meshes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'KPT'
            for mode in ['Gamma', 'MP']:
                for size in [3, 4]:
                    path.write_text(f'K_POINTS\n0\n{mode}\n{size} 2 1 0 0 0\n')
                    points, weights = k_mesh(path, np.eye(3))
                    recovered, full_weights = match_k_mesh(points, points, weights)
                    np.testing.assert_allclose(recovered, points)
                    np.testing.assert_allclose(full_weights, weights)
                    chosen = []
                    for k in points:
                        if not any(np.max(np.abs(k+p-np.rint(k+p))) < 1e-9 for p in chosen):
                            chosen.append(k)
                    recovered, reduced_weights = match_k_mesh(np.array(chosen), points, weights)
                    self.assertAlmostEqual(reduced_weights.sum(), 1)
                    # Reproduce finite-precision ABACUS wavefunction coordinates.
                    printed = np.array(chosen) + 1.4e-8
                    recovered, printed_weights = match_k_mesh(printed, points, weights)
                    np.testing.assert_allclose(recovered, chosen, atol=1e-15)
                    np.testing.assert_allclose(printed_weights, reduced_weights)
            path.write_text('K_POINTS\n2\nDirect\n0 0 0 1\n0.5 0 0 3\n')
            points, weights = k_mesh(path, np.eye(3))
            np.testing.assert_allclose(weights, [0.25, 0.75])
            with self.assertRaises(ValueError):
                match_k_mesh(points[:1], points, weights)
            path.write_text('K_POINTS\n0\nGamma\n3 1 1 0.2 0 0\n')
            points, weights = k_mesh(path, np.eye(3))
            with self.assertRaises(ValueError):
                match_k_mesh(points, points, weights)

    def test_conversion_and_bridge(self):
        # Different lattice, suffix, electron count and f-shell metadata.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root/'reference'
            output = reference/'OUT.Ce_test'
            output.mkdir(parents=True)
            lattice = np.array([[2., .2, 0], [0, 3., 0], [0, 0, 4.]])
            stru = 'LATTICE_CONSTANT\n1.5\nLATTICE_VECTORS\n'
            stru += '\n'.join(' '.join(map(str,row)) for row in lattice)+'\n'
            (reference/'STRU').write_text(stru)
            (reference/'KPT').write_text('K_POINTS\n0\nGamma\n3 1 1 0 0 0\n')
            (reference/'INPUT').write_text('INPUT_PARAMETERS\nsuffix Ce_test\nnspin 1\nnelec 4\nsymmetry 0\nnbands 8\n')
            n = 8
            energies = np.arange(n)*0.1-0.2
            def csr(name, diagonal):
                indices = np.flatnonzero(diagonal)
                text = f'STEP: 1\nMatrix Dimension: {n}\nMatrix number: 1\n0 0 0 {len(indices)}\n'
                text += '# values\n'+'\n'.join(str(diagonal[i]) for i in indices)+'\n'
                text += '# columns\n'+' '.join(map(str,indices))+'\n'
                pointers = [0]+list(np.cumsum(diagonal != 0))
                text += '# pointers\n'+' '.join(map(str,pointers))+'\n'
                (output/name).write_text(text)
            csr('srs1_nao.csr', np.ones(n))
            csr('hrs1_nao.csr', energies)
            csr('dmrs1_nao.csr', np.ones(n))
            (output/'running_scf.log').write_text('#SCF IS CONVERGED#\nE_Fermi 0.0\n')
            (output/'KPT.info').write_text('nkstot now = 2\nDIRECT\nk x y z weight\n1 0 0 0 0.3333\n2 0.33333333 0 0 0.6667\n')
            for ik,(k,weight) in enumerate([(np.zeros(3),1/3),(np.array([1/3,0,0]),2/3)]):
                text = f'{ik+1} (index)\n'+' '.join(f'{v:.17g}' for v in np.linalg.solve(lattice,k))+'\n8 (bands)\n8 (basis)\n'
                for ib in range(n):
                    text += f'{ib+1} (band)\n{energies[ib]:.17g} (Ry)\n{2*weight if ib<2 else 0:.17g} (Occupation)\n'
                    text += ' '.join(f'{int(i==ib)} 0' for i in range(n))+'\n'
                (output/f'wfk{ik+1}_nao.txt').write_text(text)
            (output/'chg.cube').write_text('mock cube\n')
            dft = root/'dft'
            dft.mkdir()
            for name in ['STRU','KPT','INPUT']:
                shutil.copy2(reference/name,dft/name)
            (dft/'OUT.Ce_test').symlink_to(output)
            metadata = root/'metadata'
            metadata.mkdir()
            info = 'atom 0\nangular_moment 3\n6.0 0.7 0\n'
            info += ''.join(f'{m} {m+3}\n' for m in range(-3,4))
            (metadata/'correlated_atoms.info').write_text(info)
            (metadata/'symmetry.dat').write_text('1\n1\n0 0\n')
            (dft/'outputs_to_DMFT').symlink_to(metadata)
            subprocess.run([sys.executable,str(scripts/'convert_abacus.py')],cwd=root,check=True,stdout=subprocess.DEVNULL)
            report = json.loads((root/'conversion_report.json').read_text())
            self.assertEqual(report['basis'],8)
            self.assertEqual(report['electrons'],4)
            self.assertAlmostEqual(report['cell_volume_bohr3'],81)
            self.assertLess(report['max_HC_minus_SCepsilon_Ry'],1e-12)
            np.testing.assert_allclose(np.loadtxt(metadata/'k_weight.dat',skiprows=1)[:,1],[1/3,2/3])
            stage = root/'abacus_charge/step1'
            stage.mkdir(parents=True)
            # Replace only the expensive external executable, preserving the bridge.
            command = [sys.executable,'-c',f'import shutil; shutil.copytree({str(output)!r}, "OUT.Ce_test")']
            (root/'abacus_charge.json').write_text(json.dumps({'reference':str(reference),'command':command}))
            subprocess.run([sys.executable,str(scripts/'prepare_abacus.py')],cwd=root,check=True)
            self.assertAlmostEqual(float((root/'abacus_charge_volume.dat').read_text()),81)
            self.assertEqual((root/'abacus_charge_reference.csr').resolve(),(output/'dmrs1_nao.csr').resolve())
            subprocess.run([sys.executable,str(scripts/'abacus_step.py'),'1'],cwd=root,check=True,stdout=subprocess.DEVNULL)
            self.assertEqual((stage/'charge.cube').read_text(),'mock cube\n')
            self.assertEqual((dft/'outputs_to_DMFT/correlated_atoms.info').read_text(),info)
            self.assertEqual((stage/'dft/outputs_to_DMFT/bands.dat').read_text().splitlines()[1].split()[:4],['4','1','8','2'])
            self.assertEqual((dft/'outputs_to_DMFT').resolve(),(stage/'dft/outputs_to_DMFT').resolve())


if __name__ == '__main__':
    unittest.main()
