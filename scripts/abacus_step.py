"""One scalar ABACUS KS update; the C++ program owns the charge loop."""
from pathlib import Path
import os
import sys
import shutil
import subprocess
import json
import shlex
from abacus_io import input_parameters, output_directory

root = Path.cwd()
config = json.loads((root/'abacus_charge.json').read_text())
reference = (root/config.get('reference', 'dft')).resolve()
parameters = input_parameters(reference/'INPUT')
if int(parameters.get('nspin', '1')) != 1 or parameters.get('lspinorb', '0').lower() in ['1', 'true', 't']:
    raise ValueError('This interface supports non-SOC nspin=1 calculations')
if parameters.get('gamma_only', '0').lower() in ['1', 'true', 't']:
    raise ValueError('Set gamma_only 0 to output the complex k-space wavefunction format')
stage = root/'abacus_charge'/f'step{int(sys.argv[1])}'
dft = stage/'dft'
dft.mkdir()
for key, default in [('stru_file', 'STRU'), ('kpoint_file', 'KPT')]:
    name = parameters.get(key, default)
    source = (reference/name).resolve()
    shutil.copy2(source, dft/default)
changes = {'init_chg':'dm','read_file_dir':stage.resolve(),
           'scf_nmax':1,'scf_thr':'1e6','out_wfc_lcao':1,'out_mat_hs2':1,
           'out_chg':'1 12','out_dmr':0,'out_ndigits':14,
           'stru_file':'STRU','kpoint_file':'KPT'}
for key in ['pseudo_dir', 'orbital_dir']:
    changes[key] = (reference/parameters.get(key, '.')).resolve()
lines = [line for line in (reference/'INPUT').read_text().splitlines()
         if not line.split() or line.split()[0].lower() not in changes]
lines.extend(f'{key} {value}' for key,value in changes.items())
(dft/'INPUT').write_text('\n'.join(lines)+'\n')
# Keep GNU/OpenMPI children independent of the parent Intel MPI launch.
env = {key:value for key,value in os.environ.items()
       if not key.startswith(('PMI','PMIX','I_MPI','OMPI','HYDRA','MPIR_','MPI_','SLURM_STEP'))
       and key not in ['SLURM_PROCID','SLURM_LOCALID','SLURM_TASK_PID']}
env.update(config.get('environment', {}))
script = ''
if config.get('environment_script'):
    setup = (root/config['environment_script']).resolve()
    script += 'source ' + shlex.quote(str(setup)) + '\n'
script += 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
script += 'exec ' + shlex.join(config.get('command', ['abacus']))
with (dft/'abacus.log').open('w') as log:
    subprocess.run(['bash','-c',script],cwd=dft,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
# Preserve the user-selected atoms, orbital order, equivalence, U and J.
interface = root/'dft/outputs_to_DMFT'
target = dft/'outputs_to_DMFT'
target.mkdir()
for name in ['correlated_atoms.info', 'symmetry.dat']:
    shutil.copy2(interface/name, target/name)
for name in ['convert_abacus.py', 'abacus_io.py']:
    shutil.copy2(Path(__file__).with_name(name), stage/name)
subprocess.run([sys.executable,'convert_abacus.py'],cwd=stage,check=True)
report = json.loads((stage/'conversion_report.json').read_text())
residual = report['max_HC_minus_SCepsilon_Ry']
if residual>1e-5:
    print(f'KS residual warning: {residual} Ry',flush=True)
if residual>1e-4 or not report['max_CdaggerSC_error']<1e-5:
    raise RuntimeError('ABACUS KS consistency check failed')
source = output_directory(dft, input_parameters(dft/'INPUT'))
cube = source/'chg.cube'
if not cube.exists():
    cube = source/'SPIN1_CHG.cube'
shutil.copy2(cube, stage/'charge.cube')
if not interface.is_symlink():
    raise RuntimeError('Expected a case-local interface symlink; refusing to overwrite directory')
interface.unlink()
interface.symlink_to(target.resolve(),target_is_directory=True)
