"""One ABACUS KS update and conversion; the C++ program owns the charge loop."""
from pathlib import Path
import os
import sys
import shutil
import subprocess
import json

project = Path('/data/home/df_iopcas_gzb/GW+DMFT')
reference = project/'abacus_dmft_example/NiO_dmr_roundtrip/reference'
root = Path.cwd()
stage = root/'abacus_charge'/f'step{int(sys.argv[1])}'
dft = stage/'dft'
dft.mkdir()
for name in ['STRU','KPT']:
    shutil.copy2(reference/name,dft/name)
for pattern in ['*.orb','*.upf']:
    for file in reference.glob(pattern):
        (dft/file.name).symlink_to(file.resolve())
changes = {'init_chg':'dm','read_file_dir':stage.resolve(),
           'scf_nmax':1,'scf_thr':'1e6','out_wfc_lcao':1,'out_mat_hs2':1,
           'out_chg':'1 12','out_dmr':0,'out_ndigits':14}
lines = [line for line in (reference/'INPUT').read_text().splitlines()
         if not line.split() or line.split()[0].lower() not in changes]
lines.extend(f'{key} {value}' for key,value in changes.items())
(dft/'INPUT').write_text('\n'.join(lines)+'\n')
# The parent is an Intel MPI rank. The ABACUS child uses an independent local
# GNU/OpenMPI launcher, with no inherited PMI rendezvous or Slurm step identity.
env = {key:value for key,value in os.environ.items()
       if not key.startswith(('PMI','PMIX','I_MPI','OMPI','HYDRA','MPIR_','MPI_','SLURM_STEP'))
       and key not in ['SLURM_PROCID','SLURM_LOCALID','SLURM_TASK_PID']}
script = f'source {project}/cluster/abacus_env.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
script += 'mpirun --mca ras ^slurm --mca plm isolated --oversubscribe --bind-to none -np 2 abacus'
with (dft/'abacus.log').open('w') as log:
    subprocess.run(['bash','-c',script],cwd=dft,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
shutil.copy2(Path(__file__).with_name('convert_abacus.py'),stage/'convert_abacus.py')
subprocess.run([sys.executable,'convert_abacus.py'],cwd=stage,check=True)
report = json.loads((stage/'conversion_report.json').read_text())
residual = report['max_HC_minus_SCepsilon_Ry']
if residual>1e-5:
    print(f'KS residual warning: {residual} Ry',flush=True)
if residual>1e-4 or not report['max_CdaggerSC_error']<1e-5:
    raise RuntimeError('ABACUS KS consistency check failed')
interface = root/'dft/outputs_to_DMFT'
if not interface.is_symlink():
    raise RuntimeError('Expected a case-local interface symlink; refusing to overwrite directory')
interface.unlink()
interface.symlink_to((dft/'outputs_to_DMFT').resolve(),target_is_directory=True)
