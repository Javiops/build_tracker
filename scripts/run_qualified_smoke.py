"""Publish gate 4, then run one fixed trunk/graft/policy smoke for gate 5."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256, write_json
from app.dataset_generations import verify_generation, validate_run_directory

CODE = ['app/'+name+'.py' for name in ('reconstruct','inventory','ddragon','shop_econ','purchase_multiset',
        'predictor','deployment','pipeline_guard','dataset_artifacts','dataset_generations')] + [
        'scripts/'+name+'.py' for name in ('baseline','corpus_rows','publish_patch_dataset',
        'train_prefix','eval_policy','run_qualified_smoke')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage',type=Path,required=True)
    p.add_argument('--name',required=True)
    a = p.parse_args()
    dataset = ROOT/'data/dataset_generations'/a.name
    run = ROOT/'data/experiments'/a.name
    validate_run_directory(dataset,run)
    run.mkdir(parents=True,exist_ok=False)
    protected = [ROOT/'data/ml'/name for name in ('prefix_model.pt','deployment_manifest.json','pipeline_hold.json')]
    before = {str(path):sha256(path) if path.exists() else None for path in protected}
    code_hashes = {name:sha256(ROOT/name) for name in CODE}
    for name in CODE:
        destination = run/'code'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    fixed = {'PREFIX_ML_DIR':str(dataset),'PREFIX_RUN_DIR':str(run),'PREFIX_EPOCHS':'1',
             'PREFIX_DMODEL':'64','PREFIX_LAYERS':'2','PREFIX_FF':'128','PREFIX_HEADS':'4',
             'PREFIX_COSINE':'0','PREFIX_EXTRAS':'1','PREFIX_HISTORY':'0','PREFIX_GOLDEST':'0',
             'PREFIX_GOLDX':'1','PREFIX_RUNES':'1','PREFIX_POSW':'8','PREFIX_POSW_SCHED':'',
             'PREFIX_POSW_CUSTOM':'0','PREFIX_TBLEND':'0','PREFIX_SPLICE':'0','PYTHONUTF8':'1'}
    write_json(run/'run_plan.json',{'purpose':'pipeline_smoke_only','dataset':str(dataset),
               'fixed_environment':fixed,'code_sha256':code_hashes,'protected_before':before,
               'validation_policy_games':10,'final_test_evaluation':False,'promotion_allowed':False,
               'stages':['atomic_export','one_epoch_trunk','one_epoch_frozen_trunk_graft','displayed_policy_smoke']})
    env = {**os.environ,**fixed}
    def unchanged():
        if any(sha256(ROOT/name) != digest for name,digest in code_hashes.items()):
            raise ValueError('Code changed during the fixed smoke run')
        after = {str(path):sha256(path) if path.exists() else None for path in protected}
        if after != before: raise ValueError('Protected serving/hold artifacts changed during smoke')
    def status(stage,state,**extra):
        write_json(run/'status.json',{'stage':stage,'state':state,'time_utc':datetime.now(timezone.utc).isoformat(),**extra})
    def stage(name,args,overrides=None):
        unchanged()
        status(name,'running')
        with (run/f'{name}.log').open('x',encoding='utf-8') as log:
            result = subprocess.run([sys.executable,*args],cwd=ROOT,env={**env,**(overrides or {})},
                                    stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit code {result.returncode}; see {name}.log')
        unchanged()
        status(name,'complete')
    try:
        stage('atomic_export',['scripts/publish_patch_dataset.py','--coverage',str(a.coverage.resolve()),'--name',a.name])
        generation = verify_generation(dataset)
        stage('trunk',['scripts/train_prefix.py'],{'PREFIX_SAVEW':'0','PREFIX_TARGETW':'0','PREFIX_INIT_FROM':'','PREFIX_FREEZE':'0','PREFIX_OUT':'trunk.pt'})
        stage('graft',['scripts/train_prefix.py'],{'PREFIX_SAVEW':'0.5','PREFIX_TARGETW':'0.1','PREFIX_INIT_FROM':'trunk.pt','PREFIX_FREEZE':'1','PREFIX_OUT':'candidate.pt'})
        stage('policy',['scripts/eval_policy.py','--artifact','candidate.pt','--split','val','--games','10','--bootstrap','0'])
        report_path = run/'policy_eval_val_candidate.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['metrics']['A']['unaffordable'] != 0:
            raise ValueError('Displayed-policy smoke produced an unaffordable basket')
        if report['games'] != 10 or report['artifact_provenance']['A']['status'] != 'probe_only':
            raise ValueError('Unexpected policy population or candidate provenance')
        verify_generation(dataset)
        unchanged()
        write_json(run/'completion.json',{'status':'pipeline_smoke_passed','purpose':'pipeline_smoke_only',
             'export_fingerprint':generation['export_fingerprint'],'candidate_sha256':sha256(run/'candidate.pt'),
             'policy_report_sha256':sha256(report_path),'policy_games':report['games'],'policy_visits':report['visits'],
             'unaffordable_baskets':0,'protected_unchanged':True,'production_hold_cleared':False,
             'promotion_allowed':False,'final_test_evaluation':False,'code_sha256':code_hashes,
             'limits':['One epoch per stage only tests mechanics; no candidate quality claim.',
                       'Displayed-policy capacity still conservatively counts owned copies as regular slots; role-slot-aware decoding remains required before comparison.']})
        status('all','complete')
    except Exception as exc:
        status('supervisor','failed',error=str(exc))
        raise


if __name__ == '__main__': main()
