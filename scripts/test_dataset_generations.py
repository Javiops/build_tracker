"""Interrupted exports never become current; readers reject tampering and mixed evidence."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
import publish_patch_dataset as publisher
from app.dataset_artifacts import write_json, sha256
from app.dataset_generations import verify_generation, validate_run_directory

def refused(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Invalid generation accepted')

with tempfile.TemporaryDirectory(dir=ROOT/'data') as temp:
    root = Path(temp)
    try:
        source = root/'source'
        source.mkdir()
        (source/'audit_code.py').write_text('# audit fixture',encoding='utf-8')
        (source/'qualification_code.py').write_text('# qualifier fixture',encoding='utf-8')
        manifest = {'db_sha256':'fixture-db','patch':'16.18','reconstruction_version':'decision-start-v2'}
        write_json(source/'corpus_manifest.json',manifest)
        audit = {'passed':True,'hard_failures':{},'db_sha256':'fixture-db',
                 'corpus_manifest_sha256':sha256(source/'corpus_manifest.json'),
                 'audit_code_sha256':sha256(source/'audit_code.py')}
        write_json(source/'semantic_audit.json',audit)
        assignment = {'A':'train','B':'val','C':'test'}
        q = {'status':'qualified_subset','purpose':'isolated_experiment_only',
             'promotion_allowed':False,'production_hold_cleared':False,
             'semantic_audit':str(source/'semantic_audit.json'),'db_sha256':'fixture-db',
             'semantic_audit_sha256':sha256(source/'semantic_audit.json'),
             'corpus_manifest_sha256':sha256(source/'corpus_manifest.json'),
             'qualification_code_sha256':sha256(source/'qualification_code.py'),
             'accepted_match_ids':list(assignment),
             'inventory_constraint_states':['board_before','shop_before','shop_after']}
        write_json(source/'qualification.json',q)
        builder = publisher.row_builder_digest()
        plan = {'assignment':assignment,'game_creation_ms':{'A':1,'B':2,'C':3},
                'game_counts':{'train':1,'val':1,'test':1},'row_counts':{'train':1,'val':1,'test':1},
                'row_builder_sha256':builder,'qualification_sha256':sha256(source/'qualification.json'),
                'purpose':'pipeline_smoke_only','eval_version':2,'benchmark_id':'fixture','policy':'temporal'}
        write_json(source/'split_plan.json',plan)
        coverage = {'status':'coverage_complete','smoke_ready':True,'qualification':str(source/'qualification.json'),
                    'qualification_sha256':sha256(source/'qualification.json'),'split_plan_sha256':sha256(source/'split_plan.json'),
                    'row_builder_sha256':builder,'target_encoding':'item-counts-v1',
                    'max_modeled_inventory_items':8,'max_basket_items':6,'max_copies_of_one_item':5}
        write_json(source/'coverage.json',coverage)
        context = SimpleNamespace(path=source,qualification=q,manifest=manifest,dragon=SimpleNamespace(version='16.18.1',signature='static-fixture'))
        def rows(_context):
            for mid in assignment:
                yield {'match_id':mid,'game_creation':plan['game_creation_ms'][mid]},[
                    {'match_id':mid,'gold_est_version':'prequential-v2','inventory_version':'causal-quests-v1',
                     'label_source_version':'purchase-net-v1','label_counts':{'1036':5},'target_encoding':'item-counts-v1'}],None
        with patch.object(publisher,'load_qualified',return_value=context),patch.object(publisher,'iter_games',side_effect=rows):
            for stage in ('game_written','validated','published'):
                target = root/stage
                write_json(target/'CURRENT.json',{'generation':'previous'})
                prior = (target/'CURRENT.json').read_bytes()
                def fail(at):
                    if at == stage: raise RuntimeError('simulated interruption')
                try: publisher.publish(source,target,'next',fault=fail)
                except RuntimeError: pass
                else: raise AssertionError('Interruption not exercised')
                assert (target/'CURRENT.json').read_bytes() == prior
                if stage == 'published':
                    verify_generation(target/'next')
                else:
                    assert not (target/'next').exists()
                    for pending in target.glob('.*.partial'):
                        refused(lambda:verify_generation(pending))
            target = root/'success'
            generation = publisher.publish(source,target,'next')
            pointer = json.loads((target/'CURRENT.json').read_text())
            assert pointer['generation_manifest_sha256'] == sha256(generation/'generation_manifest.json')
            verify_generation(generation)
            refused(lambda:publisher.publish(source,target,'next'))
            # Test bytes remain sealed: normal train/val verification must not read them.
            original = Path.open
            def no_test(path,*args,**kwargs):
                if path.name == 'visits_test.jsonl': raise AssertionError('Final test was opened')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',no_test): verify_generation(generation)
            train = generation/'visits_train.jsonl'
            train.chmod(0o666)
            train.write_bytes(train.read_bytes().replace(b'1036',b'1037'))
            refused(lambda:verify_generation(generation))
        refused(lambda:validate_run_directory(root/'dataset',root/'dataset'/'run'))
        refused(lambda:validate_run_directory(root/'dataset',ROOT/'data/ml'))
        print('ok generations: interruption at write/validate/publish, current-pointer preservation, immutable names, tamper rejection and sealed final test')
    finally:
        for path in root.rglob('*'):
            if path.is_file(): path.chmod(0o666)
