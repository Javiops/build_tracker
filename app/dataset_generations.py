"""Readers accept only complete, content-bound dataset generations."""
import json
from pathlib import Path
from app.dataset_artifacts import sha256, canonical_sha

SPLITS = ('train','val','test')
REQUIRED = {'split_manifest.json','split_assignment.json','split_plan.json','coverage.json',
            'qualification.json','qualification_code.py','semantic_audit.json','audit_code.py','corpus_manifest.json'} | {f'visits_{s}.jsonl' for s in SPLITS}


def validate_run_directory(dataset, run):
    from app.config import DATA_DIR
    dataset, run = Path(dataset).resolve(),Path(run).resolve()
    experiments = (DATA_DIR / 'experiments').resolve()
    if not run.is_relative_to(experiments) or run == experiments or run.is_relative_to(dataset) or dataset.is_relative_to(run):
        raise ValueError('Experiment outputs must be isolated under data/experiments, outside dataset and serving files')


def generation_fingerprint(data_files, qualification_sha256, row_builder_sha256, split_plan_sha256):
    return canonical_sha({'data_files':data_files,'qualification_sha256':qualification_sha256,
        'row_builder_sha256':row_builder_sha256,'split_plan_sha256':split_plan_sha256})


def verify_generation(directory: Path, *, verify_splits=('train','val'), staged=False):
    directory = directory.resolve()
    if not staged and (directory.name.startswith('.') or directory.name.endswith('.partial')):
        raise ValueError('An unfinished generation is not a training input')
    manifest = json.loads((directory / 'generation_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('state') != 'complete' or manifest.get('purpose') != 'pipeline_smoke_only' or manifest.get('production_hold_cleared') is not False or manifest.get('promotion_allowed') is not False:
        raise ValueError('Generation is not a complete, scoped dataset')
    files = manifest.get('files') or {}
    if not REQUIRED <= files.keys():
        raise ValueError('Generation is missing required artifacts')
    for name, binding in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != binding['bytes']:
            raise ValueError(f'Invalid or missing generation artifact: {name}')
        # Training/evaluation never opens final-test payloads, even for hashes.
        # Its digest was accumulated while the publisher wrote the stream.
        if name.startswith('visits_') and name.endswith('.jsonl') and name[7:-6] not in verify_splits:
            continue
        if sha256(path) != binding['sha256']:
            raise ValueError(f'Generation artifact changed: {name}')
    q = json.loads((directory / 'qualification.json').read_text(encoding='utf-8'))
    audit = json.loads((directory / 'semantic_audit.json').read_text(encoding='utf-8'))
    coverage = json.loads((directory / 'coverage.json').read_text(encoding='utf-8'))
    split = json.loads((directory / 'split_manifest.json').read_text(encoding='utf-8'))
    assignment = json.loads((directory / 'split_assignment.json').read_text(encoding='utf-8'))
    plan = json.loads((directory / 'split_plan.json').read_text(encoding='utf-8'))
    corpus = json.loads((directory / 'corpus_manifest.json').read_text(encoding='utf-8'))
    if q.get('status') != 'qualified_subset' or q.get('promotion_allowed') is not False or not audit.get('passed') or audit.get('hard_failures'):
        raise ValueError('Generation lacks qualifying semantic evidence')
    if q['semantic_audit_sha256'] != files['semantic_audit.json']['sha256'] or q['corpus_manifest_sha256'] != files['corpus_manifest.json']['sha256']:
        raise ValueError('Qualification evidence chain is broken')
    if (q['qualification_code_sha256'] != files['qualification_code.py']['sha256']
        or audit['audit_code_sha256'] != files['audit_code.py']['sha256']
        or audit['corpus_manifest_sha256'] != files['corpus_manifest.json']['sha256']
        or q['db_sha256'] != corpus['db_sha256'] or audit['db_sha256'] != corpus['db_sha256']):
        raise ValueError('Audit/code/database evidence chain is broken')
    if coverage['qualification_sha256'] != files['qualification.json']['sha256'] or coverage['split_plan_sha256'] != files['split_plan.json']['sha256']:
        raise ValueError('Coverage evidence chain is broken')
    if (plan['assignment'] != assignment or plan['qualification_sha256'] != files['qualification.json']['sha256']
        or plan['row_builder_sha256'] != coverage['row_builder_sha256']
        or plan['row_counts'] != split['rows_per_split'] or plan['game_counts'] != split['counts']
        or q.get('production_hold_cleared') is not False or q.get('purpose') != 'isolated_experiment_only'
        or coverage.get('target_encoding') != 'item-counts-v1'
        or set(q.get('inventory_constraint_states',[])) != {'board_before','shop_before','shop_after'}):
        raise ValueError('Frozen split, multiset or qualification scope mismatch')
    if set(assignment.values()) != set(SPLITS) or not assignment.keys() <= set(q['accepted_match_ids']):
        raise ValueError('Split contains unqualified games or incomplete folds')
    data_files = {name:files[name] for name in ('split_assignment.json',*(f'visits_{s}.jsonl' for s in SPLITS))}
    fingerprint = generation_fingerprint(data_files,files['qualification.json']['sha256'],coverage['row_builder_sha256'],files['split_plan.json']['sha256'])
    if manifest['export_fingerprint'] != fingerprint or split['export_fingerprint'] != fingerprint or split.get('probe') is not True:
        raise ValueError('Dataset fingerprint or experiment scope changed')
    return manifest
