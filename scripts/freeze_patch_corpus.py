"""Gate 1: snapshot a completed sample and all verified raw pairs; no network."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import DATA_DIR, RAW_MATCH_V5_DIR, CACHE_DIR
from app.dataset_artifacts import canonical_sha, publish_directory, sha256, write_json
from app.ddragon import dragon_for_patch
from app.riot import RawMatchArchive


def freeze(sample: Path, destination: Path):
    if destination.exists():
        raise SystemExit('Frozen corpus already exists; verify it instead of overwriting')
    state = json.loads((sample / 'state.json').read_text(encoding='utf-8'))
    report = json.loads((sample / 'report.json').read_text(encoding='utf-8'))
    if not state['discovery_complete'] or not report['structural_checks_passed'] or len(state['results']) != len(state['jobs']):
        raise SystemExit('Collection did not finish successfully')
    if sha256(sample / 'state.json') != report['checkpoint_sha256']:
        raise SystemExit('Collection report/checkpoint hash mismatch')
    selected = {mid: result for mid, result in state['results'].items() if result['status'] == 'stored'}
    dragon = dragon_for_patch(state['patch'])
    if dragon.signature != state['static_data_sha256']:
        raise SystemExit('Static data changed since collection')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir()
    for name in ('state.json', 'report.json'):
        shutil.copy2(sample / name, pending / name)
    print('Snapshotting completed sample database...', flush=True)
    with closing(sqlite3.connect((sample / 'tracker.db').as_uri() + '?mode=ro', uri=True)) as source:
        with closing(sqlite3.connect(pending / 'tracker.db')) as target:
            source.backup(target)
            target.execute('PRAGMA journal_mode=DELETE')
            ids = {r[0] for r in target.execute('SELECT match_id FROM games')}
            if ids != set(selected):
                raise ValueError('Checkpoint/database game IDs differ')
            versions = list(target.execute("SELECT DISTINCT json_extract(payload_json,'$.reconstruction_version'), "
                "json_extract(payload_json,'$.inventory_version'), json_extract(payload_json,'$.label_source_version'), "
                "json_extract(payload_json,'$.static_data_sha256') FROM game_events"))
            if len(versions) != 1 or any(value is None for value in versions[0]):
                raise ValueError('New freezes require uniform, explicit reconstruction/inventory/label/static versions')
            reconstruction, inventory_version, label_version, static_digest = versions[0]
            if reconstruction != state['reconstruction_version'] or static_digest != dragon.signature:
                raise ValueError('Stored event versions disagree with collection scope')
    archive = RawMatchArchive(RAW_MATCH_V5_DIR)
    inventory = pending / 'source_inventory.jsonl'
    with inventory.open('x', encoding='utf-8', newline='\n') as stream:
        for index, (mid, result) in enumerate(sorted(selected.items()), 1):
            entry = {'match_id': mid, 'cohorts': result['cohorts'], 'files': {}}
            for kind in ('match', 'timeline'):
                payload = archive.read(kind, mid)
                if payload is None or payload.get('metadata', {}).get('matchId') != mid or canonical_sha(payload) != result[f'{kind}_sha256']:
                    raise ValueError(f'{mid}: missing/changed {kind} source')
                rel = Path('raw') / kind / f'{mid}.json.gz'
                target = pending / rel
                target.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(RAW_MATCH_V5_DIR / kind / f'{mid}.json.gz', target)
                entry['files'][kind] = {'path': rel.as_posix(), 'sha256': sha256(target), 'payload_sha256': canonical_sha(payload)}
            stream.write(json.dumps(entry, sort_keys=True) + '\n')
            if index % 200 == 0:
                print(f'Frozen source pairs {index}/{len(selected)}', flush=True)
    bindings = {}
    for name, expected in state['collection_code_sha256'].items():
        source = ROOT / name
        if sha256(source) != expected:
            raise ValueError(f'Collection code changed before freeze: {name}')
        target = pending / 'code' / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(source, target)
        bindings[name] = expected
    for kind in ('item', 'champion'):
        filename = f'{kind}-{dragon.version}.json'
        (pending / 'static').mkdir(exist_ok=True)
        shutil.copy2(CACHE_DIR / filename, pending / 'static' / filename)
    files = {p.relative_to(pending).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in pending.rglob('*') if p.is_file() and 'raw' not in p.relative_to(pending).parts}
    manifest = {'schema': 1, 'status': 'frozen_collection_not_semantically_accepted',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'sample': str(sample),
        'patch': state['patch'], 'cutoff_utc': report['cutoff_utc'], 'games': len(selected),
        'reconstruction_version': state['reconstruction_version'], 'static_data_sha256': dragon.signature,
        'ddragon_version': dragon.version, 'source_inventory_sha256': sha256(inventory),
        'db_sha256': sha256(pending / 'tracker.db'), 'collection_code_sha256': bindings,
        'reconstruction_code_sha256':bindings,'inventory_version':inventory_version,'label_source_version':label_version,
        'freeze_code_sha256': sha256(Path(__file__)), 'files': files,
        'unresolved_pros': report['unresolved_pros'], 'scope_complete': report['scope_complete'],
        'exclusion_counts': report['counts'], 'training_acceptance': 'pending semantic audit'}
    write_json(pending / 'corpus_manifest.json', manifest)
    for path in pending.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    publish_directory(pending, destination)
    print(json.dumps({'corpus': str(destination), 'games': len(selected),
        'db_sha256': manifest['db_sha256'], 'manifest_sha256': sha256(destination / 'corpus_manifest.json')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    freeze(args.sample.resolve(), args.out.resolve())
