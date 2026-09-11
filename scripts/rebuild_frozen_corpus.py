"""Derive a corrected immutable corpus locally, preserving the original snapshot."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app import db
from app.dataset_artifacts import canonical_sha, sha256, write_json, publish_directory
from app.ddragon import dragon_for_patch
from app.ladder import _player_from_match, routing_for_match_id
from app.reconstruct import RECONSTRUCTION_VERSION, reconstruct_game, perspective_from_game
from app.riot import RawMatchArchive
from app.shop_econ import payload_ids
from collect_patch_sample import validate_pair


def rebuild(source, destination):
    if destination.exists():
        raise SystemExit('Derived corpus already exists; refusing overwrite')
    parent = json.loads((source / 'corpus_manifest.json').read_text(encoding='utf-8'))
    if sha256(source / 'tracker.db') != parent['db_sha256']:
        raise ValueError('Parent DB changed')
    for name, binding in parent['files'].items():
        if name != 'tracker.db' and sha256(source / name) != binding['sha256']:
            raise ValueError(f'Parent file changed: {name}')
    dragon = dragon_for_patch(parent['patch'])
    if dragon.signature != parent['static_data_sha256']:
        raise ValueError('Parent static data changed')
    pending = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir(parents=True)
    for path in source.iterdir():
        if path.name in ('tracker.db', 'corpus_manifest.json'):
            continue
        if path.is_dir():
            shutil.copytree(path, pending / path.name)
        else:
            shutil.copy2(path, pending / path.name)
    code = {}
    for name in ('app/reconstruct.py','app/inventory.py','app/shop_econ.py','app/db.py','app/config.py','app/ddragon.py',
                 'app/riot.py','app/ladder.py','scripts/baseline.py','scripts/rebuild_frozen_corpus.py'):
        code[name] = sha256(ROOT / name)
        target = pending / 'reconstruction_code' / name
        target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(ROOT / name,target)
    sources = {row['match_id']: row for row in (json.loads(line) for line in (source / 'source_inventory.jsonl').read_text().splitlines())}
    db.DB_PATH = pending / 'tracker.db'
    db.init_db()
    archive = RawMatchArchive(source / 'raw')
    changes = Counter()
    with sqlite3.connect((source / 'tracker.db').as_uri() + '?mode=ro&immutable=1', uri=True) as old:
        for index, (mid,) in enumerate(old.execute('SELECT match_id FROM games ORDER BY match_id'), 1):
            match, timeline = archive.read('match',mid), archive.read('timeline',mid)
            for kind, payload in (('match',match),('timeline',timeline)):
                binding = sources[mid]['files'][kind]
                if canonical_sha(payload) != binding['payload_sha256'] or sha256(source / binding['path']) != binding['sha256']:
                    raise ValueError(f'Frozen source changed: {mid}/{kind}')
            validate_pair(mid,match,timeline,parent['patch'])
            game, events = reconstruct_game(match,timeline,dragon)
            region, platform = routing_for_match_id(mid)
            for p in match['info']['participants']:
                db.upsert_player(_player_from_match(p,platform,region))
            entries = [perspective_from_game(game, events, p['puuid']) for p in match['info']['participants']]
            db.insert_reconstruction(game,events,entries)
            previous = [json.loads(r[0]) for r in old.execute('SELECT payload_json FROM game_events WHERE match_id=? ORDER BY event_index',(mid,))]
            old_events = {(e['participant_id'],e['ts']): e for e in previous}
            new_events = {(e['participant_id'],e['ts']): e for e in events}
            if len(old_events) != len(previous) or len(new_events) != len(events):
                raise ValueError(f'{mid}: ambiguous visit identity')
            changes['added_events'] += len(new_events.keys() - old_events.keys())
            changes['removed_events'] += len(old_events.keys() - new_events.keys())
            for key in old_events.keys() & new_events.keys():
                before, after = old_events[key], new_events[key]
                for field in ('score','gold_est','kills','deaths','assists'):
                    if canonical_sha(before[field]) != canonical_sha(after[field]):
                        changes[field] += 1
                for field in ('inventory_before','inventory_after','bought'):
                    if payload_ids(before[field]) != payload_ids(after[field]):
                        changes[field] += 1
                before_labels = [i['item_id'] for i in before['bought'] if not i.get('skip')]
                after_labels = [i['item_id'] for i in after['label_bought'] if not i.get('skip')]
                if before_labels != after_labels:
                    changes['purchase_label_events'] += 1
            if index % 100 == 0:
                print(f'Rebuilt {index}/{parent["games"]} from frozen sources', flush=True)
    with db.db() as conn:
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.execute('PRAGMA journal_mode=DELETE')
        if [r[0] for r in conn.execute('PRAGMA integrity_check')] != ['ok']:
            raise ValueError('Rebuild database integrity failure')
    for name, digest in code.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Code changed during rebuild: {name}; refusing publication')
    write_json(pending / 'repair_changes.json', {'parent_manifest_sha256':sha256(source / 'corpus_manifest.json'), 'changes':dict(changes)})
    manifest = {**parent, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'frozen_derived_corpus_pending_semantic_audit', 'parent_corpus':str(source),
        'parent_manifest_sha256':sha256(source / 'corpus_manifest.json'),
        'reconstruction_version':RECONSTRUCTION_VERSION, 'label_source_version':'purchase-net-v1',
        'inventory_version':'causal-quests-v1',
        'reconstruction_code_sha256':code, 'db_sha256':sha256(pending / 'tracker.db'),
        'files':{p.relative_to(pending).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
            for p in pending.rglob('*') if p.is_file() and 'raw' not in p.relative_to(pending).parts}}
    write_json(pending / 'corpus_manifest.json',manifest)
    for p in pending.rglob('*'):
        if p.is_file(): p.chmod(0o444)
    publish_directory(pending,destination)
    print(json.dumps({'corpus':str(destination),'changes':dict(changes),'db_sha256':manifest['db_sha256']},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a = parser.parse_args()
    rebuild(a.source.resolve(),a.out.resolve())
