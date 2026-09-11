"""Shared, read-only row construction for qualified-corpus coverage and export."""
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
import baseline
from app.dataset_artifacts import canonical_sha, sha256
from app.ddragon import dragon_for_patch
from app.reconstruct import GOLD_EST_VERSION
from app.purchase_multiset import label_counts, serialize_counts, TARGET_ENCODING


@dataclass(frozen=True)
class QualifiedCorpus:
    path: Path
    qualification_path: Path
    qualification: dict
    manifest: dict
    dragon: object


@contextmanager
def read_db(path):
    with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1',uri=True) as conn:
        conn.row_factory = sqlite3.Row
        yield conn


def row_builder_digest():
    return canonical_sha({
        'functions':{f.__name__:inspect.getsource(f) for f in (baseline._example,baseline._label,baseline._item_ids,baseline._others,baseline._unmodeled_regular_slots)},
        'min_duration':baseline.MIN_DURATION_S,
        'corpus_rows':sha256(Path(__file__)),
        'static_classifier':sha256(ROOT / 'app/ddragon.py'),
        'shop_economics':sha256(ROOT / 'app/shop_econ.py'),
        'purchase_multiset':sha256(ROOT / 'app/purchase_multiset.py')})


def load_qualified(path):
    q = json.loads(path.read_text(encoding='utf-8'))
    if q.get('status') != 'qualified_subset' or q.get('purpose') != 'isolated_experiment_only' or q.get('promotion_allowed') is not False:
        raise ValueError('Expected a qualified subset with an isolated-experiment scope')
    corpus = Path(q['corpus']).resolve()
    manifest_path = corpus / 'corpus_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if sha256(manifest_path) != q['corpus_manifest_sha256'] or sha256(corpus / 'tracker.db') != q['db_sha256'] or manifest['db_sha256'] != q['db_sha256']:
        raise ValueError('Qualified corpus binding changed')
    audit_path = Path(q['semantic_audit'])
    if sha256(audit_path) != q['semantic_audit_sha256'] or sha256(path.parent / 'qualification_code.py') != q['qualification_code_sha256']:
        raise ValueError('Acceptance evidence changed')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or audit['hard_failures'] or audit['corpus_manifest_sha256'] != q['corpus_manifest_sha256']:
        raise ValueError('Audit does not certify these source checks')
    accepted = set(q['accepted_match_ids'])
    if len(accepted) != q['quality_accepted_games'] or not accepted <= set(audit['accepted_match_ids']) or accepted & q['quarantined_match_ids'].keys():
        raise ValueError('Invalid qualified game set')
    dragon = dragon_for_patch(manifest['patch'])
    if dragon.signature != manifest['static_data_sha256']:
        raise ValueError('Patch static data changed')
    for name in ('app/ddragon.py','app/shop_econ.py'):
        if sha256(ROOT / name) != manifest['reconstruction_code_sha256'][name]:
            raise ValueError(f'Row semantics changed since the source audit: {name}')
    return QualifiedCorpus(corpus,path,q,manifest,dragon)


def iter_games(context):
    """Yield metadata, one bounded game's rows, and any export exclusion."""
    accepted = set(context.qualification['accepted_match_ids'])
    with read_db(context.path / 'tracker.db') as conn:
        for game in conn.execute('SELECT * FROM games ORDER BY match_id'):
            mid = game['match_id']
            if mid not in accepted:
                continue
            metadata = dict(game)
            participants = json.loads(game['participants_json'])
            metadata['participants'] = participants
            if game['game_duration'] < baseline.MIN_DURATION_S:
                yield metadata, [], 'duration_below_600s'
                continue
            if len(participants) != 10:
                raise ValueError(f'{mid}: incomplete qualified lobby')
            by_puuid = {p['puuid']:p for p in participants}
            shoppers, rows, raw_count = set(), [], 0
            for record in conn.execute('SELECT payload_json FROM game_events WHERE match_id=? ORDER BY event_index',(mid,)):
                event = json.loads(record['payload_json'])
                raw_count += 1
                shoppers.add(event['puuid'])
                expected = {'reconstruction_version':context.manifest['reconstruction_version'],
                    'inventory_version':context.manifest['inventory_version'],
                    'label_source_version':context.manifest['label_source_version'],
                    'gold_est_version':GOLD_EST_VERSION,'patch':context.manifest['patch'],
                    'ddragon_version':context.dragon.version,'static_data_sha256':context.dragon.signature}
                if any(event.get(k) != v for k,v in expected.items()) or event.get('training_eligible') is not True:
                    raise ValueError(f'{mid}: unqualified or mixed row provenance')
                event['match_id'] = mid
                event['perks'] = by_puuid[event['puuid']]
                row = baseline._example(event,context.dragon)
                if row:
                    row['participant_id'] = event['participant_id']
                    row['label_counts'] = serialize_counts(label_counts(row))
                    row['target_encoding'] = TARGET_ENCODING
                    row.pop('label_ids')
                    # Keep the label-derived legacy wallet in the forensic DB,
                    # never in a qualified model export.
                    row.pop('gold_arrival_true',None)
                    rows.append(row)
            metadata['raw_shop_events'] = raw_count
            if shoppers != by_puuid.keys():
                yield metadata, [], 'incomplete_shop_perspectives'
            elif not rows:
                yield metadata, [], 'no_purchase_or_save_targets'
            else:
                yield metadata, rows, None
