"""Close the semantic gate with a reviewed, explicit whole-game eligibility set.

An audit pass is evidence, not permission to use every collected game. This
qualification binds the immutable corpus and audit, adds known unsupported
inventory mechanics, and permits only isolated experiments on the resulting set.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.dataset_artifacts import sha256, write_json
from app.ddragon import dragon_for_patch
from app.riot import RawMatchArchive
from app.shop_econ import payload_ids, PINK


def inventory_shape(items, role, bot_quest_complete, dragon):
    """Count modeled copies separately from occupied regular/role slots."""
    bag = Counter(i for i in payload_ids(items) if not dragon.classify(i)['skip'])
    boots = sum(n for i,n in bag.items() if dragon.classify(i)['is_boots'] or dragon.classify(i)['is_basic_boots'])
    ward_copies = sum(bag[i] for i in PINK)
    role_copies = ward_copies if role == 'UTILITY' else min(1,boots) if role == 'BOTTOM' and bot_quest_complete else 0
    slots = sum(n for i,n in bag.items() if i not in PINK)
    if role != 'UTILITY':
        slots += sum(1 for ward in PINK if bag[ward])
    if role == 'BOTTOM' and bot_quest_complete:
        slots -= min(1,boots)
    return {'modeled_item_copies':sum(bag.values()),'regular_slots':slots,
            'role_slot_item_copies':role_copies,'ward_copies':ward_copies,'boot_copies':boots}


def inventory_issues(items, role, bot_quest_complete, dragon):
    """Independent necessary constraints on modeled inventory, not a replay."""
    bag = Counter(i for i in payload_ids(items) if not dragon.classify(i)['skip'])
    shape = inventory_shape(items,role,bot_quest_complete,dragon)
    reasons = set()
    boots = shape['boot_copies']
    if boots > 1:
        reasons.add('multiple_boot_items')
    for ward in PINK:
        if bag[ward] > int((dragon.item(ward) or {}).get('stacks') or 1):
            reasons.add('ward_stack_overflow')
    if shape['regular_slots'] > 6:
        reasons.add('modeled_inventory_slot_overflow')
    return reasons


def first_bot_quests(timeline):
    result = {}
    for frame in timeline['info']['frames']:
        for event in frame.get('events',[]):
            if event.get('type') == 'ITEM_DESTROYED' and event.get('itemId') == 1202:
                pid = event['participantId']
                result[pid] = min(result.get(pid,float('inf')),event['timestamp'])
    return result


def qualify(corpus, audit_path, out):
    if out.exists():
        raise ValueError('Qualification already exists; refusing overwrite')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    manifest_path = corpus / 'corpus_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not audit.get('passed') or not audit.get('training_clearance') or audit.get('hard_failures'):
        raise ValueError('Semantic audit has not passed')
    if audit['corpus_manifest_sha256'] != sha256(manifest_path) or audit['db_sha256'] != manifest['db_sha256']:
        raise ValueError('Audit is not bound to this corpus')
    if sha256(corpus / 'tracker.db') != manifest['db_sha256']:
        raise ValueError('Audited database changed')
    if sha256(audit_path.parent / 'audit_code.py') != audit['audit_code_sha256']:
        raise ValueError('Auditor snapshot changed')
    accepted = set(audit['accepted_match_ids'])
    excluded = {mid:list(reasons) for mid,reasons in audit['quarantined_match_ids'].items()}
    if accepted & excluded.keys():
        raise ValueError('Audit eligibility sets overlap')
    dragon = dragon_for_patch(manifest['patch'])
    viego = dragon.champion_id_by_name('Viego')
    if not viego:
        raise ValueError('Pinned static data cannot identify Viego')
    with sqlite3.connect((corpus / 'tracker.db').as_uri() + '?mode=ro&immutable=1',uri=True) as conn:
        games = {mid:json.loads(ps) for mid,ps in conn.execute('SELECT match_id,participants_json FROM games')}
    if accepted | excluded.keys() != games.keys() or len(games) != audit['counts']['games']:
        raise ValueError('Audit eligibility does not account for every source game')
    for mid, players in games.items():
        if any(p['champion_id'] == viego for p in players):
            # Possession changes inventory outside observed shop transactions.
            # Matching the final bag cannot certify the intervening boards.
            excluded.setdefault(mid,[]).append('unmodeled_viego_inventory')
            accepted.discard(mid)
    archive = RawMatchArchive(corpus / 'raw')
    constraint_examples = []
    checked_events = 0
    accepted_shapes = Counter()
    accepted_max_copies = accepted_max_slots = 0
    with sqlite3.connect((corpus / 'tracker.db').as_uri() + '?mode=ro&immutable=1',uri=True) as conn:
        for index, mid in enumerate(sorted(accepted),1):
            timeline = archive.read('timeline',mid)
            bot_quests = first_bot_quests(timeline)
            reasons = set()
            shapes = Counter()
            max_copies = max_slots = 0
            for (payload,) in conn.execute('SELECT payload_json FROM game_events WHERE match_id=?',(mid,)):
                event = json.loads(payload)
                checked_events += 1
                anchor = event['ts']-15000 if event.get('is_save') else event['ts']
                self_member = next(m for m in event['board'] if m['participant_id'] == event['participant_id'])
                states = [('board_before',member,member['items'],anchor,False) for member in event['board']]
                states += [('shop_before',self_member,event['inventory_before'],anchor,False),
                           ('shop_after',self_member,event['inventory_after'],event['ts_end'],True)]
                for state, member, items, when, inclusive in states:
                    quest = bot_quests.get(member['participant_id'],float('inf'))
                    complete = quest <= when if inclusive else quest < when
                    shape = inventory_shape(items,member['team_position'],complete,dragon)
                    shapes[f"{state}:{member['team_position']}:copies={shape['modeled_item_copies']}:regular_slots={shape['regular_slots']}:role_copies={shape['role_slot_item_copies']}"] += 1
                    max_copies = max(max_copies,shape['modeled_item_copies'])
                    max_slots = max(max_slots,shape['regular_slots'])
                    issues = inventory_issues(items,member['team_position'],complete,dragon)
                    reasons.update(issues)
                    if issues and len(constraint_examples) < 12:
                        constraint_examples.append({'match_id':mid,'ts':event['ts'],
                            'participant_id':member['participant_id'],'state':state,
                            'issues':sorted(issues),'items':payload_ids(items),'shape':shape})
                if Counter(payload_ids(event.get('label_bought',[]))) - Counter(payload_ids(event['inventory_after'])):
                    reasons.add('purchase_label_absent_from_inventory_after')
            if reasons:
                accepted.discard(mid)
                excluded.setdefault(mid,[]).extend(sorted(reasons))
            else:
                accepted_shapes.update(shapes)
                accepted_max_copies = max(accepted_max_copies,max_copies)
                accepted_max_slots = max(accepted_max_slots,max_slots)
            if index % 200 == 0:
                print(f'Qualified physical inventory constraints for {index} games',flush=True)
    if not accepted:
        raise ValueError('No games qualify; no downstream clearance')
    report = {
        'schema':2, 'status':'qualified_subset', 'purpose':'isolated_experiment_only',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'corpus':str(corpus),'corpus_manifest_sha256':sha256(manifest_path),
        'db_sha256':manifest['db_sha256'],'semantic_audit':str(audit_path),'semantic_audit_sha256':sha256(audit_path),
        'qualification_code_sha256':sha256(Path(__file__)), 'patch':manifest['patch'],
        'reconstruction_version':manifest['reconstruction_version'],'inventory_version':manifest['inventory_version'],
        'source_games':len(games),'quality_accepted_games':len(accepted),'quarantined_games':len(excluded),
        'accepted_match_ids':sorted(accepted),'quarantined_match_ids':dict(sorted(excluded.items())),
        'exclusion_reason_counts':dict(Counter(reason for reasons in excluded.values() for reason in reasons)),
        'inventory_constraint_checked_events':checked_events,'inventory_constraint_examples':constraint_examples,
        'inventory_constraint_states':['board_before','shop_before','shop_after'],
        'accepted_inventory_state_shapes':dict(sorted(accepted_shapes.items())),
        'accepted_max_modeled_item_copies':accepted_max_copies,'accepted_max_regular_slots':accepted_max_slots,
        'training_clearance':'accepted IDs only, after coverage and atomic export gates',
        'production_hold_cleared':False,'promotion_allowed':False,
        'limits':[
            'The complete collected database is not certified as training-ready.',
            'Whole-game exclusion protects every perspective when any board member is unsupported.',
            'Viego inventory changes during possession are outside the documented reconstruction rules.',
            'Unreconciled terminal differences remain unresolved; they are not assumed to be API defects.',
            'Inventory capacity checks cover modeled items; ignored consumable/trinket slots are not a complete live shopping legality proof.',
            'Coverage and temporal splitting are subsequent gates; this report proves neither model quality nor sample sufficiency.',
            'Offline gold remains an approximation and historic no-buys remain post-death observations only.'
        ]}
    write_json(out,report)
    shutil.copy2(Path(__file__),out.parent / 'qualification_code.py')
    print(json.dumps({k:report[k] for k in ('status','source_games','quality_accepted_games','quarantined_games','exclusion_reason_counts')},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    qualify(a.corpus.resolve(),a.audit.resolve(),a.out.resolve())
