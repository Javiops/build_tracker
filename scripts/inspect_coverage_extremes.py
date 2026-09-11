"""Trace reported inventory/basket maxima to qualified events and raw sources."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
from corpus_rows import load_qualified, iter_games, read_db
from app.riot import RawMatchArchive
from app.dataset_artifacts import write_json, sha256

q_path = ROOT / 'data/acceptance/16.18_repaired_v2/qualification.json'
context = load_qualified(q_path)
archive = RawMatchArchive(context.path / 'raw')
examples, counts = [], Counter()
for game, rows, excluded in iter_games(context):
    if excluded: continue
    mid = game['match_id']
    for row in rows:
        maximum = max([len(row['inventory'])] + [len(p['items']) for p in row['others']])
        repeat = max(Counter(row['label_ids']).values())
        if maximum >= 8: counts['rows_with_inventory_count_ge8'] += 1
        if repeat >= 5: counts['rows_with_label_copies_ge5'] += 1
        if not (maximum >= 8 and sum(e['kind']=='inventory' for e in examples)<3 or repeat>=5 and sum(e['kind']=='purchase' for e in examples)<5):
            continue
        with read_db(context.path / 'tracker.db') as conn:
            event = next(json.loads(r['payload_json']) for r in conn.execute('SELECT payload_json FROM game_events WHERE match_id=? AND ts=?',(mid,row['ts']))
                         if json.loads(r['payload_json'])['participant_id']==row['participant_id'])
        tl = archive.read('timeline',mid)
        raw = [e for f in tl['info']['frames'] for e in f.get('events',[])]
        members = [p for p in event['board'] if len([i for i in p['items'] if not i['skip']])>=8]
        kind = 'purchase' if repeat >= 5 else 'inventory'
        relevant_ids = {row['participant_id']} if kind=='purchase' else {p['participant_id'] for p in members}
        traces = [e for e in raw if (e.get('participantId') in relevant_ids or e.get('creatorId') in relevant_ids)
                  and (event['ts']-90000 <= e['timestamp'] <= event['ts_end']+15000)
                  and (e['type'].startswith('ITEM') or e['type']=='WARD_PLACED')]
        examples.append({'kind':kind,'match_id':mid,'ts':row['ts'],'ts_end':event['ts_end'],'participant_id':row['participant_id'],
            'champion':row['champion'],'role':row['role'],'label_ids':row['label_ids'],
            'inventory_before':event['inventory_before'],'inventory_after':event['inventory_after'],
            'large_board_members':members,'raw_nearby':traces})
report = {'qualification_sha256':sha256(q_path),'counts':dict(counts),'examples':examples}
out = ROOT / 'data/acceptance/16.18_repaired_v2/coverage_extremes.json'
write_json(out,report)
print(json.dumps({'report':str(out),'counts':dict(counts),'examples':[{'kind':e['kind'],'match_id':e['match_id'],'ts':e['ts'],
    'champion':e['champion'],'labels':e['label_ids'],'large_board_members':[{'pid':p['participant_id'],'role':p['team_position'],'items':[i['item_id'] for i in p['items'] if not i['skip']]} for p in e['large_board_members']],
    'raw_nearby':e['raw_nearby'] if e['kind']=='purchase' else []} for e in examples]},indent=2))
