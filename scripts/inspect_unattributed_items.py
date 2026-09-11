"""Inspect same-time evidence around ownerless item changes; do not infer owners."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.riot import RawMatchArchive
from app.ddragon import dragon_for_patch
archive = RawMatchArchive(ROOT / 'data/corpora/16.18_initial/raw')
for mid in ('EUW1_7978988364','EUW1_7979000306'):
    match,tl = archive.read('match',mid),archive.read('timeline',mid)
    events = [e for f in tl['info']['frames'] for e in f.get('events',[])]
    targets = [e for e in events if (e.get('type','').startswith('ITEM') and not e.get('participantId') and e.get('timestamp',0)>0)]
    print(json.dumps({'match_id':mid,'participants':[{'id':p['participantId'],'role':p.get('teamPosition'),'items':[p.get(f'item{i}') for i in range(7)]} for p in match['info']['participants']],
        'ownerless_events':[{'event':e,'same_time':[x for x in events if x['timestamp']==e['timestamp']]} for e in targets[:10]]},indent=2))
    if mid == 'EUW1_7978988364':
        p = match['info']['participants'][2]
        print(json.dumps({'participant_keys':sorted(p), 'quest_fields':{k:v for k,v in p.items() if 'quest' in k.lower() or 'bound' in k.lower()},
            'challenge_quest_fields':{k:v for k,v in p.get('challenges',{}).items() if 'quest' in k.lower()},
            'boot_events':[e for e in events if e.get('participantId') == 3 and (e.get('itemId') in (1001,3158,3171) or e.get('beforeId') in (1001,3158,3171))],
            'control_ward_placements':[e for e in events if e.get('creatorId')==2 and e.get('wardType') not in ('YELLOW_TRINKET','SIGHT_WARD')][:6]},indent=2))
dragon = dragon_for_patch('16.18')
for iid in (2055,772043,3158,3171,3003,3040,3004,3042,3119,3121):
    data = dragon.item(iid) or {}
    print(json.dumps({'id':iid,**{k:data.get(k) for k in ('name','gold','from','into','specialRecipe','inStore','requiredAlly')}}))
