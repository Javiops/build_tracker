"""Independent raw ledger versus final DTO; diagnostic only, never a feature."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.ddragon import dragon_for_patch
from app.shop_econ import feature_inventory
from app.riot import RawMatchArchive
from app.dataset_artifacts import write_json, sha256

corpus = ROOT / 'data/corpora/16.18_initial'
archive = RawMatchArchive(corpus / 'raw')
dragon = dragon_for_patch('16.18')
opaque = set(json.loads((ROOT / 'data/acceptance/16.18_initial/source_ambiguities.json').read_text())['games_with_opaque_undos'])
boot_upgrades = {int(data['from'][0]):int(iid) for iid,data in dragon._items['data'].items()
    if len(data.get('from',[])) == 1 and 'Boots' in (dragon.item(int(data['from'][0])) or {}).get('tags',[]) and data.get('gold',{}).get('base') == 0 and data.get('maps',{}).get('11')}
counts, pairs, examples, affected = Counter(), Counter(), [], set()
for line in (corpus / 'source_inventory.jsonl').read_text().splitlines():
    mid = json.loads(line)['match_id']
    match, timeline = archive.read('match',mid),archive.read('timeline',mid)
    mid_complete, bot_complete, pending_boots = set(), set(), {}
    purchases_at = {}
    for f in timeline['info']['frames']:
        for e in f.get('events',[]):
            if e.get('type') == 'ITEM_PURCHASED': purchases_at.setdefault((e.get('participantId'),e['timestamp']),set()).add(e['itemId'])
    destroyed_wards = {(e.get('participantId'),e['timestamp']) for f in timeline['info']['frames'] for e in f.get('events',[])
        if e.get('type') == 'ITEM_DESTROYED' and e.get('itemId') in (2055,772043)}
    inventories = {p['participantId']: Counter([3865] if (p.get('teamPosition') or p.get('individualPosition')) == 'UTILITY' else []) for p in match['info']['participants']}
    for f in timeline['info']['frames']:
        for e in f.get('events',[]):
            if e.get('type') == 'WARD_PLACED' and e.get('wardType') == 'CONTROL_WARD' and (e.get('creatorId'),e['timestamp']) not in destroyed_wards:
                bag = inventories.get(e.get('creatorId'),Counter())
                for ward in (2055,772043):
                    if bag[ward]>0:
                        bag[ward] -= 1
                        break
            inv = inventories.get(e.get('participantId'))
            if inv is None: continue
            kind, item = e.get('type'), e.get('itemId',0)
            if kind == 'ITEM_PURCHASED':
                if e['participantId'] in mid_complete:
                    item = boot_upgrades.get(item,item)
                if item in (3869,3870,3871,3876,3877):
                    for old in (3865,3866,3867,3869,3870,3871,3876,3877): inv[old] = 0
                if item == 3865 and inv[item]: continue
                inv[item] += 1
            elif kind in ('ITEM_DESTROYED','ITEM_SOLD'):
                held = inv[item]>0
                if kind == 'ITEM_DESTROYED' and e['participantId'] in bot_complete and ('Boots' in (dragon.item(item) or {}).get('tags',[]) or item in boot_upgrades.values()):
                    other_boot_buys = {i for i in purchases_at.get((e['participantId'],e['timestamp']),[]) if i != item and ('Boots' in (dragon.item(i) or {}).get('tags',[]) or i in boot_upgrades.values())}
                    if not other_boot_buys: continue
                inv[item] = max(0,inv[item]-1)
                if kind == 'ITEM_DESTROYED' and (item in boot_upgrades or item == 1001) and held:
                    if e['participantId'] in mid_complete:
                        if item in boot_upgrades: inv[boot_upgrades[item]] += 1
                    else:
                        pending_boots.setdefault((e['participantId'],e['timestamp']),[]).append(item)
                if kind == 'ITEM_DESTROYED' and item == 1201:
                    mid_complete.add(e['participantId'])
                    for old,new in boot_upgrades.items():
                        inv[new] += inv[old]
                        inv[old] = 0
                    for old in pending_boots.get((e['participantId'],e['timestamp']),[]):
                        if old in boot_upgrades: inv[boot_upgrades[old]] += 1
                if kind == 'ITEM_DESTROYED' and item == 1202:
                    bot_complete.add(e['participantId'])
                    for old in pending_boots.get((e['participantId'],e['timestamp']),[]): inv[old] += 1
                if kind == 'ITEM_DESTROYED' and item in (3865,3866,3867):
                    inv[{3865:3866,3866:3867,3867:3867}[item]] = 1
                elif kind == 'ITEM_DESTROYED' and item in (3869,3870,3871,3876,3877):
                    inv[3867] = 0
            elif kind == 'ITEM_UNDO':
                before, after = e.get('beforeId',0),e.get('afterId',0)
                if before: inv[before] = max(0,inv[before]-1)
                if after: inv[boot_upgrades.get(after,after) if e['participantId'] in mid_complete else after] += 1
    for p in match['info']['participants']:
        actual_ids = [p.get(f'item{i}') or 0 for i in range(7)]
        bound = p.get('roleBoundItem') or 0
        if bound not in actual_ids and (bound in (2055,772043) or 'Boots' in (dragon.item(bound) or {}).get('tags',[])):
            actual_ids.append(bound)
        actual = Counter(feature_inventory(actual_ids))
        replayed = Counter(feature_inventory(list(inventories[p['participantId']].elements())))
        for counter in (actual,replayed):
            for item in list(counter):
                if dragon.classify(item)['skip']: del counter[item]
                elif item in (2055,772043): counter[item] = min(counter[item],1) # DTO omits stack size
        counts['participants'] += 1
        if actual != replayed:
            counts['terminal_mismatches'] += 1
            if mid not in opaque:
                counts['terminal_mismatches_without_opaque_undo'] += 1
                affected.add(mid)
                extra, missing = sorted((replayed-actual).elements()),sorted((actual-replayed).elements())
                pairs[(tuple(extra),tuple(missing))] += 1
                if len(examples)<12: examples.append({'match_id':mid,'participant_id':p['participantId'],'extra':extra,'missing':missing})
out = ROOT / 'data/acceptance/16.18_initial/terminal_inventory_quest_hypothesis_v2.json'
report = {'corpus_manifest_sha256':sha256(corpus / 'corpus_manifest.json'),'counts':dict(counts),'games_without_opaque_undo_with_mismatch':sorted(affected),
    'common_differences':[{'extra':extra,'missing':missing,'n':n} for (extra,missing),n in pairs.most_common(20)],'examples':examples,
    'limits':['Role-quest slots and unobserved automatic transformations may explain final differences. This diagnostic neither repairs nor certifies an inventory.']}
write_json(out,report)
print(json.dumps({**report,'games_without_opaque_undo_with_mismatch':len(affected),'examples':examples[:3]},indent=2))
