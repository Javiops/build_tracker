"""Properties for role slots, grants, wards, and reversing multi-part combines."""
import contextlib
import copy
import io
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
from app.ddragon import dragon_for_patch
from app.inventory import InventoryReplay
from app.reconstruct import reconstruct_game
from app.shop_econ import payload_ids
import baseline

dragon = dragon_for_patch('16.18')
def event(kind,item,ts=1000):
    return {'type':kind,'itemId':item,'participantId':1,'timestamp':ts}

for ordered in ((3158,1201),(1201,3158)):
    inventories = {1:[3158]}
    ledger = InventoryReplay(inventories,dragon)
    actions = ledger.apply_batch(1,1000,[event('ITEM_DESTROYED',i) for i in ordered])
    assert inventories[1] == [3171] and not actions
    assert dragon.classify(3172)['is_boots'], 'static tag omission bypassed boot family'
    assert ledger.apply_batch(1,2000,[event('ITEM_PURCHASED',3171,2000)]) == {}, 'free upgrade counted twice'

# No boots held at quest completion; a later paid boot purchase upgrades once.
inventories = {1:[1001,2022]}
ledger = InventoryReplay(inventories,dragon)
ledger.apply_batch(1,1000,[event('ITEM_DESTROYED',1201)])
actions = ledger.apply_batch(1,2000,[event('ITEM_DESTROYED',1001,2000),event('ITEM_DESTROYED',2022,2000),event('ITEM_PURCHASED',3158,2000)])
assert inventories[1] == [3171] and actions == {3171:1}

for ordered in ((3006,1202),(1202,3006)):
    inventories = {1:[3006]}
    ledger = InventoryReplay(inventories,dragon)
    ledger.apply_batch(1,1000,[event('ITEM_DESTROYED',i) for i in ordered])
    assert inventories[1] == [3006], 'role-slot transfer destroyed owned boots'

# Slot movement applies to later purchases too, without preserving consumed
# basic boots when the same transaction upgrades them.
inventories = {1:[]}
ledger = InventoryReplay(inventories,dragon)
ledger.apply_batch(1,1000,[event('ITEM_DESTROYED',1202)])
ledger.apply_batch(1,2000,[event('ITEM_PURCHASED',1001,2000),event('ITEM_DESTROYED',1001,2000)])
assert inventories[1] == [1001]
ledger.apply_batch(1,3000,[event('ITEM_DESTROYED',1001,3000),event('ITEM_PURCHASED',3006,3000),event('ITEM_DESTROYED',3006,3000)])
assert inventories[1] == [3006]

# A deterministic automatic transformation is context, not a purchase target.
inventories = {1:[3003]}
ledger = InventoryReplay(inventories,dragon)
assert not ledger.apply_batch(1,1000,[event('ITEM_DESTROYED',3003)])
assert inventories[1] == [3040]
assert not dragon.classify(3040)['skip'] and dragon.classify(3040)['target_skip']
assert baseline._label([3040],dragon) is None

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(ROOT / 'scripts/test_gold_causal.py'))
match = copy.deepcopy(fixture['MATCH'])
tl = fixture['timeline']()
tl['info']['frames'][1]['events'] = [
    {'type':'ITEM_PURCHASED','participantId':8,'itemId':i,'timestamp':100000}
    for i in (3145,3113)]
tl['info']['frames'][2]['events'] = [
    {'type':'ITEM_DESTROYED','participantId':8,'itemId':i,'timestamp':150000}
    for i in (3145,3113)] + [
    {'type':'ITEM_PURCHASED','participantId':8,'itemId':4646,'timestamp':150000},
    {'type':'ITEM_UNDO','participantId':8,'beforeId':4646,'afterId':3145,'goldGain':800,'timestamp':151000},
    {'type':'ITEM_PURCHASED','participantId':8,'itemId':1026,'timestamp':152000}]
_, visits = reconstruct_game(match,tl,fixture['dragon'])
visit = next(e for e in visits if e['ts'] == 150000)
assert sorted(payload_ids(visit['inventory_after'])) == [1026,1056,3113,3145]
assert payload_ids(visit['label_bought']) == [1026], 'restored components became new targets'

# WARD_PLACED consumes a control ward when no destroy is supplied. A supplied
# same-time ITEM_DESTROYED must not consume a second ward.
for explicit_destroy in (False,True):
    tl = fixture['timeline']()
    tl['info']['frames'][1]['events'] = [
        {'type':'ITEM_PURCHASED','participantId':8,'itemId':2055,'timestamp':100000} for _ in range(2)]
    tl['info']['frames'][2]['events'].insert(0,{'type':'WARD_PLACED','creatorId':8,'wardType':'CONTROL_WARD','timestamp':130000})
    if explicit_destroy:
        tl['info']['frames'][2]['events'].insert(1,{'type':'ITEM_DESTROYED','participantId':8,'itemId':2055,'timestamp':130000})
    _, visits = reconstruct_game(match,tl,fixture['dragon'])
    before = payload_ids(next(e for e in visits if e['ts'] == 150000)['inventory_before'])
    assert before.count(2055) == 1
print('ok role-quest order, later boot purchases, role slots, automatic transforms, combine undo, ward consumption')
