"""Inventory gains from support grants and undo are not purchase labels."""
import contextlib
import copy
import io
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.reconstruct import reconstruct_game
from app.shop_econ import payload_ids
import baseline

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(ROOT / 'scripts/test_gold_causal.py'))
match = copy.deepcopy(fixture['MATCH'])
match['info']['participants'][0]['teamPosition'] = 'UTILITY'
tl = fixture['timeline']()
tl['info']['frames'][2]['events'].extend([
    {'type':'ITEM_DESTROYED','participantId':8,'itemId':3865,'timestamp':150500},
    {'type':'BUILDING_KILL','teamId':100,'buildingType':'INHIBITOR_BUILDING','timestamp':160000},
    {'type':'BUILDING_KILL','teamId':100,'buildingType':'TOWER_BUILDING','timestamp':165000},
    {'type':'ITEM_UNDO','participantId':8,'afterId':3041,'beforeId':0,'goldGain':-1050,'timestamp':170000},
])
_, events = reconstruct_game(match, tl, fixture['dragon'])
buy = next(e for e in events if e['participant_id'] == 8 and e['ts'] == 150000)
assert 3866 in payload_ids(buy['bought'])
assert payload_ids(buy['label_bought']) == [1026]
assert baseline._example(buy, fixture['dragon'])['label_ids'] == [1026]
undo = next(e for e in events if e['participant_id'] == 8 and e['ts'] == 170000)
assert payload_ids(undo['bought']) == [3041]
assert undo['label_bought'] == [] and baseline._example(undo, fixture['dragon']) is None
assert undo['score'][200]['towers'] == 1, 'inhibitor counted as tower'
print('ok source-backed labels: real buy kept, support grant and sale undo excluded, only towers counted')

# Count labels, not just membership. A buy and its reversal cancel; repeated
# real buys remain repeated targets. An ambiguous reversal excludes both teams.
repeat = fixture['timeline']()
repeat['info']['frames'][2]['events'].append(
    {'type':'ITEM_PURCHASED','participantId':8,'itemId':1026,'timestamp':151000})
_, repeated = reconstruct_game(match, repeat, fixture['dragon'])
assert payload_ids(next(e for e in repeated if e['ts'] == 150000)['label_bought']) == [1026,1026]
cancel = fixture['timeline']()
cancel['info']['frames'][2]['events'].append(
    {'type':'ITEM_UNDO','participantId':8,'beforeId':1026,'afterId':0,'goldGain':850,'timestamp':151000})
_, cancelled = reconstruct_game(match,cancel,fixture['dragon'])
assert not any(e['ts'] == 150000 for e in cancelled)
opaque = fixture['timeline']()
opaque['info']['frames'][3]['events'].append(
    {'type':'ITEM_UNDO','participantId':1,'beforeId':0,'afterId':0,'goldGain':600,'timestamp':200000})
_, uncertain = reconstruct_game(match,opaque,fixture['dragon'])
assert uncertain and all(e['training_eligible'] is False for e in uncertain)
assert all(baseline._example(e,fixture['dragon']) is None for e in uncertain)
print('ok duplicate labels, undone purchase cancellation, and whole-board ambiguity exclusion')

# Exact prior transaction deltas include undoing a sale. No future transaction
# may affect the earlier wallet. Observe the ledger directly, not a tautology
# between reconstructed outputs.
sale = fixture['timeline']()
sale['info']['frames'][2]['events'].insert(0,
    {'type':'ITEM_UNDO','participantId':8,'beforeId':0,'afterId':1056,'goldGain':-160,'timestamp':135000})
debug = {}
_, sale_events = reconstruct_game(match,sale,fixture['dragon'],debug)
assert (135000,160.0,'spend') in debug['gold_events'][8]
original = next(e for e in reconstruct_game(match,fixture['timeline'](),fixture['dragon'])[1] if e['ts'] == 150000)
reversed_sale = next(e for e in sale_events if e['ts'] == 150000)
assert original['gold_est'] - reversed_sale['gold_est'] == 160
print('ok sale undo charges the source-reported wallet delta')
