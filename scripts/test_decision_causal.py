"""All serving input tensors must ignore changes after decision start."""
import copy
import contextlib
import io
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import torch
import baseline
import train_prefix as tp
from eval_policy import serving_row
from app.reconstruct import RECONSTRUCTION_VERSION, reconstruct_game

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(ROOT / "scripts/test_gold_causal.py"))
dragon = fixture["dragon"]

def timeline():
    tl = fixture["timeline"]()
    tl["info"]["frames"][2]["events"][0]["timestamp"] = 175000
    tl["info"]["frames"][3]["events"] = [
        {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1036, "timestamp": 185000}
    ]
    return tl

def row(tl):
    _, events = reconstruct_game(fixture["MATCH"], tl, dragon)
    event = next(e for e in events if e["participant_id"] == 8 and e["ts"] == 175000)
    assert event["reconstruction_version"] == RECONSTRUCTION_VERSION
    assert event["ts_end"] == 185000, "fixture must straddle the 180s frame"
    return baseline._example(event, dragon)

base = timeline()
future = copy.deepcopy(base)
future["info"]["frames"][2]["participantFrames"]["8"].update(
    currentGold=9000, totalGold=99000, level=18, minionsKilled=999)
future["info"]["frames"][2]["events"].extend([
    {"type": "CHAMPION_KILL", "killerId": 8, "victimId": 1, "timestamp": 178000, "bounty": 900},
    {"type": "BUILDING_KILL", "teamId": 100, "buildingType": "TOWER_BUILDING", "timestamp": 179000},
    {"type": "ELITE_MONSTER_KILL", "killerId": 8, "monsterType": "DRAGON", "timestamp": 179500},
])
future["info"]["frames"][3]["events"][0]["itemId"] = 1001
before, after = row(base), row(future)
assert before["label_ids"] != after["label_ids"], "future purchase must change the target"
for key in ("gold", "gold_est", "level", "cs", "kills", "deaths", "ally_obj", "enemy_obj", "others", "inventory"):
    assert before[key] == after[key], key

def inputs(r, goldx):
    clean = serving_row(r, r["gold_est"] if goldx else r["gold"])
    ds = tp.ShopDataset([clean], {103: 1, 64: 2}, {1056: 1, 1026: 2, 1036: 3, 1001: 4, 3867: 5, 3869: 6},
                        {1026: 0, 1036: 1, 1001: 2}, dragon)
    return {key: getattr(ds, key).clone() for key in
            ("champs", "sides", "items", "hist", "query", "inv", "legal", "budget")}

for goldx, interpolate in ((False, False), (False, True), (True, True)):
    tp.apply_config({"gold_x": goldx, "gold_est": interpolate, "extras": True,
                     "runes": True, "query_dim": tp.BASE_QUERY_DIM + tp.RUNE_DIM})
    a, b = inputs(before, goldx), inputs(after, goldx)
    for key in a:
        assert torch.equal(a[key], b[key]), (goldx, interpolate, key)
    prior = copy.deepcopy(base)
    prior["info"]["frames"][2]["events"].insert(0,
        {"type": "CHAMPION_KILL", "killerId": 8, "victimId": 1, "timestamp": 170000, "bounty": 300})
    assert not torch.equal(a["query"], inputs(row(prior), goldx)["query"]), "pre-decision control is dead"
tp.apply_config({})
print("ok decision causality: future frame, KDA, objectives and buys cannot change serving inputs; pre-decision controls do")

# Even a frame tied to the decision may already reflect its purchase.
tied = fixture['timeline']()
tied['info']['frames'][2]['events'][0]['timestamp'] = 180000
_, tied_events = reconstruct_game(fixture['MATCH'],tied,dragon)
tied_event = next(e for e in tied_events if e['ts'] == 180000)
assert tied_event['gold_gap']['frame_ts'] == 120000
changed_tie = copy.deepcopy(tied)
changed_tie['info']['frames'][2]['participantFrames']['8']['currentGold'] = 99000
assert next(e for e in reconstruct_game(fixture['MATCH'],changed_tie,dragon)[1] if e['ts'] == 180000)['gold_est'] == tied_event['gold_est']

# End-of-game slots cannot choose an earlier support variant or seed a
# non-support's initial inventory. These mutations used to alter the board.
support_match = copy.deepcopy(fixture['MATCH'])
support_match['info']['participants'][1]['teamPosition'] = 'UTILITY'
support_tl = timeline()
support_tl['info']['frames'][1]['events'] = [
    {'type':'ITEM_DESTROYED','participantId':1,'itemId':item,'timestamp':ts}
    for item, ts in ((3865,100000),(3866,101000),(3867,102000))]
changed_match = copy.deepcopy(support_match)
for p in changed_match['info']['participants']:
    p.update(item0=3869,kills=999,deaths=999,assists=999,win=not p['win'])
original_events = reconstruct_game(support_match,support_tl,dragon)[1]
assert original_events == reconstruct_game(changed_match,support_tl,dragon)[1]

# A directly observed support choice is coarsened in the shared tensor path,
# keeping offline and live inputs equal without rewriting caller data.
proxy = copy.deepcopy(before)
observed = copy.deepcopy(before)
proxy['others'][0]['items'] = [3867]
observed['others'][0]['items'] = [3869]
for key, value in inputs(proxy,False).items():
    assert torch.equal(value,inputs(observed,False)[key]), key
assert observed['others'][0]['items'] == [3869]
print('ok tied-frame exclusion, final-DTO invariance, support input parity')
