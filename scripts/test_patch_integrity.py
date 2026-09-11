"""Patch selection, static provenance, and transaction rollback; no Riot calls."""
import contextlib
import copy
import io
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import db, ladder
from app.config import selected_patches, patch_start_unix
from app.ddragon import dragon_for_patch
from app.reconstruct import reconstruct_game, reconstruct_visits

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(ROOT / 'scripts/test_gold_causal.py'))
match, timeline = fixture['MATCH'], fixture['timeline']()
assert selected_patches() == ('16.18',)
assert selected_patches('16.17,16.18,16.17') == ('16.17', '16.18')
assert patch_start_unix('16.17,16.18') == patch_start_unix('16.17')
for value in ('16.19', '16.17,16.19'):
    try:
        selected_patches(value)
    except ValueError:
        pass
    else:
        raise AssertionError('Unreviewed patch accepted')

for actual, supplied in (('16.17', '16.18'), ('16.18', '16.17')):
    dto = copy.deepcopy(match)
    dto['info']['gameVersion'] = actual + '.1.1'
    game, events = reconstruct_game(dto, timeline, dragon_for_patch(supplied))
    assert game['patch'] == actual
    assert events
    for event in events:
        assert event['patch'] == actual
        assert event['ddragon_version'] == dragon_for_patch(actual).version
        assert event['static_data_sha256'] == dragon_for_patch(actual).signature

class Client:
    def match(self, *args):
        return match
    def timeline(self, *args):
        raise AssertionError('Wrong-patch match fetched a timeline')

with patch.object(ladder, 'game_exists', return_value=False), patch.object(ladder, 'match_puuids', return_value=set()):
    assert ladder._ingest_match_perspectives(Client(), dragon_for_patch('16.18'), 'asia', 'kr',
                                           'KR_PATCH_TEST', '16.18', {'faker'}) == (0, 1)

dragon = dragon_for_patch('16.17')
game, events = reconstruct_game(match, timeline, dragon)
participant = match['info']['participants'][0]['puuid']
row, visits = reconstruct_visits(match, timeline, participant, dragon)
with tempfile.TemporaryDirectory() as tmp, patch.object(db, 'DB_PATH', Path(tmp) / 'test.db'):
    db.init_db()
    for p in match['info']['participants']:
        db.upsert_player(ladder._player_from_match(p, 'kr', 'asia'))
    db.insert_reconstruction(game, events, [(row, visits)])
    with db.db() as conn:
        original = [tuple(r) for r in conn.execute('SELECT * FROM game_events')]
    invalid = dict(row)
    del invalid['champion_id']
    try:
        db.insert_reconstruction(game, [], [(invalid, visits)])
    except Exception as exc:
        assert 'champion_id' in str(exc), str(exc)
    else:
        raise AssertionError('Invalid perspective unexpectedly written')
    with db.db() as conn:
        assert [tuple(r) for r in conn.execute('SELECT * FROM game_events')] == original
        assert conn.execute('SELECT count(*) FROM matches').fetchone()[0] == 1

print('ok patch integrity: DTO-bound static data, early rejection, atomic rollback')
