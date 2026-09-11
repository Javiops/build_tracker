"""Offline tests: fixed-window pagination, dedup/cohorts, source rejection and lease."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import collect_patch_sample as sample
import freeze_patch_corpus as freeze_script
from app.collector_lock import CollectorLease
from app.riot import RiotClient, RawMatchArchive

with tempfile.TemporaryDirectory() as temp:
    destination = Path(temp) / 'state.json'
    real_replace = sample.os.replace
    failures = {'left': 2}
    def flaky_replace(source, target):
        if failures['left']:
            failures['left'] -= 1
            raise PermissionError('simulated Windows sharing violation')
        return real_replace(source, target)
    with patch.object(sample.os, 'replace', side_effect=flaky_replace):
        sample.save(destination, {'checkpoint': 1})
    assert json.loads(destination.read_text(encoding='utf-8')) == {'checkpoint': 1}
    assert not list(Path(temp).glob('*.tmp'))

class Pages:
    def __init__(self): self.calls = []
    def _get(self, url, params=None):
        self.calls.append(params.copy())
        return ['KR_' + str(i) for i in range(100)] if params['start'] == 0 else ['KR_100']
pages = Pages()
assert len(RiotClient.match_ids(pages, 'asia', 'person', 420, 0, start_time=1000, end_time=2000)) == 101
assert [p['start'] for p in pages.calls] == [0, 100]
assert all(p['startTime'] == 1000 and p['endTime'] == 2000 for p in pages.calls)

class Discovery:
    def __init__(self): self.calls = []
    def match_ids(self, route, puuid, queue, count, **kw):
        self.calls.append((route, puuid, count, kw))
        return ['KR_SHARED']
state = {'ladders': {'kr': [{'puuid':'same', 'tier':'CHALLENGER'}],
    'euw': [{'puuid':'euw', 'tier':'GRANDMASTER'}]},
    'tracked_pros': [{'league':'LCK', 'id':'Tracked'}],
    'cached_pros': [], 'pros': {'LCK:Tracked': {'puuid':'same'}},
    'lists': {}, 'start_time':1000, 'end_time':2000}
with tempfile.TemporaryDirectory() as temp:
    checkpoint = Path(temp) / 'state.json'
    client = Discovery()
    sample.discover(client, state, checkpoint)
    assert len(client.calls) == 2, 'pro/ladder overlap must not repeat player list'
    assert state['jobs'] == {'KR_SHARED': ['euw:GRANDMASTER', 'kr:CHALLENGER', 'pro:LCK:Tracked']}
    resumed = json.loads(checkpoint.read_text(encoding='utf-8'))
    sample.discover(client, resumed, checkpoint)
    assert len(client.calls) == 2, 'resume must not rediscover completed lists'
    resumed.update(patch='16.18',tracked_pros=[{'league':'LCK','id':'Tracked'},{'league':'LEC','id':'Unresolved'}])
    resumed['pros']['LEC:Unresolved'] = {'source':'unresolved'}
    sample.save(checkpoint,resumed)
    inherited = sample.inherited_cohorts(checkpoint,'16.18')
    assert inherited['pros'] == resumed['pros'] and inherited['ladders'] == resumed['ladders']
    assert 'lists' not in inherited and 'jobs' not in inherited, 'Fresh windows must rediscover match lists'
    lock_path = Path(temp) / 'lock'
    with CollectorLease(lock_path):
        try:
            CollectorLease(lock_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError('competing collector acquired lease')
    with CollectorLease(lock_path):
        pass

participants = [{'participantId':i, 'puuid':str(i)} for i in range(1,11)]
match = {'metadata':{'matchId':'KR_SAMPLE'}, 'info':{'gameVersion':'16.18.1',
    'queueId':420, 'gameDuration':60, 'participants':participants}}
timeline = {'metadata':{'matchId':'KR_SAMPLE'}, 'info':{'participants':participants,
    'frames':[{'timestamp':0, 'participantFrames':{str(i):{} for i in range(1,11)}}]}}
sample.validate_pair('KR_SAMPLE', match, timeline, '16.18')
for mutation in ('id', 'patch', 'duplicate', 'frame', 'identity', 'truncated'):
    m, t = copy.deepcopy(match), copy.deepcopy(timeline)
    if mutation == 'id': t['metadata']['matchId'] = 'KR_OTHER'
    if mutation == 'patch': m['info']['gameVersion'] = '16.17.1'
    if mutation == 'duplicate': m['info']['participants'][0]['puuid'] = '2'
    if mutation == 'frame': del t['info']['frames'][0]['participantFrames']['1']
    if mutation == 'identity': t['info']['participants'][0]['puuid'] = 'wrong'
    if mutation == 'truncated': m['info']['gameDuration'] = 1800
    try:
        sample.validate_pair('KR_SAMPLE', m, t, '16.18')
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted invalid source: ' + mutation)
print('ok sample: fixed cutoff pagination, pro/ladder dedup, resume, source validation, exclusive collector')

# Exercise real reconstruction + all ten perspectives in an isolated database.
dragon = sample.dragon_for_patch('16.18')
full_match = copy.deepcopy(match)
full_match['info'].update(gameCreation=1000000, gameDuration=180)
for p in full_match['info']['participants']:
    p.update(championId=103, championName='Ahri', teamId=100 if p['participantId'] <= 5 else 200,
             teamPosition='MIDDLE', win=p['participantId'] <= 5, kills=0, deaths=0, assists=0)
full_timeline = copy.deepcopy(timeline)
full_timeline['info']['frames'] = [
    {'timestamp':stamp, 'participantFrames':{str(i):{'participantId':i, 'currentGold':500,
        'totalGold':500, 'level':1, 'minionsKilled':0, 'jungleMinionsKilled':0} for i in range(1,11)},
     'events':[{'type':'ITEM_PURCHASED', 'participantId':i, 'itemId':1001, 'timestamp':stamp}
               for i in range(1,11)] if stamp == 120000 else []}
    for stamp in (0,60000,120000,180000)]
class Sources:
    def __init__(self): self.calls = 0
    def match(self, *args): self.calls += 1; return full_match
    def timeline(self, *args): self.calls += 1; return full_timeline
with tempfile.TemporaryDirectory() as temp:
    checkpoint = Path(temp) / 'state.json'
    state = {'patch':'16.18', 'static_data_sha256':dragon.signature,
        'reconstruction_version':sample.RECONSTRUCTION_VERSION, 'end_time':2000,
        'jobs':{'KR_SAMPLE':['pro:LCK:Tracked']}, 'listed_by':{'KR_SAMPLE':['1']},
        'pros':{'LCK:Tracked':{'puuid':'1'}}, 'results':{},'discovery_complete':True,
        'collection_code_sha256':sample.collection_code()[0]}
    source = Sources()
    original_db = sample.db.DB_PATH
    original_archive = freeze_script.RAW_MATCH_V5_DIR
    try:
        with patch.object(sample,'reconstruct_game',wraps=sample.reconstruct_game) as replay:
            report = sample.collect(source, state, checkpoint)
            assert replay.call_count == 1, 'The ten perspectives must share one reconstruction'
        assert report['structural_checks_passed'] and report['scope_complete']
        assert report['games'] == 1 and report['perspectives'] == 10
        assert report['counts'] == {'stored':1}
        sample.collect(source, state, checkpoint)
        assert source.calls == 2, 'resume must not fetch a stored source pair'
        freeze_script.RAW_MATCH_V5_DIR = Path(temp)/'raw'
        archive = RawMatchArchive(freeze_script.RAW_MATCH_V5_DIR)
        archive.write('match','KR_SAMPLE',full_match)
        archive.write('timeline','KR_SAMPLE',full_timeline)
        freeze_script.freeze(Path(temp),Path(temp)/'frozen')
        manifest = json.loads((Path(temp)/'frozen/corpus_manifest.json').read_text(encoding='utf-8'))
        assert manifest['inventory_version'] == 'causal-quests-v1'
        assert manifest['label_source_version'] == 'purchase-net-v1'
        assert 'app/inventory.py' in manifest['reconstruction_code_sha256']
    finally:
        sample.db.DB_PATH = original_db
        freeze_script.RAW_MATCH_V5_DIR = original_archive
        for path in Path(temp).rglob('*'):
            if path.is_file(): path.chmod(0o666)
print('ok sample end-to-end: real reconstruction, ten perspectives, database validation, restart dedup')
