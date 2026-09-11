"""Isolated, resumable whole-window patch sample. No export, training or promotion."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import db
from app.collector_lock import CollectorLease
from app.config import DATA_DIR, RANKED_SOLO_QUEUE, patch_start_unix
from app.ddragon import dragon_for_patch
from app.ladder import top_solo_ladder, routing_for_match_id, _player_from_match
from app.pros import PROS, RESOLVE_PATH, regionals_for
from app.reconstruct import RECONSTRUCTION_VERSION, patch_from_version, reconstruct_game, perspective_from_game
from app.riot import RiotClient, RiotError


def save(path, value):
    # Windows scanners/readers can briefly hold the destination and make
    # os.replace fail with ERROR_ACCESS_DENIED. A unique temp avoids collisions
    # between retries/processes; bounded retries retain atomic publication.
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    try:
        for attempt in range(8):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        if temp.exists():
            temp.unlink()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def collection_code():
    files = ('scripts/collect_patch_sample.py', 'app/reconstruct.py', 'app/inventory.py',
             'app/ddragon.py', 'app/config.py', 'app/shop_econ.py', 'app/riot.py', 'app/db.py')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    return hashes, digest(hashes)


def inherited_cohorts(path, patch):
    """Freeze the same ladder/pro cohort for the next cumulative observation."""
    prior = json.loads(path.read_text(encoding='utf-8'))
    if prior.get('patch') != patch or not prior.get('discovery_complete'):
        raise ValueError('Cohort parent must be a completed same-patch discovery')
    ladders, pros, tracked = prior.get('ladders',{}),prior.get('pros',{}),prior.get('tracked_pros',[])
    if set(ladders) != {'kr','euw'} or any(not entries for entries in ladders.values()):
        raise ValueError('Cohort parent lacks either regional ladder')
    if not tracked or set(pros) != {p['league']+':'+p['id'] for p in tracked}:
        raise ValueError('Cohort parent has incomplete tracked-pro resolution outcomes')
    return {'ladders':ladders,'pros':pros,'tracked_pros':tracked,
            'roster_observed_utc':prior.get('roster_observed_utc',{}),
            'cohort_parent':str(path.resolve()),'cohort_parent_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'cohort_policy':'same original ladder roster and tracked-pro resolution outcomes; unresolved accounts remain accepted limitations'}


def validate_pair(mid, match, timeline, patch):
    if not match or not timeline:
        raise ValueError('missing source pair')
    if any(p.get('metadata', {}).get('matchId') != mid for p in (match, timeline)):
        raise ValueError('source payload match ID mismatch')
    info = match['info']
    if patch_from_version(info.get('gameVersion', '')) != patch or info.get('queueId') != RANKED_SOLO_QUEUE:
        raise ValueError('wrong patch or queue')
    ps = info.get('participants', [])
    if len(ps) != 10 or {p.get('participantId') for p in ps} != set(range(1, 11)) or len({p.get('puuid') for p in ps if p.get('puuid')}) != 10:
        raise ValueError('incomplete/duplicate participant identities')
    frames = timeline.get('info', {}).get('frames', [])
    stamps = [f.get('timestamp', -1) for f in frames]
    if not frames or stamps != sorted(set(stamps)) or stamps[0] < 0:
        raise ValueError('missing or unordered timeline frames')
    if any(set(f.get('participantFrames', {})) != {str(i) for i in range(1, 11)} for f in frames):
        raise ValueError('incomplete timeline participant frames')
    if stamps[-1] < info.get('gameDuration', 0) * 1000 - 120000:
        raise ValueError('timeline ends materially before match duration')
    if {p.get('participantId'): p.get('puuid') for p in timeline.get('info', {}).get('participants', [])} != {p['participantId']: p['puuid'] for p in ps}:
        raise ValueError('timeline participant identities disagree with match')


class SampleClient(RiotClient):
    def _get(self, url, params=None):
        value = super()._get(url, params)
        if value is None and (url.endswith('/ids') or 'leagues/by-queue/' in url):
            raise RiotError('Roster/list endpoint unavailable; cannot certify discovery', 404)
        if url.endswith('/ids') and (not isinstance(value, list) or any(not isinstance(mid, str) for mid in value)):
            raise ValueError('Malformed match-list response')
        if 'leagues/by-queue/' in url and (not isinstance(value, dict) or not isinstance(value.get('entries'), list)
                or any(not p.get('puuid') for p in value['entries'])):
            raise ValueError('Malformed/incomplete ladder account response')
        return value


def discover(client, state, checkpoint):
    def persist():
        save(checkpoint, state)
    # Freeze each response before moving on. No player-list caps.
    for region, platform in [('kr', 'kr'), ('euw', 'euw1')]:
        if region not in state['ladders']:
            state['ladders'][region] = top_solo_ladder(client, platform, 0, tiers=('challenger', 'grandmaster'))
            if not state['ladders'][region]:
                raise ValueError('Empty ladder response; review before proceeding')
            state.setdefault('roster_observed_utc', {})[region] = datetime.now(timezone.utc).isoformat()
            persist()
    for pro in state['tracked_pros']:
        key = pro['league'] + ':' + pro['id']
        if key in state['pros']:
            continue
        cached = next((p for p in state['cached_pros'] if p.get('league') == pro['league'] and p.get('id') == pro['id'] and p.get('ok') and p.get('puuid')), None)
        if cached:
            resolved = {'puuid': cached['puuid'], 'source': 'tracked resolved-account snapshot'}
        else:
            resolved = {'source': 'unresolved'}
            for name, tag in pro.get('ids', []):
                for route in regionals_for(pro['league']):
                    try:
                        account = client.account_by_riot_id(route, name, tag)
                    except RiotError as exc:
                        if exc.status != 404:
                            raise
                        continue
                    resolved = {'puuid': account['puuid'], 'source': 'account-v1 candidate ID'}
                    break
                if resolved.get('puuid'):
                    break
        state['pros'][key] = resolved
        persist()
    requests = {}
    def add(route, puuid, cohort):
        requests.setdefault((route, puuid), set()).add(cohort)
    for region, roster in state['ladders'].items():
        for player in roster:
            add('asia' if region == 'kr' else 'europe', player['puuid'], region + ':' + player['tier'])
    for pro in state['tracked_pros']:
        key = pro['league'] + ':' + pro['id']
        resolved = state['pros'][key]
        if resolved.get('puuid'):
            for route in regionals_for(pro['league']):
                add(route, resolved['puuid'], 'pro:' + key)
    jobs, listed_by = {}, {}
    for (route, puuid), cohorts in sorted(requests.items()):
        key = route + ':' + puuid
        if key not in state['lists']:
            state['lists'][key] = client.match_ids(route, puuid, RANKED_SOLO_QUEUE, 0,
                start_time=state['start_time'], end_time=state['end_time'])
            persist()
            print(f"discovery {len(state['lists'])}/{len(requests)} player-route lists", flush=True)
        for mid in state['lists'][key]:
            jobs.setdefault(mid, set()).update(cohorts)
            listed_by.setdefault(mid, set()).add(puuid)
    state['jobs'] = {mid: sorted(cohorts) for mid, cohorts in sorted(jobs.items())}
    state['listed_by'] = {mid: sorted(players) for mid, players in listed_by.items()}
    state['discovery_complete'] = True
    persist()


def collect(client, state, checkpoint):
    dragon = dragon_for_patch(state['patch'])
    if dragon.signature != state['static_data_sha256'] or state['reconstruction_version'] != RECONSTRUCTION_VERSION:
        raise ValueError('Pinned reconstruction/static provenance changed; use a new sample')
    db.DB_PATH = checkpoint.parent / 'tracker.db'
    db.init_db()
    for mid, cohorts in state['jobs'].items():
        if mid in state['results']:
            continue
        route, platform = routing_for_match_id(mid)
        try:
            match = client.match(route, mid)
            if not match:
                raise ValueError('match unavailable')
            if match.get('metadata', {}).get('matchId') != mid:
                raise ValueError('match payload ID mismatch')
            info = match.get('info', {})
            if patch_from_version(info.get('gameVersion', '')) != state['patch'] or info.get('queueId') != RANKED_SOLO_QUEUE:
                result = {'status': 'excluded_patch_or_queue', 'patch': patch_from_version(info.get('gameVersion', ''))}
            else:
                timeline = client.timeline(route, mid)
                validate_pair(mid, match, timeline, state['patch'])
                if not set(state['listed_by'][mid]).issubset({p['puuid'] for p in info['participants']}):
                    raise ValueError('match-list account missing from match participants')
                game, events = reconstruct_game(match, timeline, dragon)
                if any(e.get('reconstruction_version') != RECONSTRUCTION_VERSION or e.get('static_data_sha256') != dragon.signature or e.get('patch') != state['patch'] for e in events):
                    raise ValueError('reconstruction provenance mismatch')
                entries = [perspective_from_game(game, events, p['puuid']) for p in info['participants']]
                for p in info['participants']:
                    db.upsert_player(_player_from_match(p, platform, route))
                db.insert_reconstruction(game, events, entries)
                result = {'status': 'stored', 'events': len(events), 'perspectives': len(entries),
                    'match_sha256': digest(match), 'timeline_sha256': digest(timeline), 'cohorts': cohorts}
        except RiotError:
            raise  # Auth/retry failures stop the run; never mark discovery complete falsely.
        except (ValueError, KeyError, TypeError) as exc:
            result = {'status': 'quarantined', 'reason': str(exc), 'cohorts': cohorts}
        state['results'][mid] = result
        save(checkpoint, state)
        print(f"sample {len(state['results'])}/{len(state['jobs'])}: {mid} {result['status']}", flush=True)
    with db.db() as conn:
        integrity = [r[0] for r in conn.execute('PRAGMA integrity_check')]
        foreign_keys = [tuple(r) for r in conn.execute('PRAGMA foreign_key_check')]
        games = conn.execute('SELECT count(*) FROM games').fetchone()[0]
        perspectives = conn.execute('SELECT count(*) FROM matches').fetchone()[0]
        bad_lobbies = conn.execute('SELECT count(*) FROM (SELECT match_id FROM game_players GROUP BY match_id HAVING count(*) != 10)').fetchone()[0]
    counts = dict(Counter(r['status'] for r in state['results'].values()))
    unresolved = [key for key, value in state['pros'].items() if not value.get('puuid')]
    valid = integrity == ['ok'] and not foreign_keys and not bad_lobbies and games == counts.get('stored', 0) and perspectives == games * 10
    report = {'patch': state['patch'], 'cutoff_utc': datetime.fromtimestamp(state['end_time'], timezone.utc).isoformat(),
        'population': 'frozen KR/EUW Challenger+Grandmaster roster responses plus tracked pro accounts; not all global high-elo games',
        'roster_observed_utc': state.get('roster_observed_utc', {}),
        'listed_player_routes': len(state.get('lists', {})), 'discovered_unique_matches': len(state['jobs']),
        'tracked_pros': len(state['pros']), 'resolved_pros': len(state['pros']) - len(unresolved),
        'stored_games_by_cohort': dict(Counter(c for r in state['results'].values() if r['status'] == 'stored' for c in r['cohorts'])),
        'cohort_count_note': 'Overlapping memberships; cohort counts must not be summed as unique games.',
        'counts': counts, 'unresolved_pros': unresolved, 'db_integrity': integrity,
        'foreign_key_errors': foreign_keys, 'bad_lobbies': bad_lobbies, 'games': games,
        'perspectives': perspectives, 'structural_checks_passed': valid,
        'scope_complete': valid and bool(games) and not unresolved and not counts.get('quarantined'),
        'reconstruction_version': RECONSTRUCTION_VERSION, 'static_data_sha256': dragon.signature,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'limitations': 'Structural/source checks only; not proof of exact wallets or absence of all reconstruction errors. No training, export, or promotion.'}
    save(checkpoint.parent / 'report.json', report)
    if not valid:
        raise ValueError('Staging database failed structural checks')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true', help='Freeze scope/cutoff without contacting Riot')
    parser.add_argument('--reuse-cohorts-from',type=Path,help='For a new sample, keep the completed parent roster and pro identities')
    parser.add_argument('--accept-code-sha', help='One reviewed code-bundle digest allowed to replace the checkpoint binding')
    parser.add_argument('--print-code-sha', action='store_true')
    args = parser.parse_args()
    current_code, current_code_sha = collection_code()
    if args.print_code_sha:
        print(current_code_sha)
        return
    out = args.sample.resolve()
    parent = (DATA_DIR / 'patch_samples').resolve()
    if not out.is_relative_to(parent) or out == parent:
        raise SystemExit('Sample must be a dedicated child of data/patch_samples')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'state.json'
    if checkpoint.exists() and args.reuse_cohorts_from:
        raise SystemExit('Cohort inheritance is only allowed when creating a new sample')
    if not checkpoint.exists():
        dragon = dragon_for_patch('16.18')
        cohorts = inherited_cohorts(args.reuse_cohorts_from.resolve(),'16.18') if args.reuse_cohorts_from else {}
        save(checkpoint, {'patch': '16.18', 'start_time': patch_start_unix('16.18'),
            'end_time': int(datetime.now(timezone.utc).timestamp()), 'reconstruction_version': RECONSTRUCTION_VERSION,
            'static_data_sha256': dragon.signature, 'tracked_pros': PROS,
            'cached_pros': json.loads(RESOLVE_PATH.read_text(encoding='utf-8')) if RESOLVE_PATH.exists() else [],
            'ladders': {}, 'pros': {}, 'lists': {}, 'jobs': {}, 'results': {}, 'discovery_complete': False,**cohorts})
    if args.prepare:
        print(f'Prepared {checkpoint}; no Riot requests')
        return
    # Old running jobs predate the shared lease. Fail closed until their
    # supervisor has exited; the queued wrapper handles waiting.
    if os.name == 'nt':
        check = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            "$ErrorActionPreference='Stop'; $tasks=Get-ScheduledTask -TaskName 'BuildTracker Causal Recovery','BuildTracker Daily Pull'; if ($tasks | Where-Object State -eq 'Running') { exit 3 }"],
            capture_output=True, timeout=30)
        if check.returncode:
            raise SystemExit('Recovery/daily task still active or scheduler state unavailable; sample not started')
    with CollectorLease():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        prior_code = state.get('collection_code_sha256', current_code)
        if prior_code != current_code and args.accept_code_sha != current_code_sha:
            raise SystemExit(f'Collection implementation changed; review and pass --accept-code-sha {current_code_sha}')
        if prior_code != current_code:
            state.setdefault('accepted_code_changes', []).append({
                'at_utc': datetime.now(timezone.utc).isoformat(), 'code_bundle_sha256': current_code_sha,
                'reason': 'Reviewed checkpoint publication amendment',
            })
        state['collection_code_sha256'] = current_code
        save(checkpoint, state)
        client = SampleClient(raw_cache_enabled=True)
        try:
            client._get('https://kr.api.riotgames.com/lol/status/v4/platform-data')
            if not state['discovery_complete']:
                discover(client, state, checkpoint)
            print(json.dumps(collect(client, state, checkpoint), indent=2), flush=True)
        finally:
            client.close()


if __name__ == '__main__':
    main()
