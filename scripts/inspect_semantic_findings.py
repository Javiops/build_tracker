"""Print bounded raw evidence for inventory gains lacking purchase events."""
import json
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.riot import RawMatchArchive
corpus = ROOT / 'data/corpora/16.18_initial'
archive = RawMatchArchive(corpus / 'raw')
report = json.loads((ROOT / 'data/acceptance/16.18_initial/semantic_audit.json').read_text())
conn = sqlite3.connect((corpus / 'tracker.db').as_uri() + '?mode=ro&immutable=1', uri=True)
for finding in report['examples']['net_inventory_gain_without_purchase_event'][:5]:
    mid, ts = finding['match_id'], finding['ts']
    row = conn.execute('SELECT payload_json FROM game_events WHERE match_id=? AND ts=?', (mid,ts)).fetchone()
    event = json.loads(row[0])
    timeline = archive.read('timeline', mid)
    raw = [e for f in timeline['info']['frames'] for e in f.get('events', [])
           if ts <= e.get('timestamp', 0) <= event['ts_end'] and e.get('participantId') == event['participant_id']]
    print(json.dumps({'finding':finding, 'end':event['ts_end'], 'bought':event['bought'], 'raw':raw}, ensure_ascii=True))
conn.close()
