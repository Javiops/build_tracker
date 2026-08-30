from collections import Counter

from app.config import FAKER
from app.db import get_meta, get_match, list_matches, init_db
from app.ddragon import DataDragon
from app.riot import RiotClient

init_db()
match = get_match(list_matches(get_meta("faker_puuid"))[0]["match_id"])
print("match", match["match_id"])
supports = [p for p in match["participants"] if p["team_position"] == "UTILITY"]
print("supports", [(p["champion_name"], p["participant_id"]) for p in supports])

client = RiotClient()
timeline = client.timeline("asia", match["match_id"])
full = client.match("asia", match["match_id"])
client.close()
dragon = DataDragon()

pid_map = {p["participantId"]: p.get("championName") for p in full["info"]["participants"]}
for p in full["info"]["participants"]:
    if p.get("teamPosition") != "UTILITY":
        continue
    pid = p["participantId"]
    print("\n===", p.get("championName"), "pid", pid, "===")
    counts = Counter()
    for frame in timeline["info"]["frames"]:
        for e in frame.get("events") or []:
            if e.get("participantId") != pid:
                continue
            if e.get("type") not in {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"}:
                continue
            item_id = e.get("itemId") or e.get("beforeId") or 0
            cls = dragon.classify(item_id)
            extra = ""
            if e["type"] == "ITEM_UNDO":
                extra = f" before={e.get('beforeId')} after={e.get('afterId')}"
            print(
                f"  {e['timestamp']/1000:7.1f}s {e['type']:<16} {item_id:5} {cls['name']:<28} skip={cls['skip']}{extra}"
            )
            counts[item_id] += 1
    print(" ids", dict(counts))
    end = [p2 for p2 in full["info"]["participants"] if p2["participantId"] == pid][0]
    print(" endgame items", [end.get(f"item{i}") for i in range(7)])
