"""Independent semantic oracle must separate inhibitors/towers and time prefixes."""
from bisect import bisect_left, bisect_right
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from audit_patch_corpus import combat_context

events = [
    {'type':'BUILDING_KILL','timestamp':100,'teamId':100,'buildingType':'TOWER_BUILDING'},
    {'type':'BUILDING_KILL','timestamp':200,'teamId':100,'buildingType':'INHIBITOR_BUILDING'},
    {'type':'CHAMPION_KILL','timestamp':300,'killerId':1,'victimId':2,'assistingParticipantIds':[3]},
]
participants = [{'participantId':i,'teamId':100 if i == 2 else 200} for i in (1,2,3)]
times, states = combat_context({'info':{'frames':[{'events':events}]}}, participants)
assert states[-1][1]['200']['towers'] == 1
assert states[bisect_left(times,300)-1][0][1][0] == 0
assert states[bisect_right(times,300)-1][0][1][0] == 1
assert states[-1][0][2][1] == 1 and states[-1][0][3][2] == 1
assert states[0][0][1] == [0,0,0], 'later events must not mutate earlier snapshots'
from audit_patch_corpus import quest_completion_times
quest = lambda ts: {'type':'ITEM_DESTROYED','itemId':1201,'participantId':8,'timestamp':ts}
assert quest_completion_times({'info':{'frames':[{'events':[quest(100),quest(200),quest(90)]}]}},1201) == {8:90}
assert quest_completion_times({'info':{'frames':[{'events':[quest(100),quest(200),quest(900)]}]}},1201) == {8:100}
print('ok independent corpus oracle: tower type, KDA boundary, snapshot isolation, repeated quest marker')
