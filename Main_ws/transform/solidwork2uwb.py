from driver.referee.referee_comm import FACTION
from typing import List



def solidwork2uwb(pos_3d: List[float], faction: FACTION = FACTION.RED) -> List[float]:
    x, y, z = pos_3d
    # faction = FACTION.RED
    if faction == FACTION.RED :
        pos_x = x
        pos_y = y
    else:
        pos_x = 28.0 - x
        pos_y = 15.0 - y
    return pos_x, pos_y