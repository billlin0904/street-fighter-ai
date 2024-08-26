import numpy as np
from enum import Enum

class Punch(Enum):
    LP = 9
    MP = 10
    HP = 11
    
class Kick(Enum):
    LK = 0
    MK = 1
    HK = 8
    
class Moves(Enum):
    MODE = 2
    START = 3
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7
    
from enum import Enum

class Status(Enum):
    STANDING = 512
    CROUCHING = 514
    JUMPING = 516
    BLOCKING = 518
    NORMAL_ATTACK = 522
    SPECIAL_ATTACK = 524
    HIT_STUN = 526
    THROWN = 532

class Fighter:
    def __init__(self, info):
        if info != None:
            agent_x = info.get('agent_x', 0)
            enemy_x = info.get('enemy_x', 0)
            is_facing_right = agent_x < enemy_x  
            self.is_facing_right = is_facing_right
            self._distance = abs(enemy_x - agent_x)
            self.agent_status = info.get('agent_status', 0)
            self.enemy_status = info.get('enemy_status', 0)
        else:
            self.is_facing_right = True
            self._distance = 145
            self.agent_status = Status.STANDING.value
            self.enemy_status = Status.STANDING.value
            
    @property
    def distance(self):
        return self._distance
    
    @property
    def is_standing(self):
        return self.agent_status == Status.STANDING.value
    
    @property
    def is_enemy_jumping(self):
        return self.enemy_status == Status.JUMPING.value
    
    @property
    def is_enemy_standing(self):
        return self.enemy_status == Status.STANDING.value
    
    @property
    def is_enemy_stun(self):
        return self.enemy_status == Status.HIT_STUN.value
    
    def throw_sequence(self):
        """
        返回執行擲投的按鍵序列。
        擲投需要站在非常接近對手的位置並按下搖桿方向加上重拳。
        """
        sequence = []
        action = np.array([0] * 12)

        if self.is_facing_right:
            # 面向右邊時的擲投：→ + HP
            action[7] = 1  # 按下 → (RIGHT)
        else:
            # 面向左邊時的擲投：← + HP
            action[6] = 1  # 按下 ← (LEFT)

        action[Punch.LP.value] = 1  # 按下重拳 (Hard Punch)
        sequence.append(action.copy())

        # 調試輸出
        print(f"擲投動作: {action}")

        return sequence


    def hadouken_sequence(self, punch):
        """
        根據角色方向返回一系列的按鍵序列來執行波動拳，不保持按鍵。
        """
        sequence = []
        action = np.array([0] * 12)

        if self.is_facing_right:
            # 面向右邊的波動拳：↓ ↘ → + 拳
            action[5] = 1  # 按下 ↓ (DOWN)
            sequence.append(action.copy())  # 按一下 ↓

            action[7] = 1  # 同時按下 → (RIGHT) 和 ↓，形成 ↘
            sequence.append(action.copy())  # 按一下 ↘

            action[5] = 0  # 放開 ↓，只保持 →
            sequence.append(action.copy())  # 按一下 →

            action[7] = 0  # 放開 →
            action[punch.value] = 1
            sequence.append(action.copy())  # 按一下 拳
        else:
            # 面向左邊的波動拳：↓ ↙ ← + 拳
            action[5] = 1  # 按下 ↓ (DOWN)
            sequence.append(action.copy())  # 按一下 ↓

            action[6] = 1  # 同時按下 ← (LEFT) 和 ↓，形成 ↙
            sequence.append(action.copy())  # 按一下 ↙

            action[5] = 0  # 放開 ↓，只保持 ←
            sequence.append(action.copy())  # 按一下 ←

            action[6] = 0  # 放開 ←
            action[punch.value] = 1
            sequence.append(action.copy())  # 按一下 拳

        return sequence

    def shoryuken_sequence(self, punch):
        """
        返回執行昇龍拳（Shoryuken）的按鍵序列。
        """
        sequence = []
        action = np.array([0] * 12)

        if self.is_facing_right:
            # 面向右邊的昇龍拳：→ ↓ ↘ + 拳
            action[7] = 1  # 按下 → (RIGHT)
            sequence.append(action.copy())  # 按一下 →

            action[7] = 0  # 釋放 →
            action[5] = 1  # 按下 ↓ (DOWN)
            sequence.append(action.copy())  # 按一下 ↓

            action[7] = 1  # 同時按下 → (RIGHT) 和 ↓，形成 ↘
            sequence.append(action.copy())  # 按一下 ↘

            action[5] = 0  # 釋放 ↓
            action[punch.value] = 1
            sequence.append(action.copy())  # 按一下 拳
        else:
            # 面向左邊的昇龍拳：← ↓ ↙ + 拳
            action[6] = 1  # 按下 ← (LEFT)
            sequence.append(action.copy())  # 按一下 ←

            action[6] = 0  # 釋放 ←
            action[5] = 1  # 按下 ↓ (DOWN)
            sequence.append(action.copy())  # 按一下 ↓

            action[6] = 1  # 同時按下 ← (LEFT) 和 ↓，形成 ↙
            sequence.append(action.copy())  # 按一下 ↙

            action[5] = 0  # 釋放 ↓
            action[punch.value] = 1
            sequence.append(action.copy())  # 按一下 拳

        return sequence

    def defense_sequence(self):
        """
        返回執行防禦（Defense）的按鍵序列，根據角色面向方向。
        """
        sequence = []
        action = np.array([0] * 12)

        if self.is_facing_right:
            # 面向右邊時，按下 ← (LEFT) 進行防禦
            action[6] = 1  # 按下 ← (LEFT)
        else:
            # 面向左邊時，按下 → (RIGHT) 進行防禦
            action[7] = 1  # 按下 → (RIGHT)

        sequence.append(action.copy())
        return sequence

    def hurricane_kick_sequence(self, kick):
        """
        返回執行龙卷旋风脚（Hurricane Kick）的按鍵序列。
        """
        sequence = []
        action = np.array([0] * 12)

        if self.is_facing_right:
            # 面向右邊的龙卷旋风脚：↓ ↙ ← + 脚
            action[5] = 1  # 按下 ↓ (DOWN)
            sequence.append(action.copy())  # 按一下 ↓

            action[6] = 1  # 同時按下 ← (LEFT) 和 ↓，形成 ↙
            sequence.append(action.copy())  # 按一下 ↙

            action[5] = 0  # 釋放 ↓，只保持 ←
            sequence.append(action.copy())  # 按一下 ←

            action[6] = 0  # 釋放 ←
            action[kick.value] = 1
            sequence.append(action.copy())  # 按一下 脚
        else:
            # 面向左邊的龙卷旋风脚：↓ ↘ → + 脚
            action[5] = 1  # 按下 ↓ (DOWN)
            sequence.append(action.copy())  # 按一下 ↓

            action[7] = 1  # 同時按下 → (RIGHT) 和 ↓，形成 ↘
            sequence.append(action.copy())  # 按一下 ↘

            action[5] = 0  # 釋放 ↓，只保持 →
            sequence.append(action.copy())  # 按一下 →

            action[7] = 0  # 釋放 →
            action[kick.value] = 1
            sequence.append(action.copy())  # 按一下 脚

        return sequence

