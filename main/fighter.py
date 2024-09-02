import numpy as np
from enum import Enum

class Punch(Enum):
    LP = 9
    MP = 10
    HP = 11
    
class Kick(Enum):
    LK = 1
    MK = 0
    HK = 8
    
class Moves(Enum):
    MODE = 2
    START = 3
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7

class Status(Enum):
    STANDING = 512
    CROUCHING = 514
    JUMPING = 516
    BLOCKING = 518
    NORMAL_ATTACK = 522 #Enemy speical attack
    DEFENCE = 520
    SPECIAL_ATTACK = 524 #522?
    HIT_STUN = 526
    THROWN = 532
    
class CharacterXPosition(Enum):
    LEFT_BOUNDARY = 55   # 最左邊的 X 坐標
    RIGHT_BOUNDARY = 458  # 最右邊的 X 坐標
       
class CharacterYPosition(Enum):
    GROUND = 192  # 地面高度
    MAX_HEIGHT = 67  # 跳躍的最高點高度

class Fighter:
    def __init__(self, info):
        if info != None:
            self.agent_x = info.get('agent_x', 0)
            self.enemy_x = info.get('enemy_x', 0)
            self.agent_y = info.get('agent_y', 0)
            self.enemy_y = info.get('enemy_y', 0)
            self.is_facing_right = self.agent_x < self.enemy_x
            self._distance = abs(self.enemy_x - self.agent_x)
            self.agent_status = info.get('agent_status', 0)
            self.enemy_status = info.get('enemy_status', 0)
        else:
            self.agent_x = 0
            self.enemy_x = 0
            self.agent_y = 0
            self.enemy_y = 0
            self.is_facing_right = True
            self._distance = 145
            self.agent_status = Status.STANDING.value
            self.enemy_status = Status.STANDING.value
            
    def get_best_move(self):
        sequence = None
        
        print(f"Distance: {self.distance}")
        
        if self.is_enemy_jumping and (self.distance >= 90 and self.distance <= 105):
            sequence = self.shoryuken_sequence(Punch.HP)
            #sequence = self.diagonal_jump_kick_sequence()
        elif self.distance >= 150 and not self.is_enemy_stun:
            sequence = self.hadouken_sequence(Punch.MP)
        elif self.is_enemy_standing and self.distance < 40:
            sequence = self.attack_sequence(Punch.LP)
        else:
            sequence = self.defense_sequence()
            
        return sequence
            
    @property
    def distance(self):
        return self._distance
    
    @property
    def is_standing(self):
        return self.agent_status == Status.STANDING.value
    
    @property   
    def is_enemy_jumping(self):
        print(f"agent_status:{self.agent_status} enemy_status:{self.enemy_status} y:{self.enemy_y} x:{self.enemy_x}")
        return self.enemy_y > 105 and self.enemy_y < 130\
            or self.enemy_status == Status.JUMPING.value
    
    @property
    def is_enemy_standing(self):
        return self.enemy_y == CharacterYPosition.GROUND.value
    
    @property
    def is_enemy_stun(self):
        return self.enemy_status == Status.HIT_STUN.value
    
    def diagonal_jump_kick_sequence(self):
        """
        返回執行對角線跳踢（Diagonal Jumping Kick）的按鍵序列。
        """
        sequence = []
        action = np.array([0] * 12)

        # 1. 向上並向右/左跳躍
        action[4] = 1  # 按下 ↑ (UP) 來跳躍
        if self.is_facing_right:
            action[7] = 1  # 按下 → (Right) 來向右跳
        else:
            action[6] = 1  # 按下 ← (Left) 來向左跳
        sequence.append(action.copy())  # 添加到序列: 按一下 ↑ 和 → 或 ←
        
        # 2. 在空中按下踢擊鍵
        action[4] = 0  # 釋放 ↑
        if self.is_facing_right:
            action[7] = 1  # 保持 → (Right)
        else:
            action[6] = 1  # 保持 ← (Left)
        action[Kick.HK.value] = 1  # 按下 High Kick (HK) 按鍵
        sequence.append(action.copy())  # 添加到序列: 按一下 HK

        # 3. 結束跳踢
        action[Kick.HK.value] = 0  # 釋放 HK
        if self.is_facing_right:
            action[7] = 0  # 釋放 →
        else:
            action[6] = 0  # 釋放 ←
        sequence.append(action.copy())  # 添加到序列: 釋放所有按鍵

        return sequence

    def jump_kick_sequence(self):
        """
        返回執行跳踢（High Kick, HK）的按鍵序列。
        """
        sequence = []
        action = np.array([0] * 12)

        # 1. 跳躍
        action[4] = 1  # 按下 ↑ (UP) 來跳躍
        sequence.append(action.copy())  # 添加到序列: 按一下 ↑

        # 2. 在空中按下踢擊鍵
        action[4] = 0  # 釋放 ↑，模擬跳躍開始
        action[Kick.MK.value] = 1
        sequence.append(action.copy())  # 添加到序列: 按一下 HK

        # 3. 結束跳踢
        action[Kick.HK.value] = 0  # 釋放 HK
        sequence.append(action.copy())  # 添加到序列: 釋放按鍵
        
        return sequence


    def attack_sequence(self, punch):
        sequence = []
        action = np.array([0] * 12)
                          
        if self.is_facing_right:
            action[7] = 1
            
        else:
            action[6] = 1
        sequence.append(action.copy())
        
        action[punch.value] = 1
        sequence.append(action.copy())
        
        return sequence

    def hadouken_sequence(self, punch):
        """
        返回執行波動拳（Hadouken）的按鍵序列。
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

