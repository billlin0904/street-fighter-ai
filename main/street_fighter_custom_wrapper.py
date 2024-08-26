# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import time
import collections

import gym
import numpy as np

from fighter import Punch, \
    Kick, \
    Fighter

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
        
        self.prev_info = None
    
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        self.prev_info = None
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def best_move(self, info):
        fighter = Fighter(info)
        sequence = None
        
        if fighter.is_enemy_jumping and fighter.distance > 90:
            sequence = fighter.shoryuken_sequence(Punch.HP)
        elif fighter.is_enemy_standing and fighter.distance > 145:
            sequence = fighter.hadouken_sequence(Punch.HP)
        else:
            if not fighter.is_enemy_jumping and fighter.is_enemy_stun:
                sequence = fighter.hadouken_sequence(Punch.HP)
            else:
                if fighter.distance < 45:
                    sequence = fighter.attack(Punch.HP)
                else:
                    sequence = fighter.defense_sequence()
        return sequence
        
    def step(self, action):
        custom_done = False
        custom_reward = 0

        sequence = self.best_move(self.prev_info)
                  
        for move in sequence:
            for _ in range(self.num_step_frames):  # 保持每個按鍵組合一定的幀數
                obs, _reward, _done, info = self.env.step(move)
                self.frame_stack.append(obs[::2, ::2, :])
                if self.rendering:
                    self.env.render()
                    time.sleep(0.01)
                
                # # 更新信息，並再次調用 best_move 以獲取最新的動作序列
                # self.prev_info = info  # 更新 prev_info 為最新的 info
                # new_sequence = self.best_move(info)
                
                # # 使用 np.any() 或 np.all() 來比較數組
                # if not np.array_equal(new_sequence, sequence) and np.any(new_sequence):
                #     # 如果新的動作序列與當前序列不同，更新為新序列
                #     sequence = new_sequence
                #     break  # 退出內部循環以重新評估序列
                    
        # obs, _reward, _done, info = self.env.step(action)
        # self.frame_stack.append(obs[::2, ::2, :])
        # if self.rendering:
        #     self.env.render()
        #     time.sleep(0.01)
        # for _ in range(self.num_step_frames - 1):
        #     obs, _reward, _done, info = self.env.step(action)
        #     self.frame_stack.append(obs[::2, ::2, :])
        #     if self.rendering:
        #         self.env.render()
        #         time.sleep(0.01)                    
                
        self.prev_info = info

        # if np.array_equal(action, special_move_action):
        #     for _ in range(self.num_step_frames):
        #         obs, _reward, _done, info = self.env.step(special_move_action)
        #         self.frame_stack.append(obs[::2, ::2, :])
        #         if self.rendering:
        #             self.env.render()
        #             time.sleep(0.01)
        # else:
        #     obs, _reward, _done, info = self.env.step(action)
        #     self.frame_stack.append(obs[::2, ::2, :])
        #     if self.rendering:
        #         self.env.render()
        #         time.sleep(0.01)
        #     for _ in range(self.num_step_frames - 1):
        #         obs, _reward, _done, info = self.env.step(action)
        #         self.frame_stack.append(obs[::2, ::2, :])
        #         if self.rendering:
        #             self.env.render()
        #             time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        self.total_timesteps += self.num_step_frames

        if curr_player_health < 0:
            ustom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))
            custom_done = True
        elif curr_oppont_health < 0:
            custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True
        else:
            health_difference = self.prev_oppont_health - curr_oppont_health
            custom_reward = self.reward_coeff * health_difference - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        if not self.reset_round:
            custom_done = False

        return self._stack_observation(), 0.001 * custom_reward, custom_done, info
    