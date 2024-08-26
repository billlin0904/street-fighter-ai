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

from move_list import make_hadouken_sequence,\
make_shoryuken_sequence,\
make_defense_sequence, \
make_hurricane_kick_sequence, \
make_jump_kick_punch_shoryuken_sequence

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

    def step(self, action):
        custom_done = False
        custom_reward = 0        

        is_facing_right = True
        is_jump = False
        is_long_distance = True
        is_move_end = True
        distance = 100
        
        if self.prev_info != None:
            # 從 info 字典中獲取角色和對手的 x 坐標
            agent_x = self.prev_info.get('agent_x', 0)
            enemy_x = self.prev_info.get('enemy_x', 0)
            # 判斷角色是否面向右邊
            is_facing_right = agent_x < enemy_x  
            
            agent_y = self.prev_info.get('agent_y', 0)                
            enemy_y = self.prev_info.get('enemy_y', 0)
            distance = abs(enemy_x - agent_x)
            is_long_distance = distance > 145
            print(f"distance: {distance}")
            is_move_end = agent_y == 192
            is_jump = enemy_y != 192
            
        if not is_move_end:
            sequence = make_defense_sequence(is_facing_right)
        else:
            if is_jump:
                sequence = make_shoryuken_sequence(is_facing_right)
            elif is_long_distance:
                sequence = make_hadouken_sequence(is_facing_right)
            else:
                if distance <= 97:
                    sequence = make_jump_kick_punch_shoryuken_sequence(is_facing_right)
                else:
                    sequence = make_defense_sequence(is_facing_right)
                # sequence = make_hurricane_kick_sequence(is_facing_right)
                # sequence = make_defense_sequence(is_facing_right)
           
        for move in sequence:
            for _ in range(self.num_step_frames):  # 保持每個按鍵組合一定的幀數
                obs, _reward, _done, info = self.env.step(move)
                self.frame_stack.append(obs[::2, ::2, :])
                if self.rendering:
                    self.env.render()
                    time.sleep(0.01)
                    
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
    