# 設置遊戲和狀態
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
from stable_baselines3.common.monitor import Monitor
import retro
import pyaudio
import time
import matplotlib.pyplot as plt

game = "StreetFighterIISpecialChampionEdition-Genesis"
state = "Champion.Level12.RyuVsBison"
state = "VsKen"
#state = "VsChunli"
#state = "VsRyu"

def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )

        env = StreetFighterCustomWrapper(env, rendering=True)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

# 創建環境初始化函數
env_fn = make_env(game, state)

# 初始化環境
env = env_fn()

# 測試環境
observation = env.reset()
done = False

def get_time():
    return time.time()

frame_time = 1.0 / 60.0
p = pyaudio.PyAudio()
audio_rate = env.em.get_audio_rate()
stream = p.open(format = pyaudio.paInt16,
                channels=2,
                rate=int(audio_rate),
                output=True)

last_time = get_time()

# 無窮迴圈直到關閉視窗
try:
    while True:
        action = env.action_space.sample()  # 使用隨機動作
        observation, reward, done, info = env.step(action)        

        if done:
            observation = env.reset()  # 如果遊戲結束，重置環境重新開始            

        current_time = get_time()
        time_diff = current_time - last_time
        if time_diff < frame_time:
           time.sleep(frame_time - time_diff)
        last_time = current_time
        
        env.render()  # 渲染遊戲畫面
        
        # play audio sample for this frame
        #sound = bytes(env.em.get_audio())
        #stream.write(frames=sound)
except KeyboardInterrupt:
    # 捕捉 Ctrl+C 信號以平滑關閉
    print("遊戲已停止。")
finally:
    # close audio stream
    stream.close()
    p.terminate()
    env.close()  # 確保在關閉時釋放資源
