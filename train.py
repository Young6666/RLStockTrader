import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stock_env import StockTradingEnv
import os

# 1. 환경 생성 (훈련용 기간 설정) [cite: 95]
env = StockTradingEnv(ticker='AAPL', start_date='2018-01-01', end_date='2021-12-31')

# 2. 모델 선택 및 설정 (PPO 예시)
# MultiInputPolicy는 Dict Observation Space를 처리하기 위해 필수입니다.
# device='cuda'를 명시하여 RTX 4060을 사용하게 합니다.
model = PPO("MultiInputPolicy", env, verbose=1, device='cuda') 

# 3. 학습 시작
print("Training Start...")
model.learn(total_timesteps=50000) # 스텝 수는 조정 가능
print("Training Finished.")

# 4. 모델 저장
os.makedirs("models", exist_ok=True)
model.save("models/ppo_aapl_trader")
