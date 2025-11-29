import argparse

parser = argparse.ArgumentParser(description="parser for training")
parser.add_argument('filename', type=str, help='exp name')
args = parser.parse_args()

import gymnasium as gym
from stable_baselines3 import PPO
from stock_env import StockTradingEnv
import torch
import os

print(f"CUDA Available: {torch.cuda.is_available()}")

# 훈련 데이터 기간은 상승장/하락장이 섞여 있는게 좋음 (2015 ~ 2021)
env = StockTradingEnv(ticker='AAPL', start_date='2010-01-01', end_date='2021-12-31')

# PPO 모델 하이퍼파라미터 튜닝 (조금 더 보수적으로 학습)
model = PPO("MlpPolicy", env, verbose=1, device='cuda', 
            learning_rate=0.0001,  # 학습 속도를 늦춰서 더 신중하게
            batch_size=128,        # 배치 사이즈 증가
            ent_coef=0.01)         # 탐험 유지

print("Training Started (Smart Agent)...")
model.learn(total_timesteps=100000) # 10만 번 학습
print("Training Finished!")

os.makedirs("models", exist_ok=True)
model.save(f"models/ppo_stock_agent_{args.filename}")
print("Model Saved.")