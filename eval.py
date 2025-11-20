import gymnasium as gym
from stable_baselines3 import PPO
from stock_env import StockTradingEnv
import matplotlib.pyplot as plt

# 1. 테스트 환경 생성 (훈련에 쓰지 않은 기간 사용) [cite: 82]
env = StockTradingEnv(ticker='AAPL', start_date='2022-01-01', end_date='2023-01-01')

# 2. 저장된 모델 불러오기
model = PPO.load("models/ppo_aapl_trader")

obs, _ = env.reset()
done = False
net_worths = []

# 3. 시뮬레이션 실행
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    net_worths.append(info['net_worth'])

# 4. 결과 시각화 (과제 리포트용) [cite: 99]
plt.plot(net_worths)
plt.title("Portfolio Value Over Time (Test Period)")
plt.xlabel("Days")
plt.ylabel("Net Worth")
plt.savefig("evaluation_result.png")
plt.show()
