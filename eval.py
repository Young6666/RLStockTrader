import gymnasium as gym
from stable_baselines3 import PPO
from stock_env import StockTradingEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_model(model, ticker, start_date, end_date):
    print(f"\nTesting on {ticker} ({start_date} ~ {end_date})...")
    
    env = StockTradingEnv(ticker=ticker, start_date=start_date, end_date=end_date)
    obs, _ = env.reset()
    done = False
    net_worth_history = []
    actions_taken = [] # 행동 기록용
    
    while not done:
        # deterministic=False로 설정하여 약간의 탐험을 허용해볼 수도 있음 (테스트용)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        net_worth_history.append(info['net_worth'])
        actions_taken.append(action)

    final_value = net_worth_history[-1]
    
    # 행동 통계 출력 (디버깅 핵심)
    buy_count = actions_taken.count(1)
    sell_count = actions_taken.count(2)
    hold_count = actions_taken.count(0)
    print(f"-> Actions: Buy({buy_count}), Sell({sell_count}), Hold({hold_count})")
    print(f"-> Final Portfolio Value: ${final_value:,.2f}")
    
    return net_worth_history

# 모델 로드 및 설정
model_path = "models/ppo_stock_agent_smart"
if not os.path.exists(model_path + ".zip"):
    print("Error: 모델 파일이 없습니다.")
    exit()

model = PPO.load(model_path)

test_start = "2022-01-01"
test_end = "2023-01-01"
all_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
results = {}

for ticker in all_tickers:
    try:
        history = evaluate_model(model, ticker, test_start, test_end)
        results[ticker] = history
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for ticker, history in results.items():
    plt.plot(history, label=ticker)
plt.title("Portfolio Value Over Time (Test Period)")
plt.xlabel("Days")
plt.ylabel("Net Worth ($)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for ticker, history in results.items():
    returns = pd.Series(history).pct_change().dropna()
    mean_return = returns.mean() * 100
    std_dev = returns.std() * 100
    plt.scatter(std_dev, mean_return, s=100, label=ticker)
    plt.text(std_dev, mean_return, f" {ticker}", fontsize=9)

plt.title("Risk vs Return")
plt.xlabel("Risk (Volatility %)")
plt.ylabel("Return (%)")
plt.axhline(0, color='black', linestyle='--')
plt.grid(True)

plt.tight_layout()
plt.savefig("evaluation_result.png")
print("\nEvaluation graph saved.")