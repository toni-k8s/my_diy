import os 
os.makedirs('logs', exist_ok=True)


os.makedirs('checkpoints', exist_ok=True)
os.system('cp config.py checkpoints/')