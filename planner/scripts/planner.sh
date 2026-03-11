#!/bin/bash
# filepath: /root/numa/PctPlanner/planner/scripts/run_planner.sh

# 设置 Python 脚本的路径
PYTHON_SCRIPT="/root/numa/PctPlanner/planner/scripts/plan_system.py"

# 获取脚本所在目录，确保相对路径 import 正常工作
SCRIPT_DIR=$(dirname "$PYTHON_SCRIPT")
cd "$SCRIPT_DIR"

echo "Starting monitored PCT Planner node..."

# 使用无限循环实现自动重启
while true; do
    echo "[$(date)] Launching $PYTHON_SCRIPT..."
    
    # 执行 Python 程序
    python3 "$PYTHON_SCRIPT"
    
    # 获取退出状态码
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Program exited normally. Restarting in 2 seconds..."
    else
        echo "[$(date)] Program crashed with exit code $EXIT_CODE. Restarting in 2 seconds..."
    fi
    
    # 稍微等待一下，防止遇到严重错误时无限循环过快消耗 CPU
    sleep 2
done