import threading
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import os
import sys
import io
from datetime import datetime

# 确保cai包可import
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from cai.core import CAI
from cai.types import Agent
from cai.agents import get_available_agents
from cai.util import load_cai_config

app = FastAPI()

MAX_TURNS = 100  # 统一最大步数

# 全局任务状态
current_task = {
    "flag": None,
    "log": "",
    "result": None,
    "status": "pending",
    "thread": None,
    "stop_flag": False,
    "model": None,
    "prompt": None,
    "step": 0,
    "total": MAX_TURNS,
    "logfile": None
}

# 线程安全锁
task_lock = threading.Lock()

def get_red_team_agent(model_name: str) -> Agent:
    agents = list(get_available_agents().values())
    if len(agents) >= 14:
        agent = agents[13]
    else:
        agent = agents[0]
    agent.model = model_name
    return agent

class ModelRequest(BaseModel):
    model: str

class ChatRequest(BaseModel):
    prompt: str

@app.post("/model")
def set_model(data: ModelRequest):
    with task_lock:
        current_task["model"] = data.model
    return {"status": "ok"}

@app.post("/chat")
def chat(data: ChatRequest):
    with task_lock:
        if current_task["thread"] and current_task["thread"].is_alive():
            return {"status": "busy"}
        current_task["prompt"] = data.prompt
        current_task["flag"] = None
        current_task["log"] = ""
        current_task["result"] = None
        current_task["status"] = "pending"
        current_task["stop_flag"] = False
        current_task["step"] = 0
        current_task["total"] = MAX_TURNS
        current_task["logfile"] = None
        t = threading.Thread(target=run_cai_task)
        current_task["thread"] = t
        t.start()
    return {"status": "ok"}

@app.get("/result")
def result():
    with task_lock:
        if current_task["result"] is not None:
            return {
                "flag": current_task["flag"],
                "log": current_task["log"],
                "result": current_task["result"],
                "status": "ok",
                "step": current_task["step"],
                "total": current_task["total"]
            }
        else:
            # 未完成时返回当前进度和log
            return {
                "step": current_task["step"],
                "total": current_task["total"],
                "log": current_task["log"]
            }

@app.post("/stop")
def stop():
    with task_lock:
        current_task["stop_flag"] = True
    return {"status": "ok"}

def print_step_info(step, total, msg, agent, model):
    # 简略展示，每步不超过10行，超出用省略号
    content = msg.get("content", "")
    lines = content.splitlines()
    short = "\n".join(lines[:10])
    if len(lines) > 10:
        short += "\n..."
    time_str = datetime.now().strftime("%H:%M:%S")
    print(f"[INFO] [{step}/{total}] Agent: {msg.get('sender', agent.name)} | Model: {model} | Time: {time_str}\n{short}\n{'-'*40}")
    return f"[INFO] [{step}/{total}] Agent: {msg.get('sender', agent.name)} | Model: {model} | Time: {time_str}\n{short}\n{'-'*40}\n"

def compress_box_content(box_str, max_lines=10):
    """
    压缩框内容为最多max_lines行，超出部分用省略号。保留上下边界。
    """
    lines = box_str.splitlines()
    if len(lines) <= 2:
        return box_str  # 只有边界，无内容
    header = lines[0]
    footer = lines[-1]
    content = lines[1:-1]
    if len(content) > max_lines:
        content = content[:max_lines] + ['...']
    return '\n'.join([header] + content + [footer])


def run_cai_task():
    try:
        with task_lock:
            model = current_task["model"]
            prompt = current_task["prompt"]
        if not model or not prompt:
            with task_lock:
                current_task["status"] = "error"
            return
        agent = get_red_team_agent(model)
        cai = CAI()
        # 日志文件路径
        logfile = None
        if hasattr(cai, "rec_training_data") and cai.rec_training_data and hasattr(cai.rec_training_data, "filename"):
            logfile = os.path.abspath(cai.rec_training_data.filename)
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            print(f"日志文件: {logfile}")
            with task_lock:
                current_task["logfile"] = logfile
        history = [{"role": "user", "content": prompt}]
        context_variables = {}
        step = 0
        total = MAX_TURNS
        log_acc = ""
        agent_obj = agent
        for i in range(MAX_TURNS):
            if current_task["stop_flag"]:
                break
            step = i + 1
            # 单步推理
            result = cai.process_interaction(
                agent_obj, history, context_variables, model_override=None,
                stream=False, debug=2, execute_tools=True, n_turn=step
            )
            # 获取最新一轮消息
            msg = None
            if history:
                msg = history[-1]
            else:
                continue
            # 简略输出
            info_str = print_step_info(step, total, msg, agent_obj, model)
            log_acc += info_str
            with task_lock:
                current_task["step"] = step
                current_task["total"] = total
                current_task["log"] = log_acc
            # flag抽取
            import re
            flag = None
            if msg and isinstance(msg, dict):
                m = re.search(r"flag\{[A-Za-z0-9_\-]+\}", str(msg.get("content", "")))
                if m:
                    flag = m.group(0)
            with task_lock:
                current_task["flag"] = flag
            # 终止条件：flag找到或agent返回None
            if flag or result is None:
                break
        # 最后一轮回复
        last_msg = history[-1] if history else None
        with task_lock:
            current_task["result"] = last_msg.get("content", "") if last_msg else None
            current_task["status"] = "ok"
            current_task["step"] = step
            current_task["total"] = total
    except Exception as e:
        with task_lock:
            current_task["status"] = f"error: {e}"
            current_task["log"] += f"\n[Exception]: {e}\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 