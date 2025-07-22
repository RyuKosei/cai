import threading
import time
from fastapi import FastAPI, Request, Body
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

MAX_TURNS = 5  # 统一最大步数

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
                "flag": current_task["result"],  # 现在flag字段为最后一轮总结内容
                "logfile": current_task["logfile"] or "",  # log文件绝对路径
                "status": "ok",
                "step": current_task["step"],
                "total": current_task["total"]
            }
        else:
            # 未完成时返回当前进度和logfile为空
            return {
                "step": current_task["step"],
                "total": current_task["total"],
                "logfile": ""
            }

@app.post("/stop")
def stop():
    with task_lock:
        current_task["stop_flag"] = True
    return {"status": "ok"}

@app.post("/mcp/load")
def mcp_load(data: dict = Body(...)):
    """
    加载MCP服务器，如 {"url": "http://localhost:9876/sse", "label": "burp"}
    """
    url = data.get("url")
    label = data.get("label")
    if not url or not label:
        return {"status": "error", "msg": "url和label必填"}
    try:
        from cai.mcp import MCPManager
        mcp_mgr = MCPManager.get()
        mcp_mgr.load(label, url)
        return {"status": "ok", "msg": f"MCP {label} loaded", "tools": list(mcp_mgr.get(label).tools.keys())}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.post("/mcp/add")
def mcp_add(data: dict = Body(...)):
    """
    将MCP工具加入agent，如 {"label": "burp", "agent": "Red Team Agent"}
    """
    label = data.get("label")
    agent_name = data.get("agent", "Red Team Agent")
    try:
        from cai.mcp import MCPManager
        mcp_mgr = MCPManager.get()
        mcp = mcp_mgr.get(label)
        agents = get_available_agents()
        agent = None
        for a in agents.values():
            if a.name == agent_name or a.name.lower() == agent_name.lower():
                agent = a
                break
        if not agent:
            return {"status": "error", "msg": f"Agent {agent_name} not found"}
        mcp.add_tools_to_agent(agent)
        return {"status": "ok", "msg": f"Tools from {label} added to {agent_name}", "tools": list(agent.tools.keys())}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

def print_step_info(step, total, msg, agent, model):
    # 简略展示，每步不超过10行，超出用省略号
    content = msg.get("content", "")
    lines = content.splitlines()
    short = "\n".join(lines[:10])
    if len(lines) > 10:
        short += "\n..."
    time_str = datetime.now().strftime("%H:%M:%S")
    # print(f"[INFO] [{step}/{total}] Agent: {msg.get('sender', agent.name)} | Model: {model} | Time: {time_str}\n{short}\n{'-'*40}")  # 去掉冗余输出
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

def auto_load_burp():
    try:
        from cai.repl.commands import mcp as mcp_cmd
        from cai.agents import get_available_agents
        # 加载burp
        ok = mcp_cmd.McpCommand().handle_load(["http://localhost:9876/sse", "burp"])
        if not ok:
            print("自动加载burp失败（连接失败）")
            return
        # 注入到Red Team Agent（手动方式）
        agents = get_available_agents()
        agent = None
        for a in agents.values():
            if a.name.lower() == "red team agent":
                agent = a
                break
        if not agent:
            print("burp已连接，但Red Team Agent未找到")
            return
        tools = mcp_cmd.mcp_tools_cache.get("burp", [])
        if not hasattr(agent, "functions") or not isinstance(agent.functions, list):
            agent.functions = []
        existing_names = {f.__name__ for f in agent.functions if callable(f)}
        added = 0
        for tool in tools:
            try:
                wrapper = mcp_cmd.create_tool_wrapper(
                    server_label="burp",
                    tool_name=tool.name,
                    tool_desc=tool.description,
                    schema=tool.inputSchema
                )
                if wrapper.__name__ in existing_names:
                    continue
                agent.functions.append(wrapper)
                existing_names.add(wrapper.__name__)
                added += 1
            except Exception as e:
                print(f"注入工具{tool.name}失败: {e}")
        if added:
            print(f"已自动加载burp并注入Red Team Agent（{added}个工具）")
        else:
            print("burp已连接，但无工具注入")
    except Exception as e:
        print(f"自动加载burp失败: {e}")

def print_server_info():
    from cai.agents import get_available_agents
    agents = list(get_available_agents().values())
    agent = None
    for a in agents:
        if a.name.lower() == "red team agent":
            agent = a
            break
    if not agent:
        agent = agents[0]
    print("\n================ Agent Server 启动信息 ================" )
    print(f"Agent模式: Red Team Agent")
    print(f"描述: Agent that mimic pentester/red teamer in a security assessment...")
    print(f"最大步数: {MAX_TURNS}")
    tools = getattr(agent, "tools", {})
    if hasattr(tools, 'keys'):
        print(f"本地Tools: {list(tools.keys())}")
    else:
        print(f"本地Tools: []")
    # 展示所有functions
    functions = getattr(agent, "functions", [])
    func_names = [f.__name__ for f in functions if callable(f)]
    print(f"所有可用工具（含MCP）: {func_names}")
    print("====================================================\n")

if __name__ == "__main__":
    auto_load_burp()
    print_server_info()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 