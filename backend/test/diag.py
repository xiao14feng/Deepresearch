"""逐步诊断 MCP 连接问题"""
import sys
print(f"[PYTHON] 父进程 Python 路径: {sys.executable}", flush=True)

print("[1] 脚本启动", flush=True)

# 第一步：先测试子进程本身能不能跑
import subprocess
print("[2] 测试子进程启动...", flush=True)
try:
    proc = subprocess.Popen(
        [sys.executable, "MCP_LangGraph.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("[3] 子进程已启动，等待0.5秒...", flush=True)
    import time
    time.sleep(0.5)

    # 检查进程状态
    ret = proc.poll()
    print(f"[4] 进程状态: ret={ret}", flush=True)

    if ret is not None:
        # 进程已退出，读取输出
        out, err = proc.communicate(timeout=2)
        print(f"[5] 子进程已退出，退出码={ret}", flush=True)
        print(f"[5] stdout={out.decode('utf-8', errors='replace')}", flush=True)
        print(f"[5] stderr={err.decode('utf-8', errors='replace')}", flush=True)
    else:
        print("[5] 子进程仍在运行，写入 stdin 测试...", flush=True)
        # MCP 服务器通过 stdin/stdout 通信，我们需要发送 JSON 消息
        import json
        # 发送 initialize 请求
        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }) + "\n"
        print(f"[6] 发送: {init_msg.strip()}", flush=True)
        out, err = proc.communicate(input=init_msg, timeout=5)
        print(f"[7] 收到: stdout={out.decode('utf-8', errors='replace')}", flush=True)
        print(f"[7] 收到: stderr={err.decode('utf-8', errors='replace')}", flush=True)
        proc.terminate()
except Exception as e:
    print(f"[X] 错误: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("[8] 诊断完成", flush=True)
