import sys
import io

# Windows 终端默认 gbk，强制用 utf-8 避免 emoji/中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from agent.loop import AgentLoop


if __name__ == "__main__":
    AgentLoop().run()
