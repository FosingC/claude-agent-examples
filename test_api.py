"""调试用：测试千问 API 连通性"""
import os
import sys
import io
import json
import urllib.request
from dotenv import load_dotenv

# 强制 utf-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

api_key = os.environ.get("LLM_API_KEY", "").strip()
base_url = os.environ.get("LLM_BASE_URL", "").strip()

print(f"API KEY (前10位): {api_key[:10]}...")
print(f"BASE URL: {base_url}")

url = base_url.rstrip("/") + "/chat/completions"
print(f"\n请求 URL: {url}")

payload = {
    "model": "qwen-turbo",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100,
}

data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    url,
    data=data,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        print(f"状态码: {resp.status}")
        obj = json.loads(body)
        content = obj["choices"][0]["message"].get("content", "")
        print(f"AI回复: {content}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    body = e.read().decode("utf-8")
    print(f"响应体: {body}")
except Exception as e:
    print(f"其他错误: {e}")
