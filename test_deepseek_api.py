
import json
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

# 读取项目根目录的 .env
cfg = dotenv_values(Path(__file__).with_name(".env"))

api_key = cfg.get("DEEPSEEK_API_KEY")
base_url = cfg.get("DEEPSEEK_BASE_URL")
model = cfg.get("DEEPSEEK_MODEL")

assert api_key and api_key.strip(), "API Key 未配置"
assert base_url == "https://api.deepseek.com", "API 地址不正确"
assert model == "deepseek-flash", "模型名称不正确"

try:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=0
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an API test assistant. Return JSON only."
            },
            {
                "role": "user",
                "content": 'Return {"ok": true, "message": "connected"}.'
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=128,
        extra_body={
            "thinking": {"type": "disabled"}
        }
    )

    content = response.choices[0].message.content
    result = json.loads(content)

    assert result.get("ok") is True

    print("\n========== API TEST PASSED ==========")
    print("Status:", "SUCCESS")
    print("Model:", response.model)
    print("Thinking:", "DISABLED")
    print("Response:", content)

    if response.usage:
        print("Prompt tokens:", response.usage.prompt_tokens)
        print("Completion tokens:", response.usage.completion_tokens)
        print("Total tokens:", response.usage.total_tokens)

    print("=====================================")

except Exception as e:
    print("\n========== API TEST FAILED ==========")
    print("Error type:", type(e).__name__)
    print("HTTP status:", getattr(e, "status_code", None))
    print("Error:", str(e)[:500])
    print("=====================================")
