from __future__ import annotations

import json

import re

import urllib.request

from typing import Any

from .engine import MDKJ_KEY, RELAY_MODEL, _read_api_setting

def _call(prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    import time
    import urllib.error
    from http.client import IncompleteRead

    key = _read_api_setting(MDKJ_KEY)
    if not key:
        raise ValueError('缺少中转密钥配置')
    body = dict(model=RELAY_MODEL, system=prompt, max_tokens=20000, thinking={'type':'disabled'}, temperature=0, messages=[{'role':'user','content':json.dumps(payload, ensure_ascii=False)}])
    raw_bytes = json.dumps(body, ensure_ascii=False).encode('utf-8')
    last_err: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request('https://mdkj.lol/v1/messages', raw_bytes, {'Content-Type':'application/json','anthropic-version':'2023-06-01','User-Agent':'Mozilla/5.0'})
        request.add_unredirected_header('x-api-key', key)
        try:
            with urllib.request.urlopen(request, timeout=210) as response:
                data = json.load(response)
            if data['model'] != RELAY_MODEL or data['stop_reason'] != 'end_turn':
                raise ValueError(f"模型 {data['model']}，结束状态 {data['stop_reason']}")
            text = ''.join(b['text'] for b in data['content'] if b['type'] == 'text').strip()
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I).strip()
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            return json.loads(cleaned)
        except urllib.error.HTTPError as err:
            if err.code in {429, 500, 502, 503, 504, 524} and attempt < 3:
                last_err = err
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as err:
            if attempt < 3:
                last_err = err
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError(str(last_err))

def call_preparation_responses(model: str, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    from .engine import _read_relay_response

    key = _read_api_setting("SHENLUN_MDKJ_RESPONSES_API_KEY")
    if not key:
        raise RuntimeError("缺少 SHENLUN_MDKJ_RESPONSES_API_KEY")
    body = {
        "model": model,
        "instructions": prompt,
        "input": json.dumps(payload, ensure_ascii=False),
        "max_output_tokens": 16000,
        "store": False,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "x-openai-actor-authorization": "local-image-extension",
    }
    request = urllib.request.Request(
        "https://mdkj.lol/v1/responses", json.dumps(body, ensure_ascii=False).encode("utf-8"), headers
    )
    request.add_unredirected_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(request, timeout=180) as response:
        data = _read_relay_response(response)
    if data.get("model") != model or data.get("status") != "completed":
        raise ValueError(f"模型 {data.get('model')}，结束状态 {data.get('status')}")
    text = "".join(
        part["text"] for item in data["output"] if item["type"] == "message"
        for part in item["content"] if part["type"] == "output_text"
    ).strip()
    return json.loads(text.removeprefix("```json").removesuffix("```").strip())
