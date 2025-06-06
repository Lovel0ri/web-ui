import requests
import json

API_KEY = "AIzaSyAyaauMvkYd5GHscSrKGC42IycPhDqcvQ0"
MODEL_NAME = "models/gemini-1.5-flash-001"  # 可用模型之一
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json",
    # "Authorization": f"Bearer {API_KEY}"
}

def generate_text(prompt):
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 256
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except Exception as e:
        print("请求失败或响应解析出错：", e)
        print("响应内容：", response.text)
        return None

if __name__ == "__main__":
    prompt = "请用简体中文介绍一下台湾夜市的特色。"
    result = generate_text(prompt)
    if result:
        print("Gemini 返回的文本如下：\n")
        print(result)
