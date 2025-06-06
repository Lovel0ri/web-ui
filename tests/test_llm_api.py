import os
import pdb
import requests
import json
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

import sys

# 您的原有路徑處理代碼
sys.path.append(".")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 導入 utils，因為 create_message_content 需要它
from src.utils import utils  # 確保這裡的導入是正確的，因為 sys.path 已修改

@dataclass
class LLMConfig:
    provider: str
    model_name: str
    temperature: float = 0.8
    base_url: str = None
    api_key: str = None


def create_message_content(text, image_path=None):
    content = [{"type": "text", "text": text}]
    image_format = "png" if image_path and image_path.endswith(".png") else "jpeg"
    if image_path:
        # 這裡會使用從 project_root 構建的絕對路徑來調用 encode_image
        image_data = utils.encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}
        })
    return content


def get_env_value(key, provider):
    env_mappings = {
        "openai": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_ENDPOINT"},
        "azure_openai": {"api_key": "AZURE_OPENAI_API_KEY", "base_url": "AZURE_OPENAI_ENDPOINT"},
        "google": {"api_key": "GOOGLE_API_KEY"},
        "deepseek": {"api_key": "DEEPSEEK_API_KEY", "base_url": "DEEPSEEK_ENDPOINT"},
        "mistral": {"api_key": "MISTRAL_API_KEY", "base_url": "MISTRAL_ENDPOINT"},
        "alibaba": {"api_key": "ALIBABA_API_KEY", "base_url": "ALIBABA_ENDPOINT"},
        "moonshot": {"api_key": "MOONSHOT_API_KEY", "base_url": "MOONSHOT_ENDPOINT"},
        "ibm": {"api_key": "IBM_API_KEY", "base_url": "IBM_ENDPOINT"}
    }

    if provider in env_mappings and key in env_mappings[provider]:
        return os.getenv(env_mappings[provider][key], "")
    return ""


def test_llm(config, query, image_path=None, system_message=None):
    from src.utils import llm_provider  # utils 已經在前面導入

    # Special handling for Ollama-based models
    if config.provider == "ollama":
        if "deepseek-r1" in config.model_name:
            from src.utils.llm_provider import DeepSeekR1ChatOllama
            llm = DeepSeekR1ChatOllama(model=config.model_name)
        else:
            llm = ChatOllama(model=config.model_name)

        ai_msg = llm.invoke(query)
        print(ai_msg.content)
        if "deepseek-r1" in config.model_name:
            pdb.set_trace()
        return

    # 专门为 provider="google"（即 Gemini）插入的 HTTP 请求逻辑
    if config.provider == "google":
        # 读取 API_KEY（优先使用 config.api_key，否则从环境变量 GOOGLE_API_KEY 获取）
        api_key = config.api_key or get_env_value("api_key", "google")
        if not api_key:
            print("Error: Google API key 未配置，请检查环境变量 'GOOGLE_API_KEY' 或 config.api_key。")
            return

        # 构造 Gemini 请求的 URL（注意 use_v1beta 或 v1，根据文档选择）
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model_name}:generateContent?key={api_key}"

        # 构造请求 payload（将 query 作为 prompt）
        payload = {
            "contents": [
                {
                    "parts": [{"text": query}]
                }
            ],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": 256
            }
        }

        # 如果有 system_message，可以把它也发给 Gemini（可选）
        # 这里示例不做多轮，仅作单条 prompt
        try:
            response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            result = response.json()
            # 解析返回的文本
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            print(text)
        except Exception as e:
            print("请求失败或响应解析出错：", e)
            print("响应内容：", response.text if response is not None else "无响应")
        return

    # For other providers, use the standard configuration
    llm = llm_provider.get_llm_model(
        provider=config.provider,
        model_name=config.model_name,
        temperature=config.temperature,
        base_url=config.base_url or get_env_value("base_url", config.provider),
        api_key=config.api_key or get_env_value("api_key", config.provider)
    )

    # Prepare messages for non-Ollama models
    messages = []
    if system_message:
        messages.append(SystemMessage(content=create_message_content(system_message)))
    messages.append(HumanMessage(content=create_message_content(query, image_path)))
    ai_msg = llm.invoke(messages)

    # Handle different response types
    if hasattr(ai_msg, "reasoning_content"):
        print(ai_msg.reasoning_content)
    print(ai_msg.content)


# 以下每個測試函數都修改了圖片路徑
def test_openai_model():
    config = LLMConfig(provider="openai", model_name="gpt-4o")
    # 使用 os.path.join 結合 project_root 構建絕對路徑
    image_path = os.path.join(project_root, "assets", "examples", "test.png")
    test_llm(config, "Describe this image", image_path)


def test_google_model():
    # 這裡指定 model_name 為 Gemini 的版本
    config = LLMConfig(provider="google", model_name="gemini-2.0-flash-exp")
    # IMAGE 可選，如果只是文字交互，可不傳
    image_path = os.path.join(project_root, "assets", "examples", "test.png")
    test_llm(config, "Describe this image", image_path)


def test_azure_openai_model():
    config = LLMConfig(provider="azure_openai", model_name="gpt-4o")
    # 使用 os.path.join 結合 project_root 構建絕對路徑
    image_path = os.path.join(project_root, "assets", "examples", "test.png")
    test_llm(config, "Describe this image", image_path)


def test_deepseek_model():
    config = LLMConfig(provider="deepseek", model_name="deepseek-chat")
    test_llm(config, "Who are you?")


def test_deepseek_r1_model():
    config = LLMConfig(provider="deepseek", model_name="deepseek-reasoner")
    test_llm(config, "Which is greater, 9.11 or 9.8?", system_message="You are a helpful AI assistant.")


def test_ollama_model():
    config = LLMConfig(provider="ollama", model_name="qwen2.5:7b")
    test_llm(config, "Sing a ballad of LangChain.")


def test_deepseek_r1_ollama_model():
    config = LLMConfig(provider="ollama", model_name="deepseek-r1:14b")
    test_llm(config, "How many 'r's are in the word 'strawberry'?")


def test_mistral_model():
    config = LLMConfig(provider="mistral", model_name="pixtral-large-latest")
    # 使用 os.path.join 結合 project_root 構建絕對路徑
    image_path = os.path.join(project_root, "assets", "examples", "test.png")
    test_llm(config, "Describe this image", image_path)


def test_moonshot_model():
    config = LLMConfig(provider="moonshot", model_name="moonshot-v1-32k-vision-preview")
    # 使用 os.path.join 結合 project_root 構建絕對路徑
    image_path = os.path.join(project_root, "assets", "examples", "test.png")
    test_llm(config, "Describe this image", image_path)


def test_ibm_model():
    config = LLMConfig(provider="ibm", model_name="meta-llama/llama-4-maverick-17b-128e-instruct-fp8")
    # 使用 os.path.join 結合 project_root，构建绝对路径
    image_path = os.path.join(project_root, "assets", "examples", "test.png")
    test_llm(config, "Describe this image", image_path)


def test_qwen_model():
    config = LLMConfig(provider="alibaba", model_name="qwen-vl-max")
    # 使用 os.path.join 结合 project_root 构建绝对路径
    image_path = os.path.join(project_root, "assets", "examples", "test.png")
    test_llm(config, "How many 'r's are in the word 'strawberry'?", image_path)


if __name__ == "__main__":
    # test_openai_model()
    test_google_model()
    # test_azure_openai_model()
    # test_deepseek_model()
    # test_ollama_model()
    # test_deepseek_r1_model()
    # test_deepseek_r1_ollama_model()
    # test_mistral_model()
    # test_ibm_model()
    # test_qwen_model()
