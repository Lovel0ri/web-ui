from openai import OpenAI
import pdb
from langchain_openai import ChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
import os
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast, List,
)
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_aws import ChatBedrock
from pydantic import SecretStr, Field

# Import requests and json for直接调用 Gemini API
import requests
import json

from src.utils import config
from src.utils import utils  # 假设 utils 中有 encode_image 等函数


class DeepSeekR1ChatOpenAI(ChatOpenAI):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key")
        )

    async def ainvoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)


class DeepSeekR1ChatOllama(ChatOllama):

    async def ainvoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = await super().ainvoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = super().invoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)


class CustomChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    Custom Gemini Chat 模型，用于直接调用 Google Generative API（支持 text + image）。
    修正了 URL 里 model 前缀可能出现的重复 “models/models/...” 问题。
    """

    api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GOOGLE_API_KEY", "")))

    def __init__(self, *, model: str = "gemini-2.0-flash-exp", temperature: float = 0.0, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """
        :param model: 要调用的 Gemini 模型名称（可以是 "gemini-2.0-flash" 或者 "models/gemini-2.0-flash"）。
        :param temperature: 调用时的 temperature。
        :param api_key: 可选，如果传入则覆盖环境变量中读取到的 api_key。
        """
        if api_key is not None:
            kwargs["api_key"] = SecretStr(api_key)
        super().__init__(model=model, temperature=temperature, **kwargs)
        # ChatGoogleGenerativeAI 父类会把 model 存到 self.model，把 temperature 存到 self.temperature，把 api_key 存到 self.api_key

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步调用 Gemini API 并返回 ChatResult。"""

        # 1. 先把 messages 转成 Gemini 要求的 payload 格式
        gemini_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts = []
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append({"text": item["text"]})
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image/"):
                                mime_type = image_url.split(";")[0].split(":")[1]
                                base64_data = image_url.split(",")[1]
                                parts.append({"inline_data": {"mime_type": mime_type, "data": base64_data}})
                gemini_messages.append({"role": "user", "parts": parts})
            elif isinstance(msg, AIMessage):
                gemini_messages.append({"role": "model", "parts": [{"text": msg.content}]})
            elif isinstance(msg, SystemMessage):
                # 如果已有用户消息就插到第一条的 parts 里，否则当成新的 user 消息
                if gemini_messages and gemini_messages[0]["role"] == "user":
                    gemini_messages[0]["parts"].insert(0, {"text": msg.content + "\n"})
                else:
                    gemini_messages.insert(0, {"role": "user", "parts": [{"text": msg.content}]})

        # 2. 修正 model_name_for_url：如果 self.model 已经以 "models/" 开头，就直接用；否则手动拼 "models/{self.model}"
        if self.model.startswith("models/"):
            model_name_for_url = self.model
        else:
            model_name_for_url = f"models/{self.model}"

        # 3. 构造最终的 URL（确认只有一个 "models/" 前缀）
        api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/{model_name_for_url}:generateContent"
            f"?key={self.api_key.get_secret_value()}"
        )

        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
            },
        }

        try:
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            text_content = result["candidates"][0]["content"]["parts"][0]["text"]
            chat_generation = ChatGeneration(message=AIMessage(content=text_content))
            return ChatResult(generations=[chat_generation])

        except Exception as e:
            raise ValueError(
                f"Error calling Gemini API: {e}\n"
                f"URL: {api_url}\n"
                f"Response: {response.text if 'response' in locals() else 'No response'}"
            )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步调用，内部直接调用同步方法。"""
        return self._generate(messages, stop, run_manager, **kwargs)



def get_llm_model(provider: str, **kwargs):
    """
    根据 provider 名称返回对应的 LLM 实例。
    :param provider: LLM 提供商，比如 "google"、"openai"、"anthropic" 等
    :param kwargs: 包含 model_name、temperature、api_key、base_url 等
    :return: 对应的 Chat 模型实例
    """
    api_key = None  # 用来存储最终确认的 api_key

    if provider not in ["ollama", "bedrock"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            provider_display = config.PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
            error_msg = f"💥 {provider_display} API key not found! 🔑 请设置环境变量 `{env_var}` 或者在 UI 中填写它。"
            raise ValueError(error_msg)

    if provider == "anthropic":
        base_url = kwargs.get("base_url", "https://api.anthropic.com")
        return ChatAnthropic(
            model=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == 'mistral':
        base_url = kwargs.get("base_url", os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1"))
        return ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "openai":
        base_url = kwargs.get("base_url", os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"))
        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "grok":
        base_url = kwargs.get("base_url", os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1"))
        return ChatOpenAI(
            model=kwargs.get("model_name", "grok-3"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "deepseek":
        base_url = kwargs.get("base_url", os.getenv("DEEPSEEK_ENDPOINT", ""))
        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )

    elif provider == "google":
        # 注意这里传入的是 model（而不是 model_name），因为父类 ChatGoogleGenerativeAI 构造函数接受的是 model 参数。
        return CustomChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            api_key=api_key,
        )

    elif provider == "ollama":
        base_url = kwargs.get("base_url", os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"))
        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )

    elif provider == "azure_openai":
        base_url = kwargs.get("base_url", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        api_version = kwargs.get("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"))
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
        )

    elif provider == "alibaba":
        base_url = kwargs.get("base_url", os.getenv("ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        return ChatOpenAI(
            model=kwargs.get("model_name", "qwen-plus"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "ibm":
        parameters = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("num_ctx", 32000)
        }
        base_url = kwargs.get("base_url", os.getenv("IBM_ENDPOINT", "https://us-south.ml.cloud.ibm.com"))
        return ChatWatsonx(
            model_id=kwargs.get("model_name", "ibm/granite-vision-3.1-2b-preview"),
            url=base_url,
            project_id=os.getenv("IBM_PROJECT_ID"),
            apikey=os.getenv("IBM_API_KEY"),
            params=parameters
        )

    elif provider == "moonshot":
        return ChatOpenAI(
            model=kwargs.get("model_name", "moonshot-v1-32k-vision-preview"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=os.getenv("MOONSHOT_ENDPOINT"),
            api_key=api_key,
        )

    elif provider == "unbound":
        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o-mini"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=os.getenv("UNBOUND_ENDPOINT", "https://api.getunbound.ai"),
            api_key=api_key,
        )

    elif provider == "siliconflow":
        base_url = kwargs.get("base_url", os.getenv("SiliconFLOW_ENDPOINT", ""))
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model_name=kwargs.get("model_name", "Qwen/QwQ-32B"),
            temperature=kwargs.get("temperature", 0.0),
        )

    elif provider == "modelscope":
        base_url = kwargs.get("base_url", os.getenv("MODELSCOPE_ENDPOINT", ""))
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model_name=kwargs.get("model_name", "Qwen/QwQ-32B"),
            temperature=kwargs.get("temperature", 0.0),
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
