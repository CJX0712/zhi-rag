"""Pluggable answer generation.

  * MockGenerator  -> OFFLINE default. Deterministic, extractive synthesis from
                      retrieved contexts. Guarantees zero-dependency demo.
  * OllamaGenerator -> local LLM via the `ollama` client (optional).
  * OpenAIGenerator -> OpenAI-compatible API (optional).

All implement generate(question, contexts) -> str.
"""
import re
import warnings

import jieba

from ..errors import ZDError, ZDCode


def _keywords(text):
    return {w for w in jieba.lcut(text) if len(w) > 1}


class MockGenerator:
    name = "mock"

    def generate(self, question: str, contexts) -> str:
        if not contexts:
            return "（未检索到相关资料，无法作答）"
        qkw = _keywords(question)
        snippets = []
        for c in contexts:
            for sent in re.split(r"[。！？\n]", c.chunk.text):
                s = sent.strip()
                if not s:
                    continue
                if qkw & _keywords(s):
                    snippets.append(s)
        if not snippets:
            snippets = [c.chunk.text[:60] for c in contexts]
        srcs = "、".join(sorted({c.chunk.source for c in contexts}))
        ans = f"基于检索到的 {len(contexts)} 段资料，回答如下：\n"
        ans += "；".join(snippets[:6])
        ans += f"\n\n资料来源：{srcs}"
        imgs = [c.chunk.meta.get("image_path", "") for c in contexts if c.chunk.modality == "image"]
        if imgs:
            ans += f"\n引用图像：{'、'.join(sorted(set(imgs)))}"
        return ans


class OllamaGenerator:
    name = "ollama"

    def __init__(self, model: str, base_url: str):
        try:
            import ollama
        except Exception as e:
            raise ZDError(ZDCode.E_GEN, "ollama client unavailable (optional dep)", str(e))
        self.client = ollama.Client(host=base_url)
        self.model = model

    def generate(self, question: str, contexts) -> str:
        ctx = "\n\n".join(f"[资料{i+1}] {c.chunk.text}" for i, c in enumerate(contexts))
        prompt = (
            "以下是检索到的资料：\n" + ctx + "\n\n请根据资料回答用户问题，并注明引用。\n问题：" + question
        )
        try:
            r = self.client.generate(model=self.model, prompt=prompt, stream=False)
            return r.get("response", "")
        except Exception as e:
            raise ZDError(ZDCode.E_GEN, "ollama generate failed", str(e))


class OpenAIGenerator:
    name = "openai"

    def __init__(self, model: str, api_key: str):
        try:
            from openai import OpenAI
        except Exception as e:
            raise ZDError(ZDCode.E_GEN, "openai client unavailable (optional dep)", str(e))
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, contexts) -> str:
        ctx = "\n\n".join(f"[资料{i+1}] {c.chunk.text}" for i, c in enumerate(contexts))
        sys = "你是严谨的检索增强问答助手，仅依据给定资料作答并标注引用。"
        usr = f"资料：\n{ctx}\n\n问题：{question}"
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                temperature=0,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            raise ZDError(ZDCode.E_GEN, "openai generate failed", str(e))


def get_generator(cfg):
    if cfg.generator == "ollama":
        try:
            return OllamaGenerator(cfg.ollama_model, cfg.ollama_base_url)
        except ZDError as e:
            warnings.warn(f"ollama unavailable, falling back to mock: {e.msg}")
            return MockGenerator()
    if cfg.generator == "openai":
        if not cfg.openai_api_key:
            warnings.warn("openai_api_key missing, falling back to mock")
            return MockGenerator()
        try:
            return OpenAIGenerator(cfg.openai_model, cfg.openai_api_key)
        except ZDError as e:
            warnings.warn(f"openai unavailable, falling back to mock: {e.msg}")
            return MockGenerator()
    return MockGenerator()
