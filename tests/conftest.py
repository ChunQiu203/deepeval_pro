# =============================================================================
# Pytest 全局：仅为「未配置 OPENAI_API_KEY」的环境注入占位密钥
# =============================================================================
# 作用：DeepEval 在 GEval 等指标构造期会校验密钥；占位值只用于通过校验，不会发起成功请求。
# 含义：若本机/CI 已设置真实 OPENAI_API_KEY，则绝不覆盖——避免误把占位值写进真实 E2E。
# =============================================================================

from __future__ import annotations

import os


def pytest_configure(config) -> None:  # noqa: ARG001
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        os.environ["OPENAI_API_KEY"] = "sk-test-placeholder-for-deepeval-init-only"
