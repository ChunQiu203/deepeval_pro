# UX Evaluator: 用户体验评价工具
基于 DeepEval 框架的 LLM 应用用户体验量化评价工具，将主观用户感受转化为可自动化评估的客观指标。

## 项目结构
```
deepeval_pro/
├── ux_evaluator/         # 核心代码包
│   ├── __init__.py
│   ├── metrics/          # 自定义UX指标模块
│   ├── judge/            # LLM-as-a-Judge评判模块
│   ├── dataset/          # 测试数据集管理模块
│   └── utils/            # 通用工具类
├── tests/                # 单元测试与集成测试
├── examples/             # 示例代码与演示用例
├── scripts/              # 自动化执行脚本
├── configs/              # 指标配置与环境配置
├── .github/              # GitHub Actions配置
├── pyproject.toml        # 项目依赖配置
├── .gitignore            # Git 忽略文件配置
└── README.md             # 项目说明文档
```

## 团队分工
| 角色 | 负责模块 | 
|------|---------|
| PM&算法逻辑 | metrics/自定义UX指标模块 | 
| 技术负责人 | judge/+api/框架集成模块 | 
| 技术骨干 | dataset/ + scripts/数据与自动化模块 |
| 产品与质量负责人 | 报告与质量保障模块 | 

## 快速开始
### 1. 安装依赖
```bash
pip install -r requirements.txt
# 或者使用项目安装
pip install -e .
```

### 2. 配置 API Key
```bash
# Windows Command Prompt
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

在 `.env` 中填写你的 DashScope API Key：
- `QWEN_API_KEY=...`（推荐）
- 或 `DASHSCOPE_API_KEY=...`

### 3. 运行示例
```bash
python examples/rag_evaluation_example.py
```

### 4. 批量评估
```bash
python scripts/batch_evaluate.py --config configs/default_config.yaml
```

## 模块对接说明
1. **数据集模块**：提供标准化的 `TestCase` 格式，其他模块可直接导入使用
2. **自动化脚本**：预留了与 `metrics` 和 `judge` 模块的对接接口，待对应模块完成后可直接替换占位实现
3. 所有模块统一使用 Python >=3.12 版本，依赖版本见 `requirements.txt`
