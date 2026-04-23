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

## 快速体验 (Release 下载)

无需配置 Python 环境，直接下载打包好的可执行文件即可体验完整的交互式评测与报告生成   
👉 **[点击下载 UX Evaluator v1.0.0](https://github.com/ChunQiu203/deepeval_pro/releases/download/v1.0.0/UX_Evaluator_v1.0.0.zip
)**

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

## 质量保障（QA）

当前已建立数据加载模块的基础回归测试，覆盖以下场景：
- 空样本过滤（`clean_data=True`）
- 文件不存在时抛出 `FileNotFoundError`
- 字段别名兼容（`question/query` 与 `answer/output`）

运行方式：

```bash
python -m pytest tests\test_dataset_loader.py tests\test_batch_evaluate.py -q
```

生成评估结果与文本报告：

```bash
python scripts/batch_evaluate.py --config configs/default_config.yaml
# 首次运行前请先确保输出目录存在（Windows PowerShell）：New-Item -ItemType Directory -Path tests/reports -Force
python scripts/generate_report.py --input results/evaluation_results.json --output tests/reports/qa_report.txt
```

一键执行 QA 门禁（测试 + 评估 + 报告）：

```bash
python scripts/qa_check.py --config configs/default_config.yaml --results-path results/evaluation_results.json --report-path tests/reports/qa_report.txt
```

报告文件位置：
- `results/evaluation_results.json`
- `tests/reports/qa_report.txt`
