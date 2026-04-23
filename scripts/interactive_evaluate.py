# scripts/interactive_evaluate.py
"""交互式配置与批量评测入口脚本"""
# 注：推荐使用终端运行python.exe interactive_evaluate.py，优于在IDE中直接点击运行
import os
import sys
import yaml
import json
import traceback
import subprocess  # 用于调用自动化验证与报告生成脚本
from typing import Any, Dict, List

from dotenv import load_dotenv

# ----------------- UI 库 Rich 配置模块 -----------------
# 引入现代 CLI UI 库 Rich，大幅优化视觉体验和排版
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt
    from rich import box

    console = Console()
except ImportError:
    print("❌ 错误: 缺少必要的依赖库 'rich'。")
    print("请运行: pip install rich")
    sys.exit(1)
# ---------------------------------------------------

# --- 兼容 PyInstaller 打包环境的路径逻辑 ---
if getattr(sys, 'frozen', False):
    # 如果是打包后的 .exe 运行环境
    # 1. sys.executable 指向用户双击的 UX_Evaluator.exe 所在的真实物理路径
    project_root = os.path.dirname(sys.executable)
    # 2. sys._MEIPASS 是 exe 在后台解压代码的临时目录，确保能 import 里面的模块
    sys.path.insert(0, sys._MEIPASS)
else:
    # 如果是原生 Python 脚本运行环境
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
# --------------------------------------------------------

# 强制加载根目录下的 .env 文件
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

from ux_evaluator.judge import UXJudge
from ux_evaluator.dataset import DatasetLoader
from scripts.batch_evaluate import init_metrics, _resolve_path
# 导入指标规范字典，用于帮助说明和动态配置
from ux_evaluator.metrics import METRIC_SPECS
from ux_evaluator.metrics.dag import (
    ConversationDAGEvaluator,
    load_conversation_cases,
    load_dag_config,
    load_turn_metric_configs,
)

# 全局中英文字段映射字典，用于界面显示与结果回显
EN_TO_ZH_MAP = {
    "trust": "信任感",
    "understanding": "理解度",
    "control": "掌控感",
    "efficiency": "效率",
    "cognitive_load": "认知负荷",
    "satisfaction": "满意度",
    "safety": "安全性",
    "dependency": "依赖性",
    "anthropomorphism": "拟人化",
    "empathy": "共情性"
}


def clear_console():
    """
    跨平台的强力清屏函数。
    为防止 PyCharm/VSCode 等 IDE 终端吞噬原生清屏指令，加入换行顶屏兜底逻辑。
    """
    # 兼容处理：先打印空行将旧内容顶上去
    print("\n")
    # 原生系统级清屏
    os.system('cls' if os.name == 'nt' else 'clear')
    # 强制刷新输出流，防止缓冲区滞后（解决需要敲多次回车的问题）
    sys.stdout.flush()


def load_default_config(project_root):
    """加载默认的 yaml 配置文件到内存"""
    config_path = os.path.join(project_root, "configs/default_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], project_root: str):
    """将当前内存中的配置持久化保存到 yaml 文件"""
    config_path = os.path.join(project_root, "configs/default_config.yaml")
    try:
        # 使用 safe_dump 将字典写回 yaml，取消默认的字典排序以保持一定可读性
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        console.print(f"[bold red]⚠️ 自动保存配置失败: {e}[/]")


def ensure_interactive_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """补齐交互脚本需要的默认字段，不影响原有 classic 配置。"""
    if "batch" not in config:
        config["batch"] = {"output_path": "results/evaluation_results.json"}

    batch = config["batch"]
    batch.setdefault("output_path", "results/evaluation_results.json")
    batch.setdefault("evaluation_mode", "classic")
    batch.setdefault("dag_config_path", "examples/dag_config.json")
    batch.setdefault("metrics_path", "examples/metrics.json")

    # --- 集成组员的高级参数默认值 ---
    batch.setdefault("top_failures", 3)
    config.setdefault("dataset", {}).setdefault("clean_data", True)
    config.setdefault("judge", {}).setdefault("retry_times", 3)

    # --- 门禁与报告默认配置 ---
    batch.setdefault("min_pass_rate", 0.70)  # 默认质量门禁阈值 70%
    batch.setdefault("report_path", "tests/reports/qa_report.txt")

    return config


def show_help():
    """显示帮助菜单与所有指标的具体说明（增加翻页逻辑，使用 Rich 优化排版）"""

    # ------------------ 第 1 页：操作说明 ------------------
    clear_console()

    help_text = """
本工具支持对大模型回复进行多维度的 UX（用户体验）量化评估。您可以随时在主菜单修改相关参数。

[bold magenta]【🤖 支持的模型类型 (Supported Models)】[/]
  系统会自动根据输入的模型名称前缀匹配对应的 API 厂商：
  [cyan]- 通义千问 (Qwen)系列:[/] 如 'qwen-turbo', 'qwen-plus', 'qwen-max'
    [dim]* 需在项目根目录 .env 文件配置 QWEN_API_KEY[/]
  [cyan]- 深度求索 (DeepSeek)系列:[/] 如 'deepseek-chat', 'deepseek-reasoner'
    [dim]* 需在项目根目录 .env 文件配置 DEEPSEEK_API_KEY[/]
  [dim](如需扩展其他模型，请前往 metric_judge.py 中按需添加 API 路由)[/]

[bold magenta]【🚀 评测模式说明 (Evaluation Modes)】[/]
  [cyan]1. Classic (经典单轮) 模式:[/cyan]
     - 适用场景：简单的单轮问答、RAG生成结果评测。
     - 运行逻辑：系统使用预设的内存指标集，直接读取单行数据集对模型回复进行打分。
  [cyan]2. DAG (有向无环图多轮) 模式:[/cyan]
     - 适用场景：复杂的多轮会话、Agent 任务执行评测。
     - 运行逻辑：基于设定的 DAG 结构配置文件评估会话状态的流转、连贯性与最终目标达成度。
     [dim]* 注：DAG模式需额外配置【DAG 结构配置】与【DAG 指标配置】路径。[/]

[bold magenta]【📊 评价指标与门禁 (Metrics & Gate)】[/]
  [cyan]- 动态启停:[/] 可在菜单 [4] 中随意开启或关闭指定 UX 维度。
  [cyan]- 阈值调整:[/] 每个指标打分范围为 0.0~1.0。大于等于设定的通过阈值即判定该指标 Pass。
  [cyan]- 质量门禁:[/] 最终所有 Case 算出的总通过率若低于【门禁阈值】，将触发红色拦截预警。

[bold magenta]【🛡️ 质量保障与导出报告 (QA & Report)】[/]
  [cyan]- 健康检查:[/] 一键执行集成测试 (pytest)，确保核心代码逻辑正常。
  [cyan]- 自动报告:[/] 评测结束后自动生成分析报告。支持 [cyan].txt[/] (纯文本) 或 [cyan].md[/] (Markdown 表格) 格式。
    """

    console.print(
        Panel(help_text, title="[bold magenta]📖 UX Evaluator 详细帮助指南 - [第 1/2 页][/]", border_style="cyan"))
    Prompt.ask("\n[bold yellow]💡 提示：按 [回车键] 继续查看下一页【评价指标字典说明】...[/]")

    # ------------------ 第 2 页：指标字典 ------------------
    clear_console()

    dict_text = ""
    for key, spec in METRIC_SPECS.items():
        en_name = spec.get("name", key)
        zh_name = EN_TO_ZH_MAP.get(key, en_name)
        criteria = spec.get("criteria", "无说明").strip()

        dict_text += f"[bold cyan]🔹 {zh_name} ({en_name}) [键值/Key: {key}][/]\n"
        dict_text += f"[dim]{'-' * 50}[/]\n"
        dict_text += f"{criteria}\n"
        dict_text += f"[dim]{'-' * 50}[/]\n\n"

    console.print(Panel(dict_text.strip(), title="[bold magenta]📖 评价指标字典 - [第 2/2 页][/]", border_style="cyan"))

    console.print("\n[bold green]💡 提示：已经到底啦！[/]")
    Prompt.ask("[bold yellow]按 [回车键] 返回主菜单...[/]")


def run_health_check():
    """自动化验证：运行系统健康检查 (pytest)"""
    clear_console()

    # --- 新增：如果在 exe 环境下直接拦截 ---
    if getattr(sys, 'frozen', False):
        console.print(Panel("[bold yellow]⚠️ 提示：系统健康检查 (pytest) 仅在源码开发环境下可用。产品发布版无需运行。[/]", border_style="yellow"))
        Prompt.ask("\n[bold yellow]按 [回车键] 返回主菜单...[/]")
        return
    # --------------------------------------

    console.print(Panel("[bold cyan]🔄 正在后台运行系统健康检查 (执行 pytest)...[/]", border_style="cyan"))

    try:
        # 调用 pytest，静默输出以保持界面清爽
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"])
        if result.returncode == 0:
            console.print("\n[bold green]✅ 所有核心自动化测试用例通过！系统运行健康。[/]")
        else:
            console.print("\n[bold red]❌ 测试未完全通过，请检查最近的代码变动或依赖问题！[/]")
    except Exception as e:
        console.print(f"\n[bold red]⚠️ 运行测试时发生异常：{e}[/]")

    Prompt.ask("\n[bold yellow]按 [回车键] 返回主菜单...[/]")


def _render_gate_status(pass_rate, min_threshold):
    """渲染质量门禁判定结果"""
    status_panel = ""
    if pass_rate >= min_threshold:
        status_panel = Panel(
            f"[bold green]🟢 GATE PASSED (通过)[/]\n当前通过率 {pass_rate:.2%} >= 门禁阈值 {min_threshold:.2%}",
            border_style="green", title="质量门禁判定"
        )
    else:
        status_panel = Panel(
            f"[bold red]🔴 GATE FAILED (拦截)[/]\n当前通过率 {pass_rate:.2%} < 门禁阈值 {min_threshold:.2%}",
            border_style="red", title="质量门禁判定"
        )
    console.print(status_panel)


def _generate_report(config, output_path, report_path):
    """调用脚本自动生成分析报告（兼容 .exe 打包环境）"""
    # 直接引入生成报告的 main 函数，摆脱 subprocess，防止 exe 打包后找不到环境
    from scripts.generate_report import main as gen_report_main

    # 自动推断报告格式
    fmt = "md" if report_path.lower().endswith(".md") else "txt"
    top_failures = str(config.get("batch", {}).get("top_failures", 3))

    # 备份原生命令行参数，伪造 sys.argv 传给 generate_report
    old_argv = sys.argv.copy()
    try:
        sys.argv = [
            "generate_report.py",
            "--input", output_path,
            "--output", report_path,
            "--format", fmt,
            "--top-failures", top_failures
        ]
        gen_report_main() # 直接调用，无缝生成报告
        console.print(f"[bold green]📄 已自动生成分析报告 ({fmt.upper()} 格式): {report_path}[/]")
    except Exception as e:
        console.print(f"[bold red]⚠️ 报告生成失败: {e}[/]")
    finally:
        sys.argv = old_argv # 恢复参数


def modify_metrics_menu(config):
    """动态增删评价指标及修改参数的交互菜单（状态拨动式交互）"""

    # 构建所有指标的基础列表，方便索引
    all_metrics_info = []
    for key, spec in METRIC_SPECS.items():
        en_name = spec["name"]
        zh_name = EN_TO_ZH_MAP.get(key, en_name)
        all_metrics_info.append({
            "key": key,
            "en_name": en_name,
            "zh_name": zh_name
        })

    while True:
        clear_console()
        current_metrics = config.get("metrics", [])

        # 使用 Rich Table 渲染当前配置状态
        table = Table(title="[bold magenta]📊 评价指标配置 (Configure Metrics)[/]", box=box.ROUNDED, border_style="cyan")
        table.add_column("序号", justify="center", style="cyan", no_wrap=True)
        table.add_column("指标名称", style="white")
        table.add_column("状态", justify="center")
        table.add_column("当前阈值", justify="center", style="green")

        for idx, info in enumerate(all_metrics_info, 1):
            key = info["key"]
            zh_name = info["zh_name"]
            en_name = info["en_name"]

            # 检查当前指标是否在已启用列表中
            current_match = next(
                (m for m in current_metrics if m.get("name") in [zh_name, en_name, key] or m.get("metric_key") == key),
                None
            )

            is_enabled = current_match is not None
            curr_thresh = current_match.get("threshold", 0.5) if is_enabled else "-"
            status_text = "[bold green]🟢 已启用[/]" if is_enabled else "[dim]⚪ 未启用[/]"

            table.add_row(str(idx), f"{zh_name} ({en_name})", status_text, str(curr_thresh))

        console.print(table)
        console.print("[dim]提示：输入序号可以切换启停状态或修改阈值。输入 'q' 完成配置并返回。[/]")

        choice = Prompt.ask("\n[bold yellow]👉 请输入操作的序号 (或 'q' 返回)[/]").strip().lower()

        if choice == 'q':
            console.print(f"\n[bold green]✅ 配置已同步！当前生效指标数: {len(current_metrics)}[/]")
            Prompt.ask("[bold yellow]按 [回车键] 返回主菜单...[/]")
            break

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_metrics_info):
                target_info = all_metrics_info[idx]
                target_key = target_info["key"]

                # 查找是否存在（修复匹配逻辑：同时兼容 name 和 metric_key）
                existing_match = next(
                    (m for m in current_metrics if
                     m.get("metric_key") == target_key or m.get("name") in [target_info["zh_name"],
                                                                            target_info["en_name"], target_key]),
                    None
                )

                if existing_match:
                    # 如果已存在，提供二级菜单让用户选择【停用】或【修改阈值】
                    sub_choice = Prompt.ask(
                        f"\n[bold cyan]指标 [{target_info['zh_name']}] 已启用。[/] 请选择操作: [1] 停用该指标  [2] 修改阈值  [0] 取消",
                        choices=["1", "2", "0"],
                        default="0"
                    )

                    if sub_choice == "1":
                        # 剔除匹配到的指标
                        config["metrics"] = [
                            m for m in current_metrics
                            if not (m.get("metric_key") == target_key or m.get("name") in [target_info["zh_name"],
                                                                                           target_info["en_name"],
                                                                                           target_key])
                        ]
                        console.print(f"[dim]❌ 已停用指标: {target_info['zh_name']}[/]")
                        Prompt.ask("[dim]按 [回车键] 继续...[/]")

                    elif sub_choice == "2":
                        # 修改阈值
                        curr_val = existing_match.get("threshold", 0.5)
                        thresh_str = Prompt.ask(f"[bold yellow]➤ 请输入新的阈值 (0.0~1.0)[/]", default=str(curr_val))
                        try:
                            thresh_val = float(thresh_str)
                            existing_match["threshold"] = thresh_val
                            # 顺便强制补充 metric_key 以标准化内存中的配置
                            existing_match["metric_key"] = target_key
                            console.print(f"[bold green]✅ 阈值已更新为: {thresh_val}[/]")
                        except ValueError:
                            console.print("[bold red]⚠️ 输入无效，保持原阈值。[/]")
                        Prompt.ask("[dim]按 [回车键] 继续...[/]")
                else:
                    # 如果不存在，则启用它，并询问阈值
                    thresh_str = Prompt.ask(f"\n[bold yellow]➤ 请输入 {target_info['zh_name']} 的通过阈值 (0.0~1.0)[/]",
                                            default="0.5")
                    try:
                        thresh_val = float(thresh_str)
                    except ValueError:
                        console.print("[bold red]⚠️ 输入无效，默认设置为 0.5[/]")
                        thresh_val = 0.5

                    config["metrics"].append({
                        "name": target_info["zh_name"],
                        "metric_key": target_key,
                        "threshold": thresh_val
                    })
                    console.print(f"[bold green]✅ 已启用指标: {target_info['zh_name']} (阈值: {thresh_val})[/]")
                    Prompt.ask("[dim]按 [回车键] 继续...[/]")
            else:
                console.print("[bold red]⚠️ 无效的序号，请重新输入。[/]")
                Prompt.ask("[dim]按 [回车键] 继续...[/]")
        else:
            console.print("[bold red]⚠️ 请输入正确的数字或 'q'。[/]")
            Prompt.ask("[dim]按 [回车键] 继续...[/]")


def _render_dag_summary(results: Dict[str, Any], output_path: str):
    summary = results.get("summary", {})
    console.print(f"\n[bold green]💾 DAG 评估完成！结果已成功保存至: {output_path}[/]")

    overview = Table(title="[bold magenta]🧭 多轮对话评测摘要[/]", box=box.SIMPLE, show_header=False)
    overview.add_column("项目", style="cyan")
    overview.add_column("值", justify="right", style="bold green")
    overview.add_row("会话数", str(summary.get("case_count", 0)))
    overview.add_row("轮次结果数", str(results.get("turn_result_count", 0)))
    overview.add_row("单轮总平均分", f"{summary.get('overall_average', 0):.4f}")
    overview.add_row("DAG 全局平均分", f"{summary.get('dag_overall_average', 0):.4f}")
    console.print(overview)

    dag_node_average = summary.get("dag_node_average") or {}
    if dag_node_average:
        node_table = Table(title="[bold magenta]🕸️ DAG 节点平均分[/]", box=box.SIMPLE, show_header=False)
        node_table.add_column("节点", style="cyan")
        node_table.add_column("得分", justify="right", style="bold green")
        for node_name, score in dag_node_average.items():
            node_table.add_row(node_name, f"{score:.4f}")
        console.print(node_table)


def run_dag_evaluation(config, project_root):
    """执行 DAG 多轮会话评测逻辑"""
    clear_console()

    with console.status("[bold cyan]⏳ 正在初始化 DAG 评测器...[/]", spinner="dots"):
        try:
            judge = UXJudge(
                model=config["judge"]["model"],
                retry=config["judge"].get("retry_times", 3),
                base_url=config["judge"].get("base_url"),
            )

            data_path = _resolve_path(project_root, config["dataset"]["data_path"])
            dag_config_path = _resolve_path(project_root, config["batch"]["dag_config_path"])
            metrics_path = _resolve_path(project_root, config["batch"]["metrics_path"])
            output_path = _resolve_path(project_root, config["batch"]["output_path"])

            cases = load_conversation_cases(data_path)
            node_specs = load_dag_config(dag_config_path)
            evaluator = ConversationDAGEvaluator(judge=judge, node_specs=node_specs)

            turn_metrics: List[Any]
            if config["batch"].get("metrics_path"):
                metric_configs = load_turn_metric_configs(metrics_path)
                turn_metrics = evaluator.build_turn_metrics_from_configs(metric_configs)
            else:
                metrics_list = config.get("metrics", [])
                if not metrics_list:
                    console.print("\n[bold red]❌ 错误：DAG 模式下未找到 metrics_path，也没有 classic metrics 可回退。[/]")
                    Prompt.ask("\n[bold yellow]按 [回车键] 返回主菜单...[/]")
                    return
                turn_metrics = init_metrics(metrics_list, judge.custom_model)

        except Exception as e:
            console.print(f"\n[bold red]❌ DAG 初始化阶段出现错误: {str(e)}[/]")
            traceback.print_exc()
            Prompt.ask("\n[bold yellow]按 [回车键] 返回主菜单...[/]")
            return

    try:
        results = evaluator.evaluate_cases(cases=cases, metrics=turn_metrics)

        # 保存 JSON 结果
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        _render_dag_summary(results, output_path)

        # --- 集成门禁与报告 ---
        summary = results.get("summary", {})
        pass_rate = summary.get("passed_case_count", 0) / summary.get("case_count", 1)
        _render_gate_status(pass_rate, config["batch"].get("min_pass_rate", 0.70))

        report_path = _resolve_path(project_root, config["batch"]["report_path"])
        _generate_report(config, output_path, report_path)

    except Exception as e:
        console.print(f"\n[bold red]❌ 运行 DAG 评估过程中出现错误: {str(e)}[/]")
        traceback.print_exc()

    Prompt.ask("\n[bold yellow]测试结束，按 [回车键] 返回主菜单...[/]")


def run_batch_evaluation(config, project_root):
    """执行经典批量测试集的评测逻辑"""
    if config.get("batch", {}).get("evaluation_mode", "classic") == "dag":
        run_dag_evaluation(config, project_root)
        return

    clear_console()

    # 动态加载动画
    with console.status("[bold cyan]⏳ 正在初始化 LLM Judge 和 评估指标...[/]", spinner="dots"):
        try:
            # 初始化评判模型
            judge = UXJudge(
                model=config["judge"]["model"],
                retry=config["judge"].get("retry_times", 3)
            )

            # 使用当前内存中的 config 初始化指标
            metrics_list = config.get("metrics", [])
            if not metrics_list:
                console.print("\n[bold red]❌ 错误：当前未配置任何评价指标。[/]")
                Prompt.ask("\n[bold yellow]按 [回车键] 返回主菜单...[/]")
                return

            metrics = init_metrics(metrics_list, judge.custom_model)

            # 加载数据集
            data_path = _resolve_path(project_root, config["dataset"]["data_path"])
            loader = DatasetLoader(clean_data=config["dataset"].get("clean_data", True))
            test_cases = loader.load_from_file(data_path)

        except Exception as e:
            console.print(f"\n[bold red]❌ 初始化阶段出现错误: {str(e)}[/]")
            traceback.print_exc()
            Prompt.ask("\n[bold yellow]按 [回车键] 返回主菜单...[/]")
            return

    # 将耗时的 API 调用放在另一个 status 下
    try:
        results = judge.batch_evaluate(test_cases, metrics)
        summary = judge._aggregate_results(results)

        # 保存结果
        output_path = _resolve_path(project_root, config["batch"]["output_path"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        console.print(f"\n[bold green]💾 评估完成！结果已成功保存至: {output_path}[/]")

        # 打印得分表格
        if summary.get("metric_average"):
            res_table = Table(title="[bold magenta]📈 平均得分统计 (Average Scores)[/]", box=box.SIMPLE,
                              show_header=False)
            res_table.add_column("指标名称", style="cyan")
            res_table.add_column("得分", justify="right", style="bold green")

            # 建立英文名到中文名的反向映射字典
            name_to_zh = {spec["name"]: EN_TO_ZH_MAP.get(k, spec["name"]) for k, spec in METRIC_SPECS.items()}
            for name, avg in summary["metric_average"].items():
                # --- 清除 DeepEval 自动加上的 [GEval] 后缀 ---
                clean_name = name.replace(" [GEval]", "")
                zh_name = name_to_zh.get(clean_name, clean_name)
                # ------------------------------------------------
                res_table.add_row(f"• {zh_name} ({clean_name})", f"{avg:.4f}")
            console.print(res_table)

        console.print(
            f"\n[bold magenta]🌟 总平均分 (Overall Score):[/] [bold green]{summary.get('overall_average', 0):.4f}[/]")

        # --- 集成门禁与报告 ---
        total_count = len(results)
        passed_count = sum(1 for r in results if r.get("overall_passed"))
        pass_rate = passed_count / total_count if total_count > 0 else 0

        _render_gate_status(pass_rate, config["batch"].get("min_pass_rate", 0.70))

        report_path = _resolve_path(project_root, config["batch"]["report_path"])
        _generate_report(config, output_path, report_path)

    except Exception as e:
        console.print(f"\n[bold red]❌ 运行评测过程中出现错误: {str(e)}[/]")

    Prompt.ask("\n[bold yellow]测试结束，按 [回车键] 返回主菜单...[/]")


def main():
    # --- 使用兼容 PyInstaller 的路径，避免覆盖全局正确路径 ---
    if getattr(sys, 'frozen', False):
        project_root = os.path.dirname(sys.executable)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # ----------------------------------------------------------------

    # 初始化配置
    try:
        config = load_default_config(project_root)
    except FileNotFoundError:
        console.print("[bold red]❌ 错误：找不到 configs/default_config.yaml 文件。[/]")
        return

    ensure_interactive_defaults(config)

    # 主菜单循环
    while True:
        clear_console()

        # 面板数据构建
        current_metrics = config.get('metrics', [])
        metrics_display = "".join(
            [f"      [dim]* {m.get('name', 'Unknown')} (阈值: {m.get('threshold', 0.5)})[/]\n" for m in current_metrics])

        evaluation_mode = config['batch'].get('evaluation_mode', 'classic')
        min_pass_rate = config['batch'].get('min_pass_rate', 0.70)
        current_output_path = config['batch'].get('output_path', '未配置')
        current_report_path = config['batch'].get('report_path', '未配置')
        curr_dag = config['batch'].get('dag_config_path', '未配置')
        curr_metrics_json = config['batch'].get('metrics_path', '未配置')

        # 动态判定是否为 DAG 模式来决定路径高亮状态，使排版对齐
        if evaluation_mode == "dag":
            dag_display = f"""
      - [cyan]DAG 结构配置 (DAG Path)  [/]: [bold yellow]{curr_dag}[/]
      - [cyan]DAG 指标配置 (Metrics Path)[/]: [bold yellow]{curr_metrics_json}[/]"""
        else:
            dag_display = f"""
      - [dim]DAG 结构配置 (DAG Path)  [/]: [dim]{curr_dag} (仅DAG模式生效)[/]
      - [dim]DAG 指标配置 (Metrics Path)[/]: [dim]{curr_metrics_json} (仅DAG模式生效)[/]"""

        menu_text = f"""
    [bold magenta]【当前配置概览 (Configuration)】[/]
      - [cyan]评测模式 (Mode)          [/]: [bold green]{evaluation_mode.upper()}[/]
      - [cyan]评判模型 (Model)         [/]: [bold green]{config['judge']['model']}[/]
      - [cyan]门禁阈值 (Threshold)     [/]: [bold green]{min_pass_rate:.0%}[/]
      - [cyan]数据集路径 (Data In)     [/]: [bold green]{config['dataset']['data_path']}[/]
      - [cyan]输出报告路径 (Report Out)[/]: [bold green]{current_report_path} (支持 .txt / .md)[/]
      - [cyan]输出结果路径 (JSON Out)  [/]: [bold green]{current_output_path}[/]{dag_display}
      - [cyan]生效指标 (Active Metrics)[/]: [bold green]{len(current_metrics)} 个[/]
{metrics_display}
    [bold magenta]【操作菜单 (Main Menu)】[/]
      [bold green][1] ⚡ 运行批量评测 (Run Evaluation)[/]
      [bold cyan][9] 🛠️ 系统健康检查 (Health Check / pytest)[/]
      [dim]-----------------------------------------[/]
      [cyan][2] 🔧 修改评判模型 (Change Judge Model)[/]
      [cyan][3] 📂 修改数据集路径 (Change Data Path)[/]
      [cyan][4] 📊 配置评价指标 (Configure Metrics)[/]
      [cyan][5] 📁 修改输出与报告路径 (Change Output Paths)[/]
      [cyan][6] 🕸️ 切换评测模式 (Classic <-> DAG)[/]
      [cyan][7] 🧭 修改 DAG 相关配置路径 (DAG Config Paths)[/]
      [cyan][8] 🧾 修改门禁阈值与高级参数 (Thresholds & Advanced)[/]
      [dim]\\[/help] 📖 查看帮助与说明 (Help)[/]
      [dim][0] 🚪 退出程序 (Exit)[/]
    """
        console.print(
            Panel(menu_text.strip(), title="[bold magenta]🚀 UX Evaluator 交互式控制台[/]", border_style="cyan"))

        choice = Prompt.ask("[bold yellow]👉 请输入操作指令[/]").strip().lower()

        if choice == '1':
            run_batch_evaluation(config, project_root)

        elif choice == '2':
            new_model = Prompt.ask(
                f"\n[bold yellow]👉 请输入新模型名 (直接回车保持当前: [dim]{config['judge']['model']}[/])[/]").strip()
            if new_model:
                config['judge']['model'] = new_model
                save_config(config, project_root)
                console.print(f"[bold green]✅ 模型已更新并自动保存为: {new_model}[/]")
                Prompt.ask("[dim]按回车继续...[/]")

        elif choice == '3':
            new_path = Prompt.ask(
                f"\n[bold yellow]👉 请输入新数据集路径 (直接回车保持当前: [dim]{config['dataset']['data_path']}[/])[/]").strip()
            if new_path:
                config['dataset']['data_path'] = new_path
                save_config(config, project_root)
                console.print(f"[bold green]✅ 数据集路径已更新并保存为: {new_path}[/]")
                Prompt.ask("[dim]按回车继续...[/]")

        elif choice == '4':
            modify_metrics_menu(config)
            save_config(config, project_root)

        elif choice == '5':
            new_out = Prompt.ask(
                f"\n[bold yellow]👉 请输入新输出 JSON 路径 (直接回车保持当前: [dim]{current_output_path}[/])[/]").strip()
            if new_out:
                config['batch']['output_path'] = new_out
                save_config(config, project_root)
                console.print(f"[bold green]✅ JSON 输出路径已更新并保存为: {new_out}[/]")

            new_rep = Prompt.ask(
                f"[bold yellow]👉 请输入新报告路径 [支持 .txt 或 .md] (直接回车保持当前: [dim]{current_report_path}[/])[/]").strip()
            if new_rep:
                config['batch']['report_path'] = new_rep
                save_config(config, project_root)
                console.print(f"[bold green]✅ 报告路径已更新并保存为: {new_rep}[/]")

            Prompt.ask("\n[bold yellow]按 [回车键] 返回...[/]")

        elif choice == '6':
            config['batch']['evaluation_mode'] = "dag" if evaluation_mode == "classic" else "classic"
            save_config(config, project_root)
            console.print(f"\n[bold green]✅ 模式已切换并保存为: {config['batch']['evaluation_mode'].upper()}[/]")
            Prompt.ask("[dim]按回车继续...[/]")

        elif choice == '7':
            new_dag = Prompt.ask(
                f"\n[bold yellow]👉 请输入新的 DAG 结构配置 JSON 路径 (直接回车保持当前: [dim]{curr_dag}[/])[/]").strip()
            if new_dag:
                config['batch']['dag_config_path'] = new_dag

            new_metrics = Prompt.ask(
                f"[bold yellow]👉 请输入新的 DAG 评测指标 JSON 路径 (直接回车保持当前: [dim]{curr_metrics_json}[/])[/]").strip()
            if new_metrics:
                config['batch']['metrics_path'] = new_metrics

            save_config(config, project_root)
            console.print("\n[bold green]✅ DAG 相关路径配置已更新并保存！[/]")
            Prompt.ask("[bold yellow]按 [回车键] 返回...[/]")

        elif choice == '8':
            clear_console()
            console.print(Panel("[bold magenta]⚙️ 门禁阈值与高级参数设置[/]", border_style="cyan"))

            # 1. 门禁通过率
            new_rate = Prompt.ask(
                f"\n[bold yellow]👉 请输入新门禁阈值 0.0-1.0 (直接回车保持当前: [dim]{min_pass_rate}[/])[/]").strip()
            if new_rate:
                try:
                    val = float(new_rate)
                    if 0 <= val <= 1:
                        config['batch']['min_pass_rate'] = val
                        save_config(config, project_root)
                        console.print(f"[bold green]✅ 门禁阈值已更新为: {val:.0%}[/]")
                    else:
                        console.print("[bold red]⚠️ 阈值必须在 0.0 到 1.0 之间。[/]")
                except ValueError:
                    console.print("[bold red]⚠️ 输入的不是有效的数字。[/]")

            # 2. 失败用例展示数量
            curr_top = config['batch'].get('top_failures', 3)
            new_top = Prompt.ask(f"[bold yellow]👉 报告中展示的失败用例数 (当前: {curr_top})[/]",
                                 default=str(curr_top)).strip()
            if new_top.isdigit():
                config['batch']['top_failures'] = int(new_top)

            # 3. 数据集清洗开关
            curr_clean = config['dataset'].get('clean_data', True)
            clean_status = "开" if curr_clean else "关"
            new_clean = Prompt.ask(f"[bold yellow]👉 是否自动过滤空数据样本 [y/n] (当前: {clean_status})[/]",
                                   choices=["y", "n"], default="y" if curr_clean else "n")
            config['dataset']['clean_data'] = (new_clean == "y")

            # 4. LLM 重试次数
            curr_retry = config['judge'].get('retry_times', 3)
            new_retry = Prompt.ask(f"[bold yellow]👉 API 请求失败重试次数 (当前: {curr_retry})[/]",
                                   default=str(curr_retry)).strip()
            if new_retry.isdigit():
                config['judge']['retry_times'] = int(new_retry)

            save_config(config, project_root)
            console.print("\n[bold green]✅ 所有运行参数已同步并保存！[/]")
            Prompt.ask("\n[bold yellow]按 [回车键] 返回...[/]")

        elif choice == '9':
            run_health_check()

        elif choice == '/help':
            show_help()

        elif choice in ['0', 'quit', 'exit']:
            clear_console()
            console.print("[bold green]👋 感谢使用 UX Evaluator，再见！\n[/]")
            break

        else:
            console.print("\n[bold red]⚠️ 无效的输入，请重新选择。[/]")
            Prompt.ask("[bold yellow]按 [回车键] 继续...[/]")


if __name__ == "__main__":
    main()