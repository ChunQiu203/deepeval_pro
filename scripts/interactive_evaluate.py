"""交互式配置与批量评测入口脚本"""
# 注：推荐使用终端运行python.exe interactive_evaluate.py，优于在IDE中直接点击运行
import os
import sys
import yaml
import json
import traceback
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

# 确保能引用到项目根目录的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def ensure_interactive_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """补齐交互脚本需要的默认字段，不影响原有 classic 配置。"""
    if "batch" not in config:
        config["batch"] = {"output_path": "results/evaluation_results.json"}

    batch = config["batch"]
    batch.setdefault("output_path", "results/evaluation_results.json")
    batch.setdefault("evaluation_mode", "classic")
    batch.setdefault("dag_config_path", "examples/dag_config.json")
    batch.setdefault("metrics_path", "examples/metrics.json")
    return config


def show_help():
    """显示帮助菜单与所有指标的具体说明（增加翻页逻辑，使用 Rich 优化排版）"""

    # ------------------ 第 1 页：操作说明 ------------------
    clear_console()

    help_text = """
本工具支持对大模型回复进行多维度的 UX（用户体验）量化评估。

[bold magenta]【🤖 支持的模型类型 (Supported Models)】[/]
  系统会自动根据输入的模型名称前缀匹配对应的 API 厂商：
  [cyan]- 通义千问 (Qwen)系列:[/] 如 'qwen-turbo', 'qwen-plus', 'qwen-max'
    [dim]* 需在项目根目录 .env 文件配置 QWEN_API_KEY[/]
  [cyan]- 深度求索 (DeepSeek)系列:[/] 如 'deepseek-chat', 'deepseek-reasoner'
    [dim]* 需在项目根目录 .env 文件配置 DEEPSEEK_API_KEY[/]
  [dim](如需扩展其他模型，请前往 metric_judge.py 中按需添加 API 路由)[/]

[bold magenta]【🛠️ 操作指南 (Operation Guide)】[/]
  [cyan][1] 运行评测:[/] 根据当前面板上的【模型】、【数据集】和【生效指标】立刻执行批量评测。
  [cyan][2] 修改模型:[/] 输入你要测试的目标模型名称，系统会自动切换对应的 API。
  [cyan][3] 修改数据集:[/] 更换你想测试的对话 JSON 文件路径。
  [cyan][4] 配置评价指标:[/] 进入指标设置面板，你可以便捷地开启/关闭指标，并自定义阈值。
    """

    console.print(Panel(help_text, title="[bold magenta]📖 UX Evaluator 帮助指南 - [第 1/2 页][/]", border_style="cyan"))
    Prompt.ask("\n[bold yellow]💡 提示：按 [回车键] 继续查看下一页【评价指标字典】...[/]")

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
            console.print(f"\n[bold green]✅ 配置已保存！当前生效指标数: {len(current_metrics)}[/]")
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


def _render_dag_summary(results: Dict[str, Any], output_path: str) -> None:
    summary = results.get("summary", {})
    console.print(f"\n[bold green]💾 DAG 评估完成！结果已成功保存至: {output_path}[/]")

    overview = Table(title="[bold magenta]🧭 多轮对话评测摘要[/]", box=box.SIMPLE, show_header=False)
    overview.add_column("项目", style="cyan")
    overview.add_column("值", justify="right", style="bold green")
    overview.add_row("会话数", str(summary.get("case_count", 0)))
    overview.add_row("轮次结果数", str(results.get("turn_result_count", 0)))
    overview.add_row("总平均分", f"{summary.get('overall_average', 0):.4f}")
    overview.add_row("DAG 总平均分", f"{summary.get('dag_overall_average', 0):.4f}")
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
    """执行 DAG 多轮会话评测逻辑，仅作为 classic 流程的补充分支。"""
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

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        _render_dag_summary(results, output_path)
    except Exception as e:
        console.print(f"\n[bold red]❌ 运行 DAG 评估过程中出现错误: {str(e)}[/]")
        traceback.print_exc()

    Prompt.ask("\n[bold yellow]测试结束，按 [回车键] 返回主菜单...[/]")


def run_batch_evaluation(config, project_root):
    """执行批量测试集的评测逻辑（加入 loading 和对齐优化）"""
    if config.get("batch", {}).get("evaluation_mode", "classic") == "dag":
        run_dag_evaluation(config, project_root)
        return

    clear_console()

    # 动态加载动画，提升长耗时任务体验
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
                console.print("\n[bold red]❌ 错误：当前未配置任何评价指标，请先在菜单 [4] 中开启需要的指标！[/]")
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

        # 结果保存
        output_path = _resolve_path(project_root, config["batch"]["output_path"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        console.print(f"\n[bold green]💾 评估完成！结果已成功保存至: {output_path}[/]")

        # 使用 Rich Table 完美解决中英文混合排版不对齐的问题
        if summary.get("metric_average"):
            res_table = Table(title="[bold magenta]📈 【平均得分统计 (Average Scores)】[/]", box=box.SIMPLE,
                              show_header=False)
            res_table.add_column("指标名称", style="cyan")
            res_table.add_column("得分", justify="right", style="bold green")

            # 建立英文名到中文名的反向映射字典
            name_to_zh = {spec["name"]: EN_TO_ZH_MAP.get(k, spec["name"]) for k, spec in METRIC_SPECS.items()}

            for name, avg in summary["metric_average"].items():
                zh_name = name_to_zh.get(name, name)
                display_name = f"{zh_name} ({name})"
                res_table.add_row(f"• {display_name}", f"{avg:.4f}")

            console.print(res_table)
        else:
            console.print("[dim]  (无有效指标得分)[/]")

        console.print(
            f"\n[bold magenta]🌟 总平均分 (Overall Score):[/] [bold green]{summary.get('overall_average', 0):.4f}[/]")

    except Exception as e:
        console.print(f"\n[bold red]❌ 运行评估过程中出现错误: {str(e)}[/]")
        traceback.print_exc()

    Prompt.ask("\n[bold yellow]测试结束，按 [回车键] 返回主菜单...[/]")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 初始化时加载默认配置
    try:
        config = load_default_config(project_root)
    except FileNotFoundError:
        console.print("[bold red]❌ 错误：找不到 configs/default_config.yaml 文件，请确保在项目根目录运行或文件存在。[/]")
        return

    ensure_interactive_defaults(config)

    # 主菜单循环
    while True:
        clear_console()

        # 构建主菜单面板内容
        current_metrics = config.get('metrics', [])
        metrics_display = ""
        for m in current_metrics:
            m_name = m.get('name', 'Unknown')
            metrics_display += f"    [dim]* {m_name} (阈值: {m.get('threshold', 0.5)})[/]\n"

        # 获取当前的输出路径（带默认值兜底）
        current_output_path = config['batch'].get('output_path', 'results/evaluation_results.json')
        evaluation_mode = config['batch'].get('evaluation_mode', 'classic')
        dag_config_path = config['batch'].get('dag_config_path', 'examples/dag_config.json')
        metrics_path = config['batch'].get('metrics_path', 'examples/metrics.json')

        dag_display = ""
        if evaluation_mode == "dag":
            dag_display = (
                f"  - [cyan]DAG 配置路径 (DAG Config)[/] : [bold green]{dag_config_path}[/]\n"
                f"  - [cyan]指标配置路径 (Metrics JSON)[/]: [bold green]{metrics_path}[/]\n"
            )

        menu_text = f"""
[bold magenta]【当前配置概览 (Current Configuration)】[/]
  - [cyan]评测模式 (Evaluation Mode)[/] : [bold green]{evaluation_mode}[/]
  - [cyan]评判模型 (Judge Model)[/]    : [bold green]{config['judge']['model']}[/]
  - [cyan]数据集路径 (Data Path)[/]    : [bold green]{config['dataset']['data_path']}[/]
  - [cyan]输出保存路径 (Output Path)[/]: [bold green]{current_output_path}[/]
{dag_display}  - [cyan]当前生效指标 (Metrics)[/]    : [bold green]{len(current_metrics)} 个[/]
{metrics_display}
[bold magenta]【操作菜单 (Main Menu)】[/]
  [bold green][1] ⚡ 运行当前测试集评测 (Run Batch Evaluation)[/]
  [cyan][2] 🔧 修改大模型配置 (Change Judge Model)[/]
  [cyan][3] 📂 修改数据集路径 (Change Dataset Path)[/]
  [cyan][4] 📊 配置评价指标 (Configure Evaluation Metrics)[/]
  [cyan][5] 📁 修改输出保存路径 (Change Output Path)[/]
  [cyan][6] 🕸️ 切换评测模式 (Classic / DAG)[/]
  [cyan][7] 🧭 修改 DAG 配置路径 (Change DAG Config Path)[/]
  [cyan][8] 🧾 修改 DAG 指标配置路径 (Change Metrics JSON Path)[/]
  [dim]\\[/help] 📖 查看帮助与指标说明 (Show Help & Info)[/]
  [dim][0] 🚪 退出程序 (Exit)[/]
"""
        # 使用 Panel 渲染主菜单，自动适应终端宽度
        console.print(Panel(
            menu_text.strip(),
            title="[bold magenta]🚀 UX Evaluator 交互式评测控制台 🚀[/]",
            border_style="cyan"
        ))

        choice = Prompt.ask("[bold yellow]👉 请选择操作指令[/]").strip().lower()

        if choice == '1':
            run_batch_evaluation(config, project_root)
        elif choice == '2':
            new_model = Prompt.ask(
                f"\n[bold yellow]👉 请输入新的评测模型名 (例: deepseek-chat) [[dim]当前: {config['judge']['model']}[/]][/]"
            ).strip()
            if new_model:
                config['judge']['model'] = new_model
                console.print("[bold green]✅ 模型配置已更新！[/]")
                Prompt.ask("[bold yellow]按 [回车键] 返回...[/]")
        elif choice == '3':
            new_path = Prompt.ask(
                f"\n[bold yellow]👉 请输入新的数据集相对路径 [[dim]当前: {config['dataset']['data_path']}[/]][/]"
            ).strip()
            if new_path:
                config['dataset']['data_path'] = new_path
                console.print("[bold green]✅ 数据集配置已更新！[/]")
                Prompt.ask("[bold yellow]按 [回车键] 返回...[/]")
        elif choice == '4':
            modify_metrics_menu(config)
        elif choice == '5':
            new_out_path = Prompt.ask(
                f"\n[bold yellow]👉 请输入新的结果保存路径 (例: results/my_test.json) [[dim]当前: {current_output_path}[/]][/]"
            ).strip()
            if new_out_path:
                config['batch']['output_path'] = new_out_path
                console.print("[bold green]✅ 输出保存路径已更新！[/]")
                Prompt.ask("[bold yellow]按 [回车键] 返回...[/]")
        elif choice == '6':
            next_mode = "dag" if evaluation_mode == "classic" else "classic"
            config['batch']['evaluation_mode'] = next_mode
            console.print(f"[bold green]✅ 评测模式已切换为: {next_mode}[/]")
            Prompt.ask("[bold yellow]按 [回车键] 返回...[/]")
        elif choice == '7':
            new_dag_path = Prompt.ask(
                f"\n[bold yellow]👉 请输入新的 DAG 配置路径 [[dim]当前: {dag_config_path}[/]][/]"
            ).strip()
            if new_dag_path:
                config['batch']['dag_config_path'] = new_dag_path
                console.print("[bold green]✅ DAG 配置路径已更新！[/]")
                Prompt.ask("[bold yellow]按 [回车键] 返回...[/]")
        elif choice == '8':
            new_metrics_path = Prompt.ask(
                f"\n[bold yellow]👉 请输入新的 DAG 指标配置路径 [[dim]当前: {metrics_path}[/]][/]"
            ).strip()
            if new_metrics_path:
                config['batch']['metrics_path'] = new_metrics_path
                console.print("[bold green]✅ DAG 指标配置路径已更新！[/]")
                Prompt.ask("[bold yellow]按 [回车键] 返回...[/]")
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
