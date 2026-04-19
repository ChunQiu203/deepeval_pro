import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional


class TestCase:
    """标准化的测试用例结构，统一全项目的输入格式，兼容DeepEval框架"""
    def __init__(
        self,
        input: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            input: 用户的输入/问题
            actual_output: LLM的实际输出
            expected_output: 期望的输出（可选）
            context: 对话上下文（可选）
            retrieval_context: RAG检索到的上下文（可选）
            metadata: 额外的元数据（可选）
        """
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.context = context or []
        self.retrieval_context = retrieval_context or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，方便序列化"""
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """从字典创建测试用例"""
        return cls(
            input=data.get("input", ""),
            actual_output=data.get("actual_output", ""),
            expected_output=data.get("expected_output"),
            context=data.get("context"),
            retrieval_context=data.get("retrieval_context"),
            metadata=data.get("metadata")
        )


class DatasetLoader:
    """数据集加载器，支持CSV/JSON格式导入，自动清洗与标准化数据"""
    
    def __init__(self, clean_data: bool = True):
        """
        Args:
            clean_data: 是否自动清洗数据（过滤空样本、标准化字段）
        """
        self.clean_data = clean_data
    
    def load_from_file(self, file_path: str) -> List[TestCase]:
        """
        从文件加载数据集，自动识别文件格式
        
        Args:
            file_path: 数据文件路径，支持.csv/.json
        
        Returns:
            标准化后的测试用例列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return self.load_from_csv(file_path)
        elif ext == ".json":
            return self.load_from_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，目前仅支持CSV和JSON")
    
    def load_from_csv(self, file_path: str) -> List[TestCase]:
        """从CSV文件加载数据集"""
        df = pd.read_csv(file_path)
        # 转换为字典列表
        raw_data = df.to_dict("records")
        return self._process_raw_data(raw_data)
    
    def load_from_json(self, file_path: str) -> List[TestCase]:
        """从JSON文件加载数据集，支持单行JSON或JSON数组"""
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                # 尝试加载JSON数组
                raw_data = json.load(f)
                if not isinstance(raw_data, list):
                    raw_data = [raw_data]
            except json.JSONDecodeError:
                # 尝试加载JSONL（每行一个JSON）
                f.seek(0)
                raw_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        raw_data.append(json.loads(line))
        
        return self._process_raw_data(raw_data)
    
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[TestCase]:
        """处理原始数据，清洗并标准化为TestCase"""
        test_cases = []
        for item in raw_data:
            # 自动适配不同的字段名，兼容用户的自定义字段
            input_val = item.get("input", item.get("question", item.get("query", "")))
            output_val = item.get("actual_output", item.get("output", item.get("answer", "")))
            
            # 清洗数据：过滤空样本
            if self.clean_data:
                if not input_val or not output_val:
                    # 跳过空输入或空输出的样本
                    continue
            
            test_case = TestCase(
                input=input_val,
                actual_output=output_val,
                expected_output=item.get("expected_output"),
                context=item.get("context"),
                retrieval_context=item.get("retrieval_context"),
                metadata=item.get("metadata")
            )
            test_cases.append(test_case)
        
        return test_cases
