import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

class QuantumSourceAgent:
    """
    Quantum Source Agent: Reads quantum CSV data, normalizes it, calculates statistics, 
    and provides a default prior distribution for the scheduler.
    """
    def __init__(self):
        self.priors: Dict[str, np.ndarray] = {}
        self.summary: Dict[str, Any] = {}
        self.default_prior_name: str = None
        self.default_prior: np.ndarray = None

    def load_quantum_prior(self, data_dir: str) -> Dict[str, Any]:
        """
        1. 递归扫描 data/ 下所有 CSV 文件
        2. 自动识别概率列或计数列
        3. 统一归一化为概率向量 p
        4. 计算基础统计量
        5. 输出一个标准 source prior 文件
        6. 选择一份“默认量子源分布”供后续 scheduler 使用
        """
        csv_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                
                # 自动识别概率列或计数列
                target_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if any(k in col_lower for k in ["prob", "count", "freq", "rate", "val"]):
                        if pd.api.types.is_numeric_dtype(df[col]):
                            target_col = col
                            break
                
                # Fallback: 找第一个数值列且不是第一列(通常第一列是ID/State)
                if target_col is None:
                    for col in df.columns[1:]:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            target_col = col
                            break
                            
                # Fallback again: any numeric column
                if target_col is None:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            target_col = col
                            break

                if target_col is None:
                    print(f"[QuantumSourceAgent] Warning: Could not identify numeric column in {file}")
                    continue

                raw_values = df[target_col].dropna().values
                raw_values = raw_values[raw_values >= 0]
                
                if len(raw_values) == 0 or raw_values.sum() == 0:
                    continue

                # 3. 统一归一化为概率向量 p
                p = raw_values / raw_values.sum()
                
                # 4. 计算基础统计量
                support_size = int(np.sum(p > 0))
                p_sum = float(np.sum(p))
                
                p_safe = p[p > 0]
                entropy = float(-np.sum(p_safe * np.log(p_safe)))
                
                mean_p = np.mean(p)
                std_p = np.std(p)
                cv = float(std_p / mean_p) if mean_p > 0 else 0.0
                
                p_sorted = np.sort(p)[::-1]
                top_1_mass = float(p_sorted[0]) if len(p_sorted) > 0 else 0.0
                top_5_mass = float(np.sum(p_sorted[:5])) if len(p_sorted) >= 5 else float(np.sum(p_sorted))
                
                # Gini coefficient
                n = len(p_sorted)
                if n > 0:
                    index = np.arange(1, n + 1)
                    gini = float((np.sum((2 * index - n - 1) * p_sorted)) / (n * np.sum(p_sorted)))
                else:
                    gini = 0.0

                name = os.path.basename(file)
                self.priors[name] = p
                self.summary[name] = {
                    "support_size": support_size,
                    "sum": p_sum,
                    "entropy": entropy,
                    "cv": cv,
                    "top_1_mass": top_1_mass,
                    "top_5_mass": top_5_mass,
                    "gini": gini,
                    "file_path": file
                }
                print(f"[QuantumSourceAgent] Processed {name}: support={support_size}, entropy={entropy:.4f}")
                
            except Exception as e:
                print(f"[QuantumSourceAgent] Error processing {file}: {e}")

        # 6. 选择一份“默认量子源分布”
        best_name = None
        best_score = -1
        for name, stats in self.summary.items():
            if name == "default_prior": continue
            # 综合考虑熵和支持集大小，寻找探索空间大且稳定的长尾分布
            score = stats["entropy"] * np.log1p(stats["support_size"])
            if score > best_score:
                best_score = score
                best_name = name
                
        if best_name:
            self.default_prior_name = best_name
            self.default_prior = self.priors[best_name]
            self.summary["default_prior"] = {
                "name": best_name,
                "reason": "Selected based on highest combined score of entropy and support size, indicating a rich and stable heavy-tail distribution suitable for broad exploration."
            }
            print(f"[QuantumSourceAgent] Selected default prior: {best_name}")

        # 5. 输出标准 source prior 文件
        os.makedirs("results", exist_ok=True)
        summary_path = "results/quantum_source_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=4)
        print(f"[QuantumSourceAgent] Summary saved to {summary_path}")
            
        # 7. 生成页面展示所需的数据和图表
        if best_name:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 画图：概率分布排序图
            p_sorted = np.sort(self.default_prior)[::-1]
            plt.figure(figsize=(10, 6))
            plt.plot(p_sorted[:1000], color='#8B5CF6', linewidth=2) # 只画前1000个点展示重尾
            plt.fill_between(range(len(p_sorted[:1000])), p_sorted[:1000], color='#8B5CF6', alpha=0.3)
            plt.yscale('log')
            plt.title('Quantum Source Prior (Sorted Top 1000 States)', color='white')
            plt.xlabel('State Rank', color='white')
            plt.ylabel('Probability (Log Scale)', color='white')
            
            # 设置暗色背景适配网页
            ax = plt.gca()
            ax.set_facecolor('#050508')
            plt.gcf().patch.set_facecolor('#050508')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            
            plt.tight_layout()
            plt.savefig('results/fig_quantum_prob_dist.png', dpi=300, transparent=True)
            plt.close()
            
            # 生成页面数据
            best_stats = self.summary[best_name]
            page_data = {
                "title": "量子源数据：真实物理系统的非均匀先验",
                "subtitle": "Quantum Random Circuit Sampling",
                "source_file": best_name,
                "support_size": best_stats["support_size"],
                "entropy": best_stats["entropy"],
                "cv": best_stats["cv"],
                "gini": best_stats["gini"],
                "short_description": "基于真实超导量子芯片的随机电路采样数据，呈现出典型的重尾（Heavy-tail）非均匀分布特性，为具身智能训练提供结构化探索空间。",
                "chart_paths": [
                    "results/fig_quantum_prob_dist.png"
                ]
            }
            page_json_path = "results/page_quantum_source.json"
            with open(page_json_path, "w") as f:
                json.dump(page_data, f, indent=4)
            print(f"[QuantumSourceAgent] Page summary saved to {page_json_path}")

        return self.summary

    def get_default_prior(self) -> np.ndarray:
        """返回选定的默认量子先验概率向量"""
        if self.default_prior is None:
            raise ValueError("No default prior found. Call load_quantum_prior first.")
        return self.default_prior

# 便捷的模块级接口
_agent_instance = QuantumSourceAgent()

def load_quantum_prior(data_dir: str) -> Dict[str, Any]:
    return _agent_instance.load_quantum_prior(data_dir)

def get_default_prior() -> np.ndarray:
    return _agent_instance.get_default_prior()

if __name__ == "__main__":
    # Test execution
    summary = load_quantum_prior("data")
    prior = get_default_prior()
    print(f"Default prior shape: {prior.shape}, sum: {prior.sum()}")
