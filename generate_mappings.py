import os
import json
import numpy as np
import pandas as pd
import sys

# Ensure agents can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.semantic_mapper_agent import SemanticMapperAgent
from agents.quantum_scheduler_agent import QuantumSchedulerAgent

def generate_all_mappings():
    data_dir = "data"
    out_dir = "results/mappings"
    os.makedirs(out_dir, exist_ok=True)
    
    semantic_agent = SemanticMapperAgent()
    scheduler_agent = QuantumSchedulerAgent()
    
    # Load tasks and base prior
    taxonomy = semantic_agent.build_mt10_tail_taxonomy()
    tasks = semantic_agent.mt10_tasks
    base_prior = semantic_agent.get_base_prior("empirical")
    tail_score = semantic_agent.get_tail_scores()
    eta = 0.5 # Default eta for demonstration
    
    generated_files = []
    
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".csv"):
            csv_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(csv_path)
                
                # Identify probability column
                target_col = None
                for col in df.columns:
                    if any(k in col.lower() for k in ["prob", "count", "freq", "rate", "val"]):
                        if pd.api.types.is_numeric_dtype(df[col]):
                            target_col = col
                            break
                if not target_col:
                    target_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
                raw_values = df[target_col].dropna().values
                raw_values = raw_values[raw_values >= 0]
                if len(raw_values) == 0 or raw_values.sum() == 0:
                    continue
                    
                p = raw_values / raw_values.sum()
                
                # Compute q using pt-rank (Rank-Based OT)
                q = scheduler_agent.build_scheduler(
                    strategy="pt-rank",
                    source_prior=p,
                    base_prior=base_prior,
                    tail_score=tail_score,
                    eta=eta
                )
                
                # Format output mapping JSON
                mapping_data = {
                    "source_csv": file,
                    "mapping_algorithm": "Rank-Based Optimal Transport (pt-rank)",
                    "quantum_fusion_ratio_eta": eta,
                    "total_quantum_states_sampled": len(p),
                    "task_mapping": []
                }
                
                for i, task in enumerate(tasks):
                    mapping_data["task_mapping"].append({
                        "task_id": i,
                        "task_name": task,
                        "semantic_tier": semantic_agent.taxonomy[task]["category"],
                        "tail_score_tau": float(tail_score[i]),
                        "base_prior_b": float(base_prior[i]),
                        "mapped_probability_Ps": float(q[i])
                    })
                    
                # Sort by mapped probability descending
                mapping_data["task_mapping"].sort(key=lambda x: x["mapped_probability_Ps"], reverse=True)
                
                out_filename = file.replace(".csv", "_ot_mapping.json")
                out_path = os.path.join(out_dir, out_filename)
                
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(mapping_data, f, indent=4, ensure_ascii=False)
                    
                generated_files.append({
                    "name": out_filename,
                    "url": out_path,
                    "source": file
                })
                print(f"Generated {out_path} for {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
    # Write an index file
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(generated_files, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully generated {len(generated_files)} JSON mappings.")

if __name__ == "__main__":
    generate_all_mappings()
