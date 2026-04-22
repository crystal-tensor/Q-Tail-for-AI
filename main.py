import os
import yaml
import numpy as np

# Import all agents
import agents.quantum_source_agent as source_agent
import agents.semantic_mapper_agent as mapper_agent
import agents.quantum_scheduler_agent as scheduler_agent
import agents.training_agent as training_agent

def load_config(path="config/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("="*60)
    print(" Q-TAIL-MVP: Orchestrator Pipeline")
    print("="*60)
    
    config = load_config()
    
    # 1. Quantum Source
    print("\n[1/4] Initializing Quantum Source...")
    source_agent.load_quantum_prior("data")
    s_prior = source_agent.get_default_prior()
    
    # 2. Semantic Mapper
    print("\n[2/4] Initializing Semantic Mapper...")
    taxonomy = mapper_agent.build_mt10_tail_taxonomy()
    tail_scores = mapper_agent.get_tail_scores()
    
    # 3. Quantum Scheduler & Training Loop
    print("\n[3/4] Running Scheduling & Training...")
    strategies = config.get("scheduler", {}).get("strategies", ["uniform", "empirical", "invfreq", "pt-rank"])
    eta = config.get("scheduler", {}).get("eta", 0.6)
    n_seeds = config.get("training", {}).get("n_seeds", 3)
    
    all_results = {}
    
    for strategy in strategies:
        # Get base prior
        # Note: pt-rank uses uniform as its base prior for comparison here
        base_mode = strategy if strategy in ["uniform", "empirical", "invfreq"] else "uniform"
        b_prior = mapper_agent.get_base_prior(base_mode)
        
        # Build task sampling distribution
        q_dist = scheduler_agent.build_scheduler(strategy, s_prior, b_prior, tail_scores, eta)
        print(f"  -> Built {strategy} sampler: {np.round(q_dist, 3)}")
        
        # Run training loop across seeds
        strategy_results = []
        for seed in range(n_seeds):
            res = training_agent.run_experiment(strategy, q_dist, seed=seed)
            strategy_results.append(res)
            
        all_results[strategy] = strategy_results

    # 4. Save results
    print("\n[4/4] Saving Final Results...")
    training_agent.save_training_logs("results", all_results)
    
    print("\n[Pipeline Complete] Ready for Evaluation Agent!")

if __name__ == "__main__":
    main()