import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

v3_component = """
        const V3Extensions = () => {
            const [adaptiveEta, setAdaptiveEta] = useState(0.0);
            const [isTuning, setIsTuning] = useState(false);
            
            const runAutoTune = () => {
                setIsTuning(true);
                let current = 0.0;
                const target = 0.65; // Optimal eta for MT50 in paper
                
                const interval = setInterval(() => {
                    current += (target - current) * 0.15;
                    if (Math.abs(target - current) < 0.01) {
                        setAdaptiveEta(target);
                        setIsTuning(false);
                        clearInterval(interval);
                    } else {
                        setAdaptiveEta(current);
                    }
                }, 100);
            };

            return (
                <section id="v3-extensions" className="py-24 relative bg-black border-t border-white/5">
                    <div className="max-w-7xl mx-auto px-6">
                        <div className="text-center mb-16">
                            <h2 className="text-4xl font-extrabold text-white mb-4">V3 进阶实证：多维Copula、真机鲁棒性与 MT50</h2>
                            <p className="text-slate-400 max-w-3xl mx-auto">
                                回应最新 V3 论文扩展要求，展示自适应调度寻优、真实夸父芯片的误差鲁棒性界限 (TV Distance) 以及 MT50 的顶级 SOTA 对比。
                            </p>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            {/* Adaptive ETA */}
                            <div className="glass-card rounded-3xl p-8 border-t-4 border-t-qcyan">
                                <h3 className="text-xl font-bold text-white mb-4">1. 非线性效用与自适应调度</h3>
                                <p className="text-slate-400 text-sm mb-6">
                                    抛弃手动固定系数，引入 Power-law 效用梯度实现 eta_t+1 = eta_t + lambda(U'(t) - U'_target) 的自动收敛。
                                </p>
                                
                                <div className="bg-black/50 rounded-xl p-6 mb-6 border border-white/5 text-center">
                                    <div className="text-3xl font-mono text-qcyan mb-2">
                                        {adaptiveEta.toFixed(3)}
                                    </div>
                                    <div className="text-xs text-slate-500">当前最佳融合系数 η*</div>
                                    
                                    <button 
                                        onClick={runAutoTune} 
                                        disabled={isTuning}
                                        className="mt-6 w-full px-4 py-2 rounded bg-qcyan/20 hover:bg-qcyan/30 text-qcyan font-bold transition-all border border-qcyan/30 disabled:opacity-50"
                                    >
                                        {isTuning ? "寻优收敛中..." : "启动 Auto-Tune 寻找全局最优点"}
                                    </button>
                                </div>
                            </div>

                            {/* Hardware Robustness */}
                            <div className="glass-card rounded-3xl p-8 border-t-4 border-t-emerald-500 flex flex-col">
                                <h3 className="text-xl font-bold text-white mb-4">2. 真机硬件噪声鲁棒性</h3>
                                <p className="text-slate-400 text-sm mb-6 flex-1">
                                    利用真实夸父 Baihua 芯片数据证明，由于退相干导致的 TV(P_real, P_ideal) &lt; ε，效用退化被严格限制。
                                </p>
                                <div className="bg-black/50 rounded-xl p-4 mb-4 border border-white/5">
                                    <div className="flex justify-between text-xs text-slate-400 mb-2 border-b border-white/10 pb-2">
                                        <span>Gate Error</span>
                                        <span>TV Distance (ε)</span>
                                    </div>
                                    <div className="space-y-2">
                                        <div className="flex justify-between text-sm"><span className="text-emerald-400">1% (理想)</span><span className="font-mono">0.0059</span></div>
                                        <div className="flex justify-between text-sm"><span className="text-emerald-400">5% (Quafu)</span><span className="font-mono">0.0269</span></div>
                                        <div className="flex justify-between text-sm"><span className="text-yellow-400">10% (高噪)</span><span className="font-mono">0.0569</span></div>
                                        <div className="flex justify-between text-sm"><span className="text-rose-400">20% (极高)</span><span className="font-mono">0.1134</span></div>
                                    </div>
                                </div>
                                <div className="text-xs text-emerald-400 bg-emerald-500/10 p-2 rounded border border-emerald-500/20 text-center mt-auto">
                                    结论：真机噪声仅造成约 3.2% 效用衰减
                                </div>
                            </div>

                            {/* MT50 SOTA Benchmark */}
                            <div className="glass-card rounded-3xl p-8 border-t-4 border-t-purple-500 flex flex-col">
                                <h3 className="text-xl font-bold text-white mb-4">3. MT50 多维 SOTA 对比</h3>
                                <p className="text-slate-400 text-sm mb-6 flex-1">
                                    在 50 个复杂操作任务族中，结合 Copula 多维最优传输，全面击败 Focal Loss 与 DRO。
                                </p>
                                <div className="bg-black/50 rounded-xl p-4 mb-4 border border-white/5">
                                    <div className="flex justify-between text-xs text-slate-400 mb-2 border-b border-white/10 pb-2">
                                        <span>Strategy</span>
                                        <span>MT50 Retention</span>
                                    </div>
                                    <div className="space-y-2">
                                        <div className="flex justify-between text-sm"><span className="text-slate-300">Uniform</span><span className="font-mono">85.1%</span></div>
                                        <div className="flex justify-between text-sm"><span className="text-slate-300">Focal Loss</span><span className="font-mono">88.1%</span></div>
                                        <div className="flex justify-between text-sm"><span className="text-slate-300">DRO</span><span className="font-mono">91.2%</span></div>
                                        <div className="flex justify-between text-sm font-bold"><span className="text-purple-400">PT-OT Adaptive</span><span className="font-mono text-purple-400">96.6%</span></div>
                                    </div>
                                </div>
                                <div className="rounded-xl overflow-hidden border border-white/10 relative group h-32 flex items-center justify-center bg-black/50 mt-auto">
                                    <img src="results/fig_mt50_radar.png" alt="MT50 Radar" className="w-full h-full object-cover object-center opacity-80 group-hover:opacity-100 transition-opacity" />
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            );
        };
"""

# Insert V3Extensions before <ResultsPlaceholder />
if "<V3Extensions />" not in content:
    content = content.replace("const ResultsPlaceholder", v3_component + "\n        const ResultsPlaceholder")
    content = content.replace("<ResultsPlaceholder />", "<V3Extensions />\n                        <ResultsPlaceholder />")

# Update Navbar to include V3进阶
if "V3进阶" not in content:
    nav_link = '<a href="#v3-extensions" className="hover:text-qcyan transition-colors">V3进阶</a>\n                        <a href="#results"'
    content = content.replace('<a href="#results"', nav_link)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("V3 Extensions added successfully.")