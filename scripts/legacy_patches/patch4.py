import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

ot_component = """
        const OTExtensions = () => {
            return (
                <section id="ot-extensions" className="py-24 relative bg-[#050508] border-t border-white/5">
                    <div className="max-w-7xl mx-auto px-6">
                        <div className="text-center mb-16">
                            <h2 className="text-4xl font-extrabold text-white mb-4">平台扩展：OT 映射实证</h2>
                            <p className="text-slate-400 max-w-2xl mx-auto">
                                补充论文核心要求：通过单调最优传输 (Monotone Optimal Transport)，实现 
                                <span className="text-rose-400">风险场景</span> 与 
                                <span className="text-blue-400">探索噪声</span> 的高保真分布映射。
                            </p>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            {/* Mapping II: Risk-Scene */}
                            <div className="glass-card rounded-3xl p-8 border-t-4 border-t-rose-500">
                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/10 text-rose-400 text-xs font-mono mb-6">
                                    Mapping II
                                </div>
                                <h3 className="text-2xl font-bold text-white mb-4">风险场景生成 (Risk-Scene)</h3>
                                <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                                    利用 $T(y) = G^{-1}(F_{PT}(y))$ 将量子先验无损转换为风险分布。评估指标：与目标真实混合 Beta 风险分布的 Wasserstein-1 距离。
                                </p>
                                
                                <div className="bg-black/50 rounded-xl p-4 mb-6 border border-white/5">
                                    <div className="text-xs text-slate-500 mb-2 font-mono">Wasserstein-1 Distance (越低越好)</div>
                                    <div className="space-y-3">
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm text-slate-300">Uniform</span>
                                            <span className="font-mono text-sm text-slate-400">0.1773</span>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm text-slate-300">Linear Scaling</span>
                                            <span className="font-mono text-sm text-slate-400">0.2106</span>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm text-slate-300">Gaussian Baseline</span>
                                            <span className="font-mono text-sm text-slate-400">0.0289</span>
                                        </div>
                                        <div className="flex items-center justify-between bg-rose-500/10 px-2 py-1 rounded border border-rose-500/30">
                                            <span className="text-sm font-bold text-rose-400">PT-OT (Ours)</span>
                                            <span className="font-mono font-bold text-rose-400">0.0020</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="rounded-xl overflow-hidden border border-white/10 relative group h-48 flex items-center justify-center bg-black/50">
                                    <img src="results/fig_risk_wasserstein.png" alt="Risk Scene Distribution" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                                </div>
                            </div>

                            {/* Mapping III: Exploration-Noise */}
                            <div className="glass-card rounded-3xl p-8 border-t-4 border-t-blue-500">
                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs font-mono mb-6">
                                    Mapping III
                                </div>
                                <h3 className="text-2xl font-bold text-white mb-4">探索噪声生成 (Exploration-Noise)</h3>
                                <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                                    通过构造具有“罕见大跳跃” (Rare-large-jump) 特性的噪声分布，打破策略在局部极值点的停滞，极大提升最佳臂发现率 (Best-arm discovery)。
                                </p>

                                <div className="bg-black/50 rounded-xl p-4 mb-6 border border-white/5">
                                    <div className="text-xs text-slate-500 mb-2 font-mono">Best-Arm Discovery Rate (越高越好)</div>
                                    <div className="space-y-3">
                                        <div>
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="text-slate-300">Gaussian Baseline</span>
                                                <span className="font-mono">0.00%</span>
                                            </div>
                                            <div className="w-full bg-white/5 rounded h-1"><div className="bg-slate-500 h-full rounded" style={{width: '0%'}}></div></div>
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="text-slate-300">Uniform</span>
                                                <span className="font-mono">0.00%</span>
                                            </div>
                                            <div className="w-full bg-white/5 rounded h-1"><div className="bg-slate-500 h-full rounded" style={{width: '0%'}}></div></div>
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="text-blue-400 font-bold">PT-OT (Ours)</span>
                                                <span className="font-mono text-blue-400 font-bold">95.80%</span>
                                            </div>
                                            <div className="w-full bg-white/5 rounded h-1"><div className="bg-blue-500 h-full rounded shadow-[0_0_10px_rgba(59,130,246,0.5)]" style={{width: '95.8%'}}></div></div>
                                        </div>
                                    </div>
                                </div>

                                <div className="rounded-xl overflow-hidden border border-white/10 relative group h-48 flex items-center justify-center bg-black/50">
                                    <img src="results/fig_exploration_discovery.png" alt="Exploration Noise Discovery" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            );
        };
"""

if "<OTExtensions />" not in content:
    content = content.replace("const ResultsPlaceholder", ot_component + "\n        const ResultsPlaceholder")
    content = content.replace("<ResultsPlaceholder />", "<OTExtensions />\n                        <ResultsPlaceholder />")

if "OT扩展" not in content:
    nav_link = '<a href="#ot-extensions" className="hover:text-qcyan transition-colors">OT扩展</a>\n                        <a href="#results"'
    content = content.replace('<a href="#results"', nav_link)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)
