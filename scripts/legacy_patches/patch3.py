import re

with open('qtail-mvp-presentation.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Instead of regex, let's just find the indices
start_idx = content.find('const MVPTarget = () => (')
end_idx = content.find('const ResultsPlaceholder = () => (')

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + """const InteractiveMVPTarget = () => {
            const [step, setStep] = useState(1);
            const [qState, setQState] = useState(0); 
            const [semanticState, setSemanticState] = useState(0); 
            const [eta, setEta] = useState(0.5);
            const [trainState, setTrainState] = useState(0); 
            
            const runQuantumProcess = () => {
                setQState(1);
                setTimeout(() => setQState(2), 1500);
                setTimeout(() => setQState(3), 3000);
                setTimeout(() => setQState(4), 4500);
            };

            const runSemanticProcess = () => {
                setSemanticState(1);
            };
            
            const runTrainProcess = () => {
                setTrainState(1);
                setTimeout(() => setTrainState(2), 2000);
            };

            return (
                <section id="mvp" className="py-24 relative">
                    <div className="max-w-5xl mx-auto px-6">
                        <div className="text-center mb-12">
                            <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-4">当前 MVP 交互全流程</h2>
                            <p className="text-slate-400">体验从量子采样到具身智能训练的完整链路</p>
                        </div>

                        <div className="flex flex-wrap gap-2 justify-center mb-12">
                            {[1, 2, 3].map(i => (
                                <button 
                                    key={i} 
                                    onClick={() => setStep(i)}
                                    className={`px-6 py-3 rounded-full font-bold transition-all ${step === i ? 'bg-qcyan text-black' : 'glass-card text-white hover:border-qcyan/50'}`}
                                >
                                    {i === 1 ? "1. 量子源先验生成" : i === 2 ? "2. 任务语义映射" : "3. Head/Tail 调度训练"}
                                </button>
                            ))}
                        </div>

                        {step === 1 && (
                            <div className="glass-card rounded-3xl p-8 border border-qcyan/30 animate-fade-in">
                                <h3 className="text-2xl font-bold text-white mb-6">1. 量子源数据流</h3>
                                <div className="flex flex-col md:flex-row gap-6 items-center">
                                    <div className="flex-1 w-full space-y-4">
                                        <button onClick={runQuantumProcess} disabled={qState > 0} className="w-full px-4 py-3 rounded bg-blue-600 hover:bg-blue-500 text-white font-bold flex items-center justify-center gap-2 disabled:opacity-50 transition-colors">
                                            <IconPlay /> 启动全自动数据流
                                        </button>
                                        <div className={`p-4 rounded border ${qState >= 1 ? 'border-qcyan bg-qcyan/10' : 'border-white/10 bg-white/5'} transition-all`}>
                                            <div className="font-bold text-white mb-1 flex items-center gap-2"><IconCpu className="w-4 h-4"/> A. 真机随机电路采样</div>
                                            <p className="text-xs text-slate-400">{qState >= 1 ? '已提交任务到 Quafu 集群...' : '等待执行'}</p>
                                        </div>
                                        <div className={`p-4 rounded border ${qState >= 2 ? 'border-qcyan bg-qcyan/10' : 'border-white/10 bg-white/5'} transition-all`}>
                                            <div className="font-bold text-white mb-1 flex items-center gap-2"><IconDownload className="w-4 h-4"/> B. 下载并导入 CSV</div>
                                            <p className="text-xs text-slate-400">{qState >= 2 ? '成功导入测量比特串' : '等待生成'}</p>
                                        </div>
                                        <div className={`p-4 rounded border ${qState >= 3 ? 'border-qcyan bg-qcyan/10' : 'border-white/10 bg-white/5'} transition-all`}>
                                            <div className="font-bold text-white mb-1 flex items-center gap-2"><IconCheck className="w-4 h-4"/> C. PT 分布物理验证与先验构建</div>
                                            <p className="text-xs text-slate-400">{qState >= 3 ? '散度检验通过，呈现完美重尾，构建先验Ps完成' : '等待校验'}</p>
                                        </div>
                                    </div>
                                    <div className="flex-1 w-full bg-black/50 border border-white/10 rounded-xl p-6 min-h-[300px] flex flex-col items-center justify-center relative overflow-hidden">
                                        {qState === 0 && <div className="text-slate-500 text-sm">点击左侧启动数据流</div>}
                                        {qState === 1 && <div className="animate-pulse text-qcyan flex flex-col items-center"><IconCpu className="w-12 h-12 mb-2"/><span className="mt-2">量子硬件运行中...</span></div>}
                                        {qState === 2 && <div className="text-emerald-400 flex flex-col items-center"><IconDatabase className="w-12 h-12 mb-2"/><span className="font-mono text-xs mt-2 text-left bg-black/50 p-2 rounded">bitstring, count<br/>00000, 120<br/>00001, 15<br/>...</span></div>}
                                        {qState >= 3 && (
                                            <div className="w-full h-full flex flex-col items-center justify-center animate-fade-in">
                                                <div className="w-full h-32 flex items-end justify-between px-2 gap-1 opacity-80 border-b border-l border-white/20 pb-1">
                                                    {[...Array(20)].map((_, i) => (
                                                        <div key={i} className="bg-gradient-to-t from-qcyan to-qpurple w-full rounded-t" style={{height: `${Math.max(5, 100 * Math.exp(-i/5))}%`}}></div>
                                                    ))}
                                                </div>
                                                <div className="mt-6 text-qcyan font-mono text-sm bg-qcyan/10 px-3 py-1 rounded border border-qcyan/30">✓ 先验向量 Ps 构建完成</div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}

                        {step === 2 && (
                            <div className="glass-card rounded-3xl p-8 border border-qpurple/30 animate-fade-in">
                                <h3 className="text-2xl font-bold text-white mb-6">2. 任务语义映射器</h3>
                                <p className="text-slate-400 text-sm mb-6">将环境中的物理任务，根据其语义难度与量子重尾分布的概率质量进行一一映射。</p>
                                <div className="flex flex-col md:flex-row gap-6">
                                    <div className="flex-1 w-full">
                                        <div className="bg-black/50 rounded-xl p-4 border border-white/10 mb-4">
                                            <div className="font-bold text-white mb-4">Meta-World MT10 任务池</div>
                                            <div className="flex flex-wrap gap-2">
                                                {['reach', 'push', 'pick-place', 'door-open', 'drawer-close', 'button-press', 'peg-insert-side', 'window-open', 'sweep', 'basketball'].map((t, i) => (
                                                    <span key={i} className="px-2 py-1 bg-white/5 border border-white/10 rounded text-xs text-slate-300">{t}</span>
                                                ))}
                                            </div>
                                        </div>
                                        <button onClick={runSemanticProcess} disabled={semanticState > 0} className="w-full px-4 py-3 rounded bg-qpurple hover:bg-qpurple/80 text-white font-bold transition-all flex justify-center items-center gap-2">
                                            <IconPlay /> 执行语义稀有度映射
                                        </button>
                                    </div>
                                    <div className="flex-1 w-full min-h-[300px]">
                                        {semanticState === 0 ? (
                                            <div className="h-full border border-dashed border-white/10 rounded-xl flex items-center justify-center text-slate-500 text-sm bg-black/30">
                                                等待执行映射...
                                            </div>
                                        ) : (
                                            <div className="bg-black/50 border border-qpurple/30 rounded-xl p-4 animate-fade-in h-full flex flex-col">
                                                <div className="text-xs text-slate-400 mb-4 flex justify-between px-2">
                                                    <span>最困难 (分配最高 Ps)</span>
                                                    <span>最简单 (分配最低 Ps)</span>
                                                </div>
                                                <div className="space-y-2 flex-1 flex flex-col justify-center">
                                                    <div className="flex justify-between items-center bg-rose-500/20 px-3 py-2 rounded border border-rose-500/30 text-xs">
                                                        <span className="text-rose-400 font-bold">1. peg-insert-side</span>
                                                        <span className="font-mono text-white">Ps: 0.28</span>
                                                    </div>
                                                    <div className="flex justify-between items-center bg-rose-400/20 px-3 py-2 rounded border border-rose-400/30 text-xs">
                                                        <span className="text-rose-300 font-bold">2. basketball</span>
                                                        <span className="font-mono text-white">Ps: 0.19</span>
                                                    </div>
                                                    <div className="flex justify-between items-center bg-white/5 px-3 py-2 rounded border border-white/10 text-xs text-slate-400">
                                                        <span>...</span>
                                                        <span>...</span>
                                                    </div>
                                                    <div className="flex justify-between items-center bg-emerald-500/10 px-3 py-2 rounded border border-emerald-500/20 text-xs">
                                                        <span className="text-emerald-400">10. reach</span>
                                                        <span className="font-mono text-white">Ps: 0.01</span>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}

                        {step === 3 && (
                            <div className="glass-card rounded-3xl p-8 border border-emerald-500/30 animate-fade-in">
                                <h3 className="text-2xl font-bold text-white mb-6">3. Head/Tail 调度训练</h3>
                                
                                <div className="mb-8 bg-black/50 rounded-xl p-6 border border-white/10">
                                    <div className="flex justify-between items-center mb-4">
                                        <span className="text-white font-bold">调节量子先验融合比例 (η)</span>
                                        <span className="font-mono text-qcyan bg-qcyan/10 px-3 py-1 rounded border border-qcyan/20">η = {eta.toFixed(2)}</span>
                                    </div>
                                    <input 
                                        type="range" min="0" max="1" step="0.1" 
                                        value={eta} onChange={(e) => setEta(parseFloat(e.target.value))}
                                        className="w-full accent-qcyan cursor-pointer"
                                    />
                                    <div className="flex justify-between text-xs text-slate-400 mt-2 font-mono">
                                        <span>0.0 (Uniform)</span>
                                        <span>1.0 (Quantum Only)</span>
                                    </div>
                                </div>

                                <div className="flex flex-col md:flex-row gap-6">
                                    <div className="flex-1 w-full flex flex-col justify-center">
                                        <button onClick={runTrainProcess} className="w-full px-4 py-3 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-bold transition-all flex items-center justify-center gap-2">
                                            {trainState === 1 ? <IconPlay className="animate-pulse"/> : <IconPlay />} 
                                            {trainState === 1 ? '模拟训练中...' : '运行训练与评估'}
                                        </button>
                                        {trainState === 2 && (
                                            <div className="mt-4 p-4 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm animate-fade-in">
                                                ✓ 训练完成！在 η={eta} 下，长尾任务的胜率得到了针对性强化。
                                            </div>
                                        )}
                                    </div>
                                    <div className="flex-1 w-full bg-black/50 rounded-xl p-6 border border-white/10 min-h-[200px]">
                                        <div className="text-sm font-bold text-white mb-6 flex items-center gap-2"><IconChart className="w-4 h-4 text-qcyan"/> 表现预测 (实时计算)</div>
                                        <div className="space-y-6">
                                            <div>
                                                <div className="flex justify-between text-xs mb-2">
                                                    <span className="text-slate-300">Head Success (头部任务)</span>
                                                    <span className="font-mono">{Math.round(90 - eta * 15)}%</span>
                                                </div>
                                                <div className="w-full bg-white/10 rounded h-2 overflow-hidden">
                                                    <div className="bg-blue-400 h-full rounded transition-all duration-500" style={{width: `${90 - eta * 15}%`}}></div>
                                                </div>
                                            </div>
                                            <div>
                                                <div className="flex justify-between text-xs mb-2">
                                                    <span className="text-slate-300">Tail Success (长尾任务)</span>
                                                    <span className="font-mono text-qcyan">{Math.round(20 + eta * 45)}%</span>
                                                </div>
                                                <div className="w-full bg-white/10 rounded h-2 overflow-hidden">
                                                    <div className="bg-qcyan h-full rounded transition-all duration-500" style={{width: `${20 + eta * 45}%`}}></div>
                                                </div>
                                            </div>
                                            <div>
                                                <div className="flex justify-between text-xs mb-2">
                                                    <span className="text-slate-300">CVaR@20 (最差表现)</span>
                                                    <span className="font-mono text-emerald-400">{Math.round(15 + eta * 40)}%</span>
                                                </div>
                                                <div className="w-full bg-white/10 rounded h-2 overflow-hidden">
                                                    <div className="bg-emerald-400 h-full rounded transition-all duration-500" style={{width: `${15 + eta * 40}%`}}></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </section>
            );
        };
""" + content[end_idx:]

    new_content = new_content.replace("<MVPTarget />", "<InteractiveMVPTarget />")
    
    with open('qtail-mvp-presentation.html', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Replaced MVPTarget with InteractiveMVPTarget successfully")
else:
    print(f"Failed to find indices. start: {start_idx}, end: {end_idx}")
