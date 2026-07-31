import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

new_component = """
        const QuantumAILoop = () => (
            <section className="py-24 relative overflow-hidden bg-[#020617] border-y border-white/5">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-qcyan/5 rounded-full blur-[100px] pointer-events-none"></div>
                
                <div className="max-w-7xl mx-auto px-6 relative z-10">
                    <div className="text-center mb-20">
                        <span className="inline-block px-4 py-1 rounded-full bg-qcyan/10 border border-qcyan/30 text-qcyan text-sm font-bold tracking-widest mb-4 shadow-[0_0_15px_rgba(69,243,255,0.2)]">
                            BUSINESS MODEL / 商业模式演进
                        </span>
                        <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-6">
                            从单一算法到 <span className="text-transparent bg-clip-text bg-gradient-to-r from-qcyan to-emerald-400">量子-AI 闭环数据引擎</span>
                        </h2>
                        <p className="text-slate-400 text-lg max-w-3xl mx-auto font-light">
                            “量子随机电路采样产生可控分布，AI Agent 负责校准、解释、调参和数据服务化，最终形成量子-AI闭环数据引擎。”
                        </p>
                    </div>

                    <div className="relative max-w-5xl mx-auto mt-12">
                        {/* Connecting infinite loop line */}
                        <div className="hidden md:block absolute top-1/2 left-1/4 right-1/4 h-64 border-y-2 border-dashed border-white/10 rounded-[100px] -translate-y-1/2 animate-[spin_30s_linear_infinite] opacity-30"></div>
                        
                        <div className="flex flex-col md:flex-row items-center justify-between gap-8 relative z-10">
                            
                            {/* Quantum Side */}
                            <div className="glass-card w-full md:w-5/12 rounded-3xl p-8 border-2 border-qcyan/30 bg-black/60 backdrop-blur-xl relative group">
                                <div className="absolute inset-0 bg-gradient-to-br from-qcyan/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-3xl"></div>
                                <div className="text-center mb-6">
                                    <div className="w-20 h-20 mx-auto rounded-full bg-qcyan/10 flex items-center justify-center mb-4 border border-qcyan/30 shadow-[0_0_30px_rgba(69,243,255,0.2)]">
                                        <IconAtom className="w-10 h-10 text-qcyan" />
                                    </div>
                                    <h3 className="text-2xl font-bold text-white mb-2">量子引擎层 (Quantum)</h3>
                                    <div className="text-qcyan text-sm font-mono">稳定生成可控 PT 分布</div>
                                </div>
                                <ul className="space-y-3 text-slate-300 text-sm">
                                    <li className="flex items-start gap-2">
                                        <span className="text-qcyan mt-0.5">▹</span> 真实 Quafu 硬件采样，提取物理随机性
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-qcyan mt-0.5">▹</span> 提供具备极端长尾特性的高维数学先验
                                    </li>
                                </ul>
                            </div>

                            {/* Center Flow Arrows */}
                            <div className="flex md:flex-col items-center justify-center gap-4 py-8 md:py-0 w-full md:w-2/12">
                                <div className="flex flex-col items-center">
                                    <div className="text-xs text-slate-400 mb-2 font-mono">数据分发</div>
                                    <svg className="w-8 h-8 text-qcyan animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                    </svg>
                                </div>
                                <div className="flex flex-col items-center mt-4">
                                    <svg className="w-8 h-8 text-emerald-400 animate-pulse delay-75" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16l-4-4m0 0l4-4m-4 4h18" />
                                    </svg>
                                    <div className="text-xs text-slate-400 mt-2 font-mono">反馈寻优</div>
                                </div>
                            </div>

                            {/* AI Side */}
                            <div className="glass-card w-full md:w-5/12 rounded-3xl p-8 border-2 border-emerald-500/30 bg-black/60 backdrop-blur-xl relative group">
                                <div className="absolute inset-0 bg-gradient-to-bl from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-3xl"></div>
                                <div className="text-center mb-6">
                                    <div className="w-20 h-20 mx-auto rounded-full bg-emerald-500/10 flex items-center justify-center mb-4 border border-emerald-500/30 shadow-[0_0_30px_rgba(16,185,129,0.2)]">
                                        <svg className="w-10 h-10 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                                    </div>
                                    <h3 className="text-2xl font-bold text-white mb-2">AI 智能体层 (Agent)</h3>
                                    <div className="text-emerald-400 text-sm font-mono">负责校准、解释与服务化</div>
                                </div>
                                <ul className="space-y-3 text-slate-300 text-sm">
                                    <li className="flex items-start gap-2">
                                        <span className="text-emerald-400 mt-0.5">▹</span> 接入机器人、传感器与多模态校准场景
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-emerald-400 mt-0.5">▹</span> 自动调参 (Auto-Tune η) 与最优传输 (OT) 对齐
                                    </li>
                                </ul>
                            </div>

                        </div>
                        
                        {/* Bottom Output */}
                        <div className="mt-12 text-center relative z-10">
                            <div className="inline-flex flex-col items-center justify-center p-6 glass-card rounded-2xl border border-white/10 bg-gradient-to-b from-white/5 to-transparent min-w-[300px]">
                                <div className="text-sm text-slate-400 mb-2">最终形态交付</div>
                                <div className="text-xl font-bold text-white">规模化输出高质量、多样、可控数据集</div>
                                <div className="mt-3 flex gap-2">
                                    <span className="px-2 py-1 bg-white/5 rounded text-xs text-slate-300 border border-white/10">持续服务</span>
                                    <span className="px-2 py-1 bg-white/5 rounded text-xs text-slate-300 border border-white/10">商业变现</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        );

"""

content = content.replace("        const DataEngineArchitecture", new_component + "        const DataEngineArchitecture")

if "<QuantumAILoop />" not in content:
    content = content.replace("<DataEngineArchitecture />", "<QuantumAILoop />\n                        <DataEngineArchitecture />")

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("QuantumAILoop component added successfully.")