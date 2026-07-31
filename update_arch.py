import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract the old Architecture component
match = re.search(r'const Architecture = \(\) => \([\s\S]*?\n        \);[\s]*const InteractiveMVPTarget', content)
if not match:
    print("Could not find Architecture component")
    exit(1)

old_arch = match.group(0).replace('const InteractiveMVPTarget', '')

new_arch = """const DataEngineArchitecture = () => (
        <section id="architecture" className="py-24 relative overflow-hidden bg-[#020617] border-y border-white/5">
            {/* Background glows */}
            <div className="absolute top-0 left-1/4 w-96 h-96 bg-qcyan/10 rounded-full blur-[120px] pointer-events-none"></div>
            <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-[120px] pointer-events-none"></div>
            
            <div className="max-w-7xl mx-auto px-6 relative z-10">
                <div className="text-center mb-16">
                    <h2 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white via-qcyan to-emerald-400 mb-6">
                        量子驱动的具身智能数据生成引擎
                    </h2>
                    <p className="text-slate-400 text-lg max-w-3xl mx-auto font-light tracking-wide">
                        数据革新 · 智能跃迁 · 持续可设计的数据服务商
                    </p>
                </div>

                <div className="flex flex-col gap-12 relative">
                    
                    {/* Layer 1: Quantum Source Engine */}
                    <div className="relative">
                        <div className="text-center mb-6">
                            <span className="inline-block px-4 py-1 rounded-full bg-qcyan/10 border border-qcyan/30 text-qcyan text-sm font-bold tracking-widest shadow-[0_0_15px_rgba(69,243,255,0.2)]">
                                LAYER 01 / 量子源引擎
                            </span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative z-10">
                            <div className="glass-card rounded-2xl p-8 border border-white/10 hover:border-qcyan/50 transition-all group bg-black/40 backdrop-blur-xl relative overflow-hidden">
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-qcyan to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                                <div className="w-16 h-16 rounded-xl bg-qcyan/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                                    <IconAtom className="text-qcyan w-8 h-8" />
                                </div>
                                <h3 className="text-xl font-bold text-white mb-3">量子随机电路采样</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    依托 Quafu 真实超导量子芯片，执行深度随机电路采样，提取纯粹的物理随机性。
                                </p>
                            </div>
                            
                            <div className="glass-card rounded-2xl p-8 border border-white/10 hover:border-blue-400/50 transition-all group bg-black/40 backdrop-blur-xl relative overflow-hidden">
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-blue-400 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                                <div className="w-16 h-16 rounded-xl bg-blue-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                                    <svg className="w-8 h-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>
                                </div>
                                <h3 className="text-xl font-bold text-white mb-3">生成可控 PT 分布</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    将量子比特串转化为数学上完美的无偏差重尾先验，具备罕见长尾大跳跃特性。
                                </p>
                            </div>
                            
                            <div className="glass-card rounded-2xl p-8 border border-white/10 hover:border-emerald-400/50 transition-all group bg-black/40 backdrop-blur-xl relative overflow-hidden">
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                                <div className="w-16 h-16 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                                    <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                                </div>
                                <h3 className="text-xl font-bold text-white mb-3">具身数据生成中枢</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    利用多维最优传输与 Copula 技术，将量子先验无损转换为具身智能所需的物理扰动分布。
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Connecting animated particles/lines between Layer 1 and 2 (Hidden on mobile) */}
                    <div className="hidden md:flex justify-center -my-8 relative z-0">
                        <div className="w-px h-24 bg-gradient-to-b from-blue-500/50 to-transparent relative">
                            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-8 bg-blue-400 rounded-full blur-[2px] animate-pulse"></div>
                        </div>
                    </div>

                    {/* Layer 2: Calibration Scene Integration */}
                    <div className="relative">
                        <div className="text-center mb-6 mt-8 md:mt-0">
                            <span className="inline-block px-4 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-sm font-bold tracking-widest shadow-[0_0_15px_rgba(59,130,246,0.2)]">
                                LAYER 02 / 校准场景接入
                            </span>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 relative z-10">
                            {[
                                { name: "机器人操作", desc: "扭矩/摩擦力", icon: "🦾" },
                                { name: "传感器标定", desc: "噪点/畸变漂移", icon: "📷" },
                                { name: "环境建模", desc: "光照/材质突变", icon: "🌍" },
                                { name: "运动控制", desc: "复杂地形平衡", icon: "🏃" },
                                { name: "多模态对齐", desc: "时空对齐偏移", icon: "🧠" }
                            ].map((scene, i) => (
                                <div key={i} className="glass-card rounded-xl p-5 border border-white/5 hover:bg-white/10 hover:border-blue-400/30 transition-all text-center group cursor-default bg-black/40">
                                    <div className="text-3xl mb-3 grayscale group-hover:grayscale-0 transition-all scale-90 group-hover:scale-110">{scene.icon}</div>
                                    <h4 className="text-white font-bold text-sm mb-1 group-hover:text-blue-400 transition-colors">{scene.name}</h4>
                                    <div className="text-xs text-slate-500">{scene.desc}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Connecting animated particles/lines between Layer 2 and 3 (Hidden on mobile) */}
                    <div className="hidden md:flex justify-center -my-8 relative z-0">
                        <div className="w-px h-24 bg-gradient-to-b from-emerald-500/50 to-transparent relative">
                            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-8 bg-emerald-400 rounded-full blur-[2px] animate-pulse" style={{animationDelay: '150ms'}}></div>
                        </div>
                    </div>

                    {/* Layer 3: Value Output */}
                    <div className="relative mt-12 md:mt-0">
                        <div className="glass-card rounded-3xl p-1 relative overflow-hidden bg-gradient-to-b from-white/10 to-transparent">
                            <div className="absolute inset-0 bg-gradient-to-r from-qcyan/20 via-blue-500/20 to-emerald-500/20 blur-xl"></div>
                            <div className="bg-black/80 backdrop-blur-2xl rounded-[23px] p-10 relative z-10 border border-white/10 text-center">
                                <h3 className="text-2xl md:text-3xl font-extrabold text-white mb-10 tracking-wide">
                                    高质量具身数据集持续输出
                                </h3>
                                
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                                    {[
                                        { title: "高效生成", subtitle: "突破采集瓶颈", color: "text-qcyan", icon: "⚡" },
                                        { title: "分布可控", subtitle: "精确调比先验", color: "text-blue-400", icon: "🎯" },
                                        { title: "场景多样", subtitle: "MT50+任务族", color: "text-purple-400", icon: "🌌" },
                                        { title: "泛化增强", subtitle: "攻克 CVaR@20", color: "text-emerald-400", icon: "🛡️" }
                                    ].map((badge, i) => (
                                        <div key={i} className="flex flex-col items-center group cursor-default">
                                            <div className="text-3xl mb-3 group-hover:scale-110 transition-transform">{badge.icon}</div>
                                            <div className={`font-bold text-lg ${badge.color}`}>{badge.title}</div>
                                            <div className="text-xs text-slate-400 mt-1">{badge.subtitle}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </section>
    );\n\n        """

content = content.replace(old_arch, new_arch)
content = content.replace("<Architecture />", "<DataEngineArchitecture />")

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Replaced successfully!")
