import sys

def add_download_section(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_section = """
                                {semanticState > 0 && (
                                    <div className="mt-8 border-t border-white/10 pt-6 animate-fade-in">
                                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-2">
                                            <div>
                                                <h4 className="text-lg font-bold text-white flex items-center gap-2">批量量子映射配置 (Rank-Based OT)</h4>
                                                <p className="text-xs text-slate-400 mt-1">基于最优传输原理，将所有 data/ 目录下的量子重尾概率转化为 MT10 的任务采样分布</p>
                                            </div>
                                        </div>
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                                            {[
                                                { name: "task_2604201111555115048_result 2_ot_mapping.json", source: "task_2604201111555115048_result 2.csv", url: "results/mappings/task_2604201111555115048_result 2_ot_mapping.json" },
                                                { name: "task_2604201358523029585_result_ot_mapping.json", source: "task_2604201358523029585_result.csv", url: "results/mappings/task_2604201358523029585_result_ot_mapping.json" },
                                                { name: "task_2604201408246794658_result_ot_mapping.json", source: "task_2604201408246794658_result.csv", url: "results/mappings/task_2604201408246794658_result_ot_mapping.json" },
                                                { name: "task_2604201410376483513_result_ot_mapping.json", source: "task_2604201410376483513_result.csv", url: "results/mappings/task_2604201410376483513_result_ot_mapping.json" },
                                                { name: "task_2604221030543992190_result_ot_mapping.json", source: "task_2604221030543992190_result.csv", url: "results/mappings/task_2604221030543992190_result_ot_mapping.json" }
                                            ].map((file, idx) => (
                                                <div key={idx} className="bg-black/40 border border-white/5 rounded-lg p-3 flex justify-between items-center hover:border-qpurple/50 transition-colors group">
                                                    <div className="flex flex-col overflow-hidden mr-3">
                                                        <span className="text-sm text-qpurple truncate font-mono" title={file.name}>{file.name}</span>
                                                        <span className="text-[10px] text-slate-500 truncate mt-0.5">Source: {file.source}</span>
                                                    </div>
                                                    <a href={file.url} download className="p-2 bg-white/5 hover:bg-qpurple/20 rounded-md text-slate-300 hover:text-white transition-colors flex-shrink-0" title="下载配置 JSON">
                                                        <IconDownload className="w-4 h-4" />
                                                    </a>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}"""

    search_str = """                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}"""
                        
    replace_str = """                                            </div>
                                        )}
                                    </div>
                                </div>""" + new_section

    if search_str in content:
        new_content = content.replace(search_str, replace_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully added download section to {filepath}")
    else:
        print(f"Could not find the target string in {filepath}")

add_download_section('index.html')
add_download_section('qtail-mvp-presentation.html')
