import re

with open('temp_results.js', 'r') as f:
    code = f.read()

code = code.replace("            return (", """
            const pt_rank = data?.metrics['pt-rank'];
            const empirical = data?.metrics['empirical'];
            const tail_diff = pt_rank && empirical ? (pt_rank.tail_sr - empirical.tail_sr).toFixed(1) : '41.1';
            const cvar_diff = pt_rank && empirical ? (pt_rank.cvar20 - empirical.cvar20).toFixed(1) : '44.5';
            const overall_pt = pt_rank ? pt_rank.overall.toFixed(1) : '81.7';

            return (""")

code = code.replace(">+41.1pp<", ">+{tail_diff}pp<")
code = code.replace(">+44.5pp<", ">+{cvar_diff}pp<")
code = code.replace(">81.7%<", ">{overall_pt}%<")

with open('temp_results.js', 'w') as f:
    f.write(code)
