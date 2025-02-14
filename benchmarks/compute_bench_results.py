#!/usr/bin/env python3
import sys
from statistics import mean, stdev

def parse_input(data):
    """
    Parse l'entrée texte et retourne une liste de sections.
    Chaque section est un dictionnaire contenant :
      - pop       : la taille de population (ex. "30")
      - fc        : le nom de la fonction (ex. "SPHERE")
      - dim       : la dimension (ex. "10")
      - times     : liste des temps d'exécution (colonne Time)
      - minimums  : liste des valeurs fitness (colonne Minimum)
    """
    sections = []
    current_section = None
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("POP"):
            if current_section is not None:
                sections.append(current_section)
            current_section = {"pop": line.split()[1],
                               "fc": "",
                               "dim": "",
                               "times": [],
                               "minimums": []}
        elif line.startswith("FC"):
            if current_section is not None:
                current_section["fc"] = line.split()[1]
        elif line.startswith("DIM"):
            if current_section is not None:
                current_section["dim"] = line.split()[1]
        elif line.startswith("Type"):
            # Ligne d'en-tête à ignorer
            continue
        elif line.startswith("GPU") or line.startswith("CPU"):
            # Ligne de données : "GPU <Time> <Minimum>"
            parts = line.split()
            if len(parts) >= 3:
                try:
                    time_val = float(parts[1])
                    min_val  = float(parts[2])
                    current_section["times"].append(time_val)
                    current_section["minimums"].append(min_val)
                except ValueError:
                    pass
    if current_section is not None:
        sections.append(current_section)
    return sections

def format_number(x):
    """
    Formate le nombre x pour l'affichage en LaTeX.
    - Si |x| < 1e-2 ou |x| >= 1e+2, on utilise la notation scientifique.
    - Sinon, on affiche avec 2 décimales (virgule comme séparateur décimal).
    """
    if abs(x) < 1e-2 or abs(x) >= 1e+2:
        s = "{:.1e}".format(x)  # ex. "4.1e-11"
        mantissa, exp = s.split("e")
        mantissa = round(float(mantissa))
        exponent = int(exp)  # enlève le signe +
        return "$\\scriptstyle " + str(mantissa) + "." + "10^{" + str(exponent) + "}$"
    else:
        s = "{:.2f}".format(x)
        s = s.replace(".", ",")
        return "$\\scriptstyle " + s + "$"

def aggregate_results(sections):
    """
    Regroupe les résultats par dimension, fonction et taille de population.
    Pour chaque section (unique combinaison DIM, FC, POP) on calcule :
      - fitness_mean et fitness_std : moyenne et écart-type de la fitness (colonne Minimum)
      - time_mean : moyenne du temps (colonne Time)
    Retourne un dictionnaire structuré de la forme :
      results[dim][fc][pop] = { "fitness_mean": ..., "fitness_std": ..., "time_mean": ... }
    """
    results = {}
    for sec in sections:
        dim = sec["dim"]
        fc = sec["fc"]
        pop = sec["pop"]
        if not sec["minimums"] or not sec["times"]:
            continue
        fmean = mean(sec["minimums"])
        fstd = stdev(sec["minimums"]) if len(sec["minimums"]) > 1 else 0.0
        tmean = mean(sec["times"])
        if dim not in results:
            results[dim] = {}
        if fc not in results[dim]:
            results[dim][fc] = {}
        results[dim][fc][pop] = {"fitness_mean": fmean,
                                 "fitness_std": fstd,
                                 "time_mean": tmean}
    return results

def fc_display_name(fc):
    """
    Retourne le nom de la fonction à afficher en LaTeX.
    Par exemple, "SPHERE" sera affiché en "Sphère".
    """
    if fc.upper() == "SPHERE":
        return "Sphère"
    else:
        return fc.title()

def generate_latex_tables_dual(cpu_results, gpu_results):
    """
    Génère pour chaque dimension un tableau LaTeX reprenant exactement votre modèle de base.
    Le tableau comporte 8 colonnes (2 fixes puis 3 pour CPU et 3 pour GPU) et pour chaque fonction
    3 lignes : la moyenne ($\mu$), l'écart-type ($\sigma$) et le temps d'exécution.
    Les intitulés des colonnes pour CPU et GPU sont fixés à P30, P50 et P70.
    """
    # On fixe les populations à P30, P50, P70 pour CPU et GPU
    fixed_cpu_pops = ["30", "50", "70"]
    fixed_gpu_pops = ["30", "50", "70"]
    dims = list(set(cpu_results.keys()) | set(gpu_results.keys()))
    dims.sort(key=lambda d: int(d))
    latex = ""
    for dim in dims:
        table = "    \\centering\n"
        table += "    \\subfigure[Dimension " + dim + "]{\n"
        table += "        \\setlength{\\tabcolsep}{2pt}\n"
        table += "        \\begin{tabular}{|l|l|c|c|c|c|c|c|}\n"
        table += "            \\hline\n"
        table += "            \\multicolumn{8}{|c|}{\\textbf{Dimension " + dim + "}} \\\\ \\hline\n"
        table += "            \\multicolumn{2}{|c|}{\\multirow{2}{*}{}} & \\multicolumn{3}{c|}{\\textbf{CPU}} & \\multicolumn{3}{c|}{\\textbf{GPU}} \\\\ \\cline{3-8}\n"
        table += "            \\multicolumn{2}{|c|}{} & \\multicolumn{1}{c|}{\\textbf{P" + fixed_cpu_pops[0] + "}} & \\multicolumn{1}{c|}{\\textbf{P" + fixed_cpu_pops[1] + "}} & \\multicolumn{1}{c|}{\\textbf{P" + fixed_cpu_pops[2] + "}} & \\multicolumn{1}{c|}{\\textbf{P" + fixed_gpu_pops[0] + "}} & \\multicolumn{1}{c|}{\\textbf{P" + fixed_gpu_pops[1] + "}} & \\multicolumn{1}{c|}{\\textbf{P" + fixed_gpu_pops[2] + "}} \\\\ \\hline\n"
        # Récupère l'ensemble des fonctions pour cette dimension
        fc_set = set()
        if dim in cpu_results:
            fc_set.update(cpu_results[dim].keys())
        if dim in gpu_results:
            fc_set.update(gpu_results[dim].keys())
        # Ordre préféré
        preferred_order = ["SPHERE", "ROSENBROCK", "ACKLEY", "RASTRIGIN"]
        def order_key(fc):
            try:
                return preferred_order.index(fc.upper())
            except ValueError:
                return 100
        fc_list = list(fc_set)
        fc_list.sort(key=order_key)
        # Pour chaque fonction, 3 lignes : $\mu$, $\sigma$ et Time
        for fc in fc_list:
            disp_name = fc_display_name(fc)
            # Pour CPU
            mu_cpu = []
            sigma_cpu = []
            time_cpu = []
            for pop in fixed_cpu_pops:
                if (dim in cpu_results and fc in cpu_results[dim] and pop in cpu_results[dim][fc]):
                    vals = cpu_results[dim][fc][pop]
                    mu_cpu.append(format_number(vals["fitness_mean"]))
                    sigma_cpu.append(format_number(vals["fitness_std"]))
                    time_cpu.append(format_number(vals["time_mean"]))
                else:
                    mu_cpu.append("")
                    sigma_cpu.append("")
                    time_cpu.append("")
            # Pour GPU
            mu_gpu = []
            sigma_gpu = []
            time_gpu = []
            for pop in fixed_gpu_pops:
                if (dim in gpu_results and fc in gpu_results[dim] and pop in gpu_results[dim][fc]):
                    vals = gpu_results[dim][fc][pop]
                    mu_gpu.append(format_number(vals["fitness_mean"]))
                    sigma_gpu.append(format_number(vals["fitness_std"]))
                    time_gpu.append(format_number(vals["time_mean"]))
                else:
                    mu_gpu.append("")
                    sigma_gpu.append("")
                    time_gpu.append("")
            # Concatène les valeurs CPU et GPU
            line_mu = "            \\multicolumn{1}{|c|}{\\multirow{3}{*}{\\textbf{" + disp_name + "}}} & \\multicolumn{1}{c|}{\\boldmath$\\mu$} & " + " & ".join(mu_cpu + mu_gpu) + " \\\\ \\cline{2-8}\n"
            line_sigma = "            \\multicolumn{1}{|c|}{} & \\multicolumn{1}{c|}{\\boldmath$\\sigma$} & " + " & ".join(sigma_cpu + sigma_gpu) + " \\\\ \\cline{2-8}\n"
            line_time = "            \\multicolumn{1}{|c|}{} & \\boldmath$\\tau$ & " + " & ".join(time_cpu + time_gpu) + " \\\\ \\hline\n"
            table += line_mu + line_sigma + line_time
        table += "        \\end{tabular}\n"
        table += "    }\n"
        latex += table
    return latex

def main():
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " fichier_CPU.txt fichier_GPU.txt")
        sys.exit(1)
    cpu_filename = sys.argv[1]
    gpu_filename = sys.argv[2]
    try:
        with open(cpu_filename, "r") as f:
            cpu_data = f.read()
    except Exception as e:
        print("Erreur lors de la lecture du fichier " + cpu_filename + ": " + str(e))
        sys.exit(1)
    try:
        with open(gpu_filename, "r") as f:
            gpu_data = f.read()
    except Exception as e:
        print("Erreur lors de la lecture du fichier " + gpu_filename + ": " + str(e))
        sys.exit(1)
    
    sections_cpu = parse_input(cpu_data)
    sections_gpu = parse_input(gpu_data)
    cpu_results = aggregate_results(sections_cpu)
    gpu_results = aggregate_results(sections_gpu)
    
    latex_tables = generate_latex_tables_dual(cpu_results, gpu_results)
    print(latex_tables)

if __name__ == '__main__':
    main()
