
import pandas as pd
import numpy as np


METHOD_NAMES = {
    "original": "Original",
    "statgen": "Stat. Gen.",
    "ctgan": "CTGAN",
    "tabddpm": "TabDDPM",
    "great": "GReaT",
    "tabula": "Tabula",
    "epic": "EPIC",
    "synthloop": "SynthLoop",
    "reduced": "ReducedLoop",
    "promptrefine": "Prompt-Refine",
    "info-full": "Info - Full",
    "info-weakness": "Info - Weakness",
    "noInfo-full": "No Info - Full",
    "noInfo-weakness": "No Info - Weakness",
}





def str_to_dict(s):
    res = {}
    s = s.strip("{").strip("}").split(", ")
    for pair in s:
        k, v = pair.split(": ")
        k = k.strip("'")
        res[k] = float(v)
    return res

def split_dict_col(col):
    col_values = col.apply(str_to_dict).reset_index(drop=True)
    df = pd.DataFrame(columns=col_values[0].keys())
    for i in range(len(col_values)):
        df.loc[i] = col_values[i]
    return df



def get_table_comp_models_dataset_OLD(dname,
                                  scores,
                                  scoring="roc_auc",
                                  model_order=["original", "ctgan", "tabddpm", "great", "tabula", "epic", "synthloop"],
                                  model_tex=["Original", "CTGAN", "TabDDPM", "GReaT", "Tabula", "EPIC", "SynthLoop"],
                                  col_order=["Method", "Utility", "Detection", "precision", "recall", "density", "coverage", "alpha_precision", "beta_recall"],
                                  ):
    if scoring not in ["accuracy", "roc_auc", "f1"]:
        raise ValueError("scoring must be one of accuracy, roc_auc, f1")

    cols = []
    for col in scores.columns:
        if "detection" in col or "utility" in col:
            if scoring in col:
                cols.append(col)
        else:
            cols.append(col)
    scores = scores[cols]

    ### utility ###
    cols_utility = [col for col in scores.columns if "utility" in col]
    cols_trtr = [col for col in cols_utility if "trtr" in col]
    cols_tstr = [col for col in cols_utility if "tstr" in col]
    cols_combined = [col for col in cols_utility if "combined" in col]
    utilities = {}

    mean_og = 0
    std_og = 0
    for col in cols_trtr:
        # data has format: {"mean": ..., "std": ...}
        col_values = split_dict_col(scores[col])
        mean_og += col_values["mean"].mean()
        std_og += col_values["std"].mean()
    mean_og /= len(cols_trtr)
    std_og /= len(cols_trtr)
    utilities["original"] = {"mean": mean_og, "std": std_og}

    for model_name in model_order[1:]:
        line = scores[scores["model_name"] == model_name]
        mean_val = 0
        std_val = 0
        cols_selected = cols_tstr  # cols_combined / cols_tstr      # CHANGE HERE FOR COMBINED OR TSTR
        for col in cols_selected:
            col_values = split_dict_col(line[col])
            mean_val += col_values["mean"][0]
            std_val += col_values["std"][0]
        mean_val /= len(cols_selected)
        std_val /= len(cols_selected)
        utilities[model_name] = {"mean": mean_val, "std": std_val}

    ### detection ###
    cols_detection = [col for col in scores.columns if "detection" in col]
    detections = {}
    for model_name in model_order[1:]:
        line = scores[scores["model_name"] == model_name]
        mean_val = 0
        std_val = 0
        for col in cols_detection:
            col_values = split_dict_col(line[col])
            mean_val += col_values["mean"][0]
            std_val += col_values["std"][0]
        mean_val /= len(cols_detection)
        std_val /= len(cols_detection)
        detections[model_name] = {"mean": mean_val, "std": std_val}

    # compute best score for each col
    best_scores = {}
    for col in col_order[1:]:
        if col == "Utility":
            best_scores[col] = max([utilities[model_name]["mean"] for model_name in model_order])
        elif col == "Detection":
            best_scores[col] = min([detections[model_name]["mean"] for model_name in model_order[1:]])
        elif col == "Duplicates":
            best_scores[col] = min(scores[col].astype(float))
        else:
            best_scores[col] = max(scores[col].astype(float))


    table = "\\begin[tabular][l".replace("[", "{").replace("]", "}")
    for _ in col_order[1:]:
        table += "c"
    table += "}\n"
    table += "\\hline\n"
    for c in col_order:
        if "_" in c:
            c = " ".join([w.capitalize() for w in c.split("_")])
            table += f"{c} $\\uparrow$ & "
        else:
            if c == "Detection" or c == "Duplicates":
                table += f"{c.capitalize()} $\\downarrow$ & "
            elif c == "Method":
                table += f"{c} & "
            else:
                table += f"{c.capitalize()} $\\uparrow$ & "
    table = table[:-2] + "\\\\\n"
    table += "\\hline\n"
    for i, model_name in enumerate(model_order):
        if model_name == "original":
            val = utilities['original']['mean']
            if val == best_scores["Utility"]:
                table += f"Original & \\textbf{{{val:.2f}}} ({utilities['original']['std']:.2f}) & "
            else:
                table += f"Original & {val:.2f} ({utilities['original']['std']:.2f}) & "
            for c in col_order[2:]:
                table += "- & "
        
        else:
            table += f"{model_tex[i]} & "
            for col in col_order[1:]:
                if col == "Utility":
                    val = utilities[model_name]['mean']
                    if val == best_scores["Utility"]:
                        table += f"\\textbf{{{val:.2f}}} ({utilities[model_name]['std']:.2f}) & "
                    else:
                        table += f"{val:.2f} ({utilities[model_name]['std']:.2f}) & "
                elif col == "Detection":
                    val = detections[model_name]['mean']
                    if val == best_scores["Detection"]:
                        table += f"\\textbf{{{val:.2f}}} ({detections[model_name]['std']:.2f}) & "
                    else:
                        table += f"{val:.2f} ({detections[model_name]['std']:.2f}) & "
                else:
                    val = float(scores[scores["model_name"] == model_name][col].iloc[0])
                    if val == best_scores[col]:
                        table += f"\\textbf{{{val:.2f}}} & "
                    else:
                        table += f"{val:.2f} & "
        
        
        table = table[:-2] + "\\\\\n"
        if model_name == "tabula":
            table += "\\hdashline\n"

    table += "\\hline\n\\end{tabular}"
    return table


def get_table_comp_models_dataset(
                                  scores,
                                  scoring="roc_auc",
                                  model_order=["original", "ctgan", "tabddpm", "great", "tabula", "epic", "synthloop"],
                                  model_tex=["Original", "CTGAN", "TabDDPM", "GReaT", "Tabula", "EPIC", "SynthLoop"],
                                  col_order=["Method", "Utility", "Detection", "precision", "recall", "density", "coverage", "Duplicates", "Collisions"],
                                  ):
    if scoring not in ["accuracy", "roc_auc", "f1"]:
        raise ValueError("scoring must be one of accuracy, roc_auc, f1")

    cols = []
    for col in scores.columns:
        if "detection" in col or "utility" in col:
            if scoring in col:
                cols.append(col)
        else:
            cols.append(col)
    scores = scores[cols]

    ### utility ###
    cols_utility = [col for col in scores.columns if "utility" in col]
    cols_trtr = [col for col in cols_utility if "trtr" in col]
    cols_tstr = [col for col in cols_utility if "tstr" in col]
    cols_combined = [col for col in cols_utility if "combined" in col]
    utilities = {}

    mean_og = 0
    std_og = 0
    for col in cols_trtr:
        # data has format: {"mean": ..., "std": ...}
        col_values = split_dict_col(scores[col])
        mean_og += col_values["mean"].mean()
        std_og += col_values["std"].mean()
    mean_og /= len(cols_trtr)
    std_og /= len(cols_trtr)
    utilities["original"] = {"mean": mean_og, "std": std_og}

    for model_name in model_order[1:]:
        line = scores[scores["model_name"] == model_name]
        mean_val = 0
        std_val = 0
        cols_selected = cols_tstr  # cols_combined / cols_tstr      # CHANGE HERE FOR COMBINED OR TSTR
        for col in cols_selected:
            col_values = split_dict_col(line[col])
            mean_val += col_values["mean"][0]
            std_val += col_values["std"][0]
        mean_val /= len(cols_selected)
        std_val /= len(cols_selected)
        utilities[model_name] = {"mean": mean_val, "std": std_val}

    ### detection ###
    cols_detection = [col for col in scores.columns if "detection" in col]
    detections = {}
    for model_name in model_order[1:]:
        line = scores[scores["model_name"] == model_name]
        mean_val = 0
        std_val = 0
        for col in cols_detection:
            col_values = split_dict_col(line[col])
            mean_val += col_values["mean"][0]    # mean and std over classification models used
            std_val += col_values["std"][0]
        mean_val /= len(cols_detection)
        std_val /= len(cols_detection)
        detections[model_name] = {"mean": mean_val, "std": std_val}

    ### prdc ###
    names_prdc_cols = ["precision", "recall", "density", "coverage", "alpha_precision", "beta_recall"]
    prdc_scores = {}
    for col in names_prdc_cols:
        for model_name in model_order[1:]:
            prdc_scores[col] = {model_name: {}}  # init dict for the results

    for model_name in model_order[1:]:
        line = scores[scores["model_name"] == model_name]
        for col in names_prdc_cols:
            col_values = split_dict_col(line[col])
            mean_val = col_values["mean"][0]
            std_val = col_values["std"][0]
            prdc_scores[col][model_name] = {"mean": mean_val, "std": std_val}


    # compute best score for each col
    best_scores = {}
    for col in col_order[1:]:
        if col == "Utility":
            best_scores[col] = max([utilities[model_name]["mean"] for model_name in model_order])
        elif col == "Detection":
            best_scores[col] = min([detections[model_name]["mean"] for model_name in model_order[1:]])
        elif col == "Duplicates" or col == "Collisions":
            best_scores[col] = min(scores[col].astype(float))
        else:
            best_scores[col] = max([prdc_scores[col][model_name]["mean"] for model_name in model_order[1:]])  #max(scores[col].astype(float))


    table = "\\begin[tabular][l".replace("[", "{").replace("]", "}")
    for _ in col_order[1:]:
        table += "c"
    table += "}\n"
    table += "\\hline\n"
    for c in col_order:
        if "_" in c:
            c = " ".join([w.capitalize() for w in c.split("_")])
            table += f"{c} $\\uparrow$ & "
        else:
            if c == "Detection":
                table += f"{c.capitalize()} $\\downarrow$ & "
            elif c == "Collisions" or c == "Duplicates":
                table += f"{c.capitalize()} [\%] $\\downarrow$ & "
            elif c == "Method":
                table += f"{c} & "
            else:
                table += f"{c.capitalize()} $\\uparrow$ & "
    table = table[:-2] + "\\\\\n"
    table += "\\hline\n"
    for i, model_name in enumerate(model_order):
        if model_name == "original":
            val = utilities['original']['mean']
            if val == best_scores["Utility"]:
                table += f"Original & \\textbf{{{val:.2f}}} ({utilities['original']['std']:.2f}) & "
            else:
                table += f"Original & {val:.2f} ({utilities['original']['std']:.2f}) & "
            for c in col_order[2:]:
                table += "- & "
        
        else:
            table += f"{model_tex[i]} & "
            for col in col_order[1:]:
                if col == "Utility":
                    val = utilities[model_name]['mean']
                    if val == best_scores["Utility"]:
                        table += f"\\textbf{{{val:.2f}}} ({utilities[model_name]['std']:.2f}) & "
                    else:
                        table += f"{val:.2f} ({utilities[model_name]['std']:.2f}) & "
                elif col == "Detection":
                    val = detections[model_name]['mean']
                    if val == best_scores["Detection"]:
                        table += f"\\textbf{{{val:.2f}}} ({detections[model_name]['std']:.2f}) & "
                    else:
                        table += f"{val:.2f} ({detections[model_name]['std']:.2f}) & "
                elif col in names_prdc_cols:
                    val = prdc_scores[col][model_name]["mean"]
                    if val == best_scores[col]:
                        table += f"\\textbf{{{val:.2f}}} ({prdc_scores[col][model_name]['std']:.2f}) & "
                    else:
                        table += f"{val:.2f} ({prdc_scores[col][model_name]['std']:.2f}) & "
                else:
                    val = float(scores[scores["model_name"] == model_name][col].iloc[0])
                    if val == best_scores[col]:
                        table += f"\\textbf{{{val:.2f}}} & "
                    else:
                        table += f"{val:.2f} & "
        
        
        table = table[:-2] + "\\\\\n"
        if model_name == "tabula":
            table += "\\hdashline\n"

    table += "\\hline\n\\end{tabular}"
    return table



def get_all_scores(dnames, 
                   filename):
    scores = {}
    # scores format:
    # {dname: {model_name: {table_columns: value}}}

    for dname in dnames:
        scores[dname] = {}
        df = pd.read_csv(f"evaluation_gen_data/{dname}/{filename}")

        model_names = df["model_name"].tolist()

        scores[dname]["Original"] = {"Utility TSTR": None, "Utility Combined": None, "Detection": None, "precision": None, "recall": None, "density": None, "coverage": None, "Duplicates": None, "Collisions": None}

        for model_name in model_names:
            scores[dname][model_name] = {}
            line = df[df["model_name"] == model_name]

            ## compute the utility scores
            cols_utility = [col for col in line.columns if "utility" in col]
            cols_trtr = [col for col in cols_utility if "trtr" in col]
            cols_tstr = [col for col in cols_utility if "tstr" in col]
            cols_combined = [col for col in cols_utility if "combined" in col]

            # compute the mean over all columns
            mean_trtr = 0
            for col in cols_trtr:
                col_value = str_to_dict(line[col].tolist()[0])
                mean_trtr += col_value["mean"]
            mean_trtr /= len(cols_trtr)

            mean_tstr = 0
            for col in cols_tstr:
                col_value = str_to_dict(line[col].tolist()[0])
                mean_tstr += col_value["mean"]
            mean_tstr /= len(cols_tstr)

            mean_combined = 0
            for col in cols_combined:
                col_value = str_to_dict(line[col].tolist()[0])
                mean_combined += col_value["mean"]
            mean_combined /= len(cols_combined)

            scores[dname]["Original"]["Utility TSTR"] = mean_trtr
            scores[dname][model_name]["Utility TSTR"] = mean_tstr
            scores[dname][model_name]["Utility Combined"] = mean_combined

            # compute the detection scores
            cols_detection = [col for col in line.columns if "detection" in col]
            mean_detection = 0
            for col in cols_detection:
                col_value = str_to_dict(line[col].tolist()[0])
                mean_detection += col_value["mean"]
            mean_detection /= len(cols_detection)
            scores[dname][model_name]["Detection"] = mean_detection

            # get prdc scores
            for col in ["precision", "recall", "density", "coverage"]:
                scores[dname][model_name][col] = str_to_dict(line[col].tolist()[0])["mean"]

            # get collisions and duplicates
            scores[dname][model_name]["Collisions"] = float(line["Collisions"].tolist()[0])
            scores[dname][model_name]["Duplicates"] = float(line["Duplicates"].tolist()[0])

            # get number of data and limit model
            scores[dname][model_name]["n_data"] = int(line["Data size"].tolist()[0])
            scores[dname][model_name]["limit_model"] = line["Limit model"].tolist()[0]
    
    return scores


def get_max_scores(scores, dname, col_name):
    col_scores = [scores[dname][model_name][col_name] for model_name in scores[dname] if scores[dname][model_name][col_name] != None]
    return max(col_scores)

def get_min_scores(scores, dname, col_name):
    col_scores = [scores[dname][model_name][col_name] for model_name in scores[dname] if scores[dname][model_name][col_name] != None]
    return min(col_scores)


def create_tex_table(scores,
                     cols_order,
                     dnames_ignored=[],
                     model_order=[]):
    if cols_order[0] != "Dataset" and cols_order[1] != "Model":
        raise ValueError("First two columns must be Dataset and Model")

    max_cols = ["Utility TSTR", "Utility Combined", "precision", "recall", "density", "coverage"]
    min_cols = ["Detection", "Duplicates"]

    table = "\\begin{tabular}{ll"
    for _ in cols_order[2:]:
        table += "c"
    table += "}\n"
    table += "\\hline\n"
    for c in cols_order:
        if c == "Collisions" or c == "Duplicates":
            table += f"\\textbf{{{c.capitalize()}}} [\%] "
        elif c == "Utility TSTR":
            table += "\\textbf{U. TSTR} "
        elif c == "Utility Combined":
            table += "\\textbf{U. Comb} "
        # elif " " in c:
        #     table += f"\\textbf{{{c}}} "
        else:
            table += f"\\textbf{{{c.capitalize()}}} "
        if c in max_cols:
            table += "$\\uparrow$ & "
        elif c in min_cols:
            table += "$\\downarrow$ & "
        else:
            table += "& "

    table = table[:-2] + "\\\\\n"
    table += "\\hline\n\\hline\n"

    n_models = len(list(scores.values())[0].keys())

    for dname in scores:

        if dname in dnames_ignored:
            continue

        # compute all min and max scores
        best_scores = {}
        for col in cols_order[2:]:
            if col in max_cols:
                best_scores[col] = get_max_scores(scores, dname, col)
            elif col in min_cols:
                best_scores[col] = get_min_scores(scores, dname, col)

        n_data = scores[dname][list(scores[dname].keys())[1]]["n_data"]
        dnaming = dname.capitalize() + f" ({n_data})"
        table += f"\\multirow{{{n_models}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{dnaming}}}}} & "

        order = model_order if model_order else list(scores[dname].keys())

        for model_name in order:
            model_naming = model_name
            if "_" in model_naming:
                model_naming = " ".join([w.capitalize() for w in model_naming.split("_")])
            if model_naming in METHOD_NAMES:
                model_naming = METHOD_NAMES[model_naming]

            table += f"{model_naming} & "
            for col in cols_order[2:]:
                val = scores[dname][model_name][col]
                if val == None:
                    table += "- & "
                elif col in best_scores and val == best_scores[col]:
                    table += f"\\textbf{{{val:.2f}}} & "
                else:
                    table += f"{val:.2f} & "
            table = table[:-2] + "\\\\\n"
            table += "& "
        table = table[:-2] + "\n"
        table += "\\hline\n"

    table += "\\end{tabular}"
    return table


def create_tex_table_llm(scores,
                     cols_order,
                     dnames_ignored=[],
                     model_order=[]):
    if cols_order[0] != "Dataset" and cols_order[1] != "LLM" and cols_order[2] != "Model":
        raise ValueError("First three columns must be Dataset and LLM and Model")

    max_cols = ["Utility TSTR", "Utility Combined", "precision", "recall", "density", "coverage"]
    min_cols = ["Detection", "Duplicates"]

    table = "\\begin{tabular}{cll"
    for _ in cols_order[3:]:
        table += "c"
    table += "}\n"
    table += "\\hline\n"
    for c in cols_order:
        if c == "Collisions" or c == "Duplicates":
            table += f"\\textbf{{{c.capitalize()}}} [\%] "
        elif c == "Utility TSTR":
            table += "\\textbf{U. TSTR} "
        elif c == "Utility Combined":
            table += "\\textbf{U. Comb} "
        elif c == "LLM":
            table += "\\textbf{LLM} "
        # elif " " in c:
        #     table += f"\\textbf{{{c}}} "
        else:
            table += f"\\textbf{{{c.capitalize()}}} "
        if c in max_cols:
            table += "$\\uparrow$ & "
        elif c in min_cols:
            table += "$\\downarrow$ & "
        else:
            table += "& "

    table = table[:-2] + "\\\\\n"
    table += "\\hline\n\\hline\n"

    n_models = len(list(scores.values())[0].keys())

    for dname in scores:

        if dname in dnames_ignored:
            continue

        # compute all min and max scores
        best_scores = {}
        for col in cols_order[3:]:
            if col in max_cols:
                best_scores[col] = get_max_scores(scores, dname, col)
            elif col in min_cols:
                best_scores[col] = get_min_scores(scores, dname, col)

        n_data = scores[dname][list(scores[dname].keys())[1]]["n_data"]
        dnaming = dname.capitalize() + f" ({n_data})"
        table += f"\\multirow{{{n_models}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{dnaming}}}}} & "

        # line result for original
        table += "& Original & "
        for col in cols_order[3:]:
            val = scores[dname]["Original"][col]
            if val == None:
                table += "- & "
            elif col in best_scores and val == best_scores[col]:
                table += f"\\textbf{{{val:.2f}}} & "
            else:
                table += f"{val:.2f} & "
        table = table[:-2] + "\\\\\n"
        table += f"\\cdashline{{2-{len(cols_order)}}}\n"

        for i, llm in enumerate(["Llama 3.1", "GPT-4o", "DeepSeek-v3"]):
            # table += f"\\multirow{{{3}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{llm}}}}} & "
            table += f"& \\multirow{{{3}}}{{*}}{{{llm}}} & "
            order = model_order if model_order else list(scores[dname].keys())
            order = order[1:]

            for model_name in order[i*3:(i+1)*3]:
                model_naming = model_name.split("_")[0]
                if model_naming in METHOD_NAMES:
                    model_naming = METHOD_NAMES[model_naming]
                table += f"{model_naming} & "
                for col in cols_order[3:]:
                    val = scores[dname][model_name][col]
                    if val == None:
                        table += "- & "
                    elif col in best_scores and val == best_scores[col]:
                        table += f"\\textbf{{{val:.2f}}} & "
                    else:
                        table += f"{val:.2f} & "
                table = table[:-2] + "\\\\\n"
                table += "& & "
            table = table[:-4]
            if i != 2:
                table += f"\\cdashline{{2-{len(cols_order)}}}\n"
        # table = table[:-2] + "\n"
        table += "\\hline\n"
    
    table += "\\end{tabular}"
    return table


def create_tex_table_infoweakness(scores,
                     cols_order,
                     dnames_ignored=[],
                     model_order=[]):
    if cols_order[0] != "Dataset" and cols_order[1] != "Model" and cols_order[2] != "Variant":
        raise ValueError("First three columns must be Dataset and Model and Variant")

    max_cols = ["Utility TSTR", "Utility Combined", "precision", "recall", "density", "coverage"]
    min_cols = ["Detection", "Duplicates"]

    table = "\\begin{tabular}{cll"
    for _ in cols_order[3:]:
        table += "c"
    table += "}\n"
    table += "\\hline\n"
    for c in cols_order:
        if c == "Collisions" or c == "Duplicates":
            table += f"\\textbf{{{c.capitalize()}}} [\%] "
        elif c == "Utility TSTR":
            table += "\\textbf{U. TSTR} "
        elif c == "Utility Combined":
            table += "\\textbf{U. Comb} "
        elif c == "LLM":
            table += "\\textbf{LLM} "
        # elif " " in c:
        #     table += f"\\textbf{{{c}}} "
        else:
            table += f"\\textbf{{{c.capitalize()}}} "
        if c in max_cols:
            table += "$\\uparrow$ & "
        elif c in min_cols:
            table += "$\\downarrow$ & "
        else:
            table += "& "

    table = table[:-2] + "\\\\\n"
    table += "\\hline\n\\hline\n"

    n_models = len(list(scores.values())[0].keys())

    for dname in scores:

        if dname in dnames_ignored:
            continue

        # compute all min and max scores
        best_scores = {}
        for col in cols_order[3:]:
            if col in max_cols:
                best_scores[col] = get_max_scores(scores, dname, col)
            elif col in min_cols:
                best_scores[col] = get_min_scores(scores, dname, col)

        n_data = scores[dname][list(scores[dname].keys())[1]]["n_data"]
        dnaming = dname.capitalize() + f" ({n_data})"
        table += f"\\multirow{{{n_models}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{dnaming}}}}} & "

        # line result for original
        table += "& Original & "
        for col in cols_order[3:]:
            val = scores[dname]["Original"][col]
            if val == None:
                table += "- & "
            elif col in best_scores and val == best_scores[col]:
                table += f"\\textbf{{{val:.2f}}} & "
            else:
                table += f"{val:.2f} & "
        table = table[:-2] + "\\\\\n"
        table += f"\\cdashline{{2-{len(cols_order)}}}\n"

        for i, model in enumerate(["synthloop", "promptrefine"]):
            # table += f"\\multirow{{{3}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{llm}}}}} & "
            table += f"& \\multirow{{{4}}}{{*}}{{{METHOD_NAMES[model]}}} & "
            order = model_order if model_order else list(scores[dname].keys())
            order = order[1:]

            for variant in order[i*4:(i+1)*4]:
                model_naming = variant.split("_")[1]
                if model_naming in METHOD_NAMES:
                    model_naming = METHOD_NAMES[model_naming]
                table += f"{model_naming} & "
                for col in cols_order[3:]:
                    val = scores[dname][variant][col]
                    if val == None:
                        table += "- & "
                    elif col in best_scores and val == best_scores[col]:
                        table += f"\\textbf{{{val:.2f}}} & "
                    else:
                        table += f"{val:.2f} & "
                table = table[:-2] + "\\\\\n"
                table += "& & "
            table = table[:-4]
            if i != 1:
                table += f"\\cdashline{{2-{len(cols_order)}}}\n"
        # table = table[:-2] + "\n"
        table += "\\hline\n"
    
    table += "\\end{tabular}"
    return table



def create_tex_table_variants(scores,
                     cols_order,
                     dnames_ignored=[],
                     model_order=[]):
    if cols_order[0] != "Dataset" and cols_order[1] != "Model" and cols_order[2] != "Variant":
        raise ValueError("First three columns must be Dataset and Model and Variant")

    max_cols = ["Utility TSTR", "Utility Combined", "precision", "recall", "density", "coverage"]
    min_cols = ["Detection", "Duplicates"]

    table = "\\begin{tabular}{cll"
    for _ in cols_order[3:]:
        table += "c"
    table += "}\n"
    table += "\\hline\n"
    for c in cols_order:
        if c == "Collisions" or c == "Duplicates":
            table += f"\\textbf{{{c.capitalize()}}} [\%] "
        elif c == "Utility TSTR":
            table += "\\textbf{U. TSTR} "
        elif c == "Utility Combined":
            table += "\\textbf{U. Comb} "
        elif c == "LLM":
            table += "\\textbf{LLM} "
        # elif " " in c:
        #     table += f"\\textbf{{{c}}} "
        else:
            table += f"\\textbf{{{c.capitalize()}}} "
        if c in max_cols:
            table += "$\\uparrow$ & "
        elif c in min_cols:
            table += "$\\downarrow$ & "
        else:
            table += "& "

    table = table[:-2] + "\\\\\n"
    table += "\\hline\n\\hline\n"

    n_models = len(list(scores.values())[0].keys())

    for dname in scores:

        if dname in dnames_ignored:
            continue

        # compute all min and max scores
        best_scores = {}
        for col in cols_order[3:]:
            if col in max_cols:
                best_scores[col] = get_max_scores(scores, dname, col)
            elif col in min_cols:
                best_scores[col] = get_min_scores(scores, dname, col)

        n_data = scores[dname][list(scores[dname].keys())[1]]["n_data"]
        dnaming = dname.capitalize() + f" ({n_data})"
        table += f"\\multirow{{{n_models}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{dnaming}}}}} & "

        # line result for original
        table += "& Original & "
        for col in cols_order[3:]:
            val = scores[dname]["Original"][col]
            if val == None:
                table += "- & "
            elif col in best_scores and val == best_scores[col]:
                table += f"\\textbf{{{val:.2f}}} & "
            else:
                table += f"{val:.2f} & "
        table = table[:-2] + "\\\\\n"
        table += f"\\cdashline{{2-{len(cols_order)}}}\n"

        for i, model in enumerate(["synthloop", "promptrefine"]):
            # table += f"\\multirow{{{3}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{llm}}}}} & "
            table += f"& \\multirow{{{7}}}{{*}}{{{METHOD_NAMES[model]}}} & "
            order = model_order if model_order else list(scores[dname].keys())
            order = order[1:]

            for variant in order[i*7:(i+1)*7]:
                model_naming = variant.split("_")[1]
                if model_naming in METHOD_NAMES:
                    model_naming = METHOD_NAMES[model_naming]
                table += f"{model_naming} & "
                for col in cols_order[3:]:
                    val = scores[dname][variant][col]
                    if val == None:
                        table += "- & "
                    elif col in best_scores and val == best_scores[col]:
                        table += f"\\textbf{{{val:.2f}}} & "
                    else:
                        table += f"{val:.2f} & "
                table = table[:-2] + "\\\\\n"
                table += "& & "
            table = table[:-4]
            if i != 1:
                table += f"\\cdashline{{2-{len(cols_order)}}}\n"
        # table = table[:-2] + "\n"
        table += "\\hline\n"
    
    table += "\\end{tabular}"
    return table





if __name__ == "__main__":

    table_cols = ["Dataset", "Model", "Utility TSTR", "Utility Combined", "precision", "recall", "Collisions"]
    # table_cols = ["Dataset", "Model", "Utility TSTR", "Utility Combined", "Detection", "density", "coverage", "Collisions"]
    all_datasets = ["adult", "bank", "german", "sick", "thyroid", "travel"]

    ### table competitors ###
    scores = get_all_scores(["adult", "bank", "german", "thyroid"],
                            "competitors_comparison.csv")
    
    table = create_tex_table(scores,
                             table_cols,
                             dnames_ignored=[])
    
    with open("tex_tables/table_competitors.txt", "w") as f:
        f.write(table)

    
    scores = get_all_scores(["thyroid"],
                            "competitors_comparison_40_training.csv")
    table = create_tex_table(scores,
                             table_cols,
                             dnames_ignored=[])
    with open("tex_tables/table_competitors_40_training.txt", "w") as f:
        f.write(table)

    
    ### table llm comparison ###
    scores = get_all_scores(["adult", "bank", "thyroid"],
                            "llm_big_comparison.csv")
    
    table = create_tex_table_llm(scores,
                             ["Dataset", "LLM", "Model", "Utility TSTR", "Utility Combined", "precision", "recall", "Collisions"],
                             dnames_ignored=[],
                             model_order=["Original", "synthloop_llama3.1", "reduced_llama3.1", "promptrefine_llama3.1",
                                             "synthloop_gpt4o", "reduced_gpt4o", "promptrefine_gpt4o",
                                             "synthloop_deepseek-v3", "reduced_deepseek-v3", "promptrefine_deepseek-v3"])
    
    with open("tex_tables/table_llm_big_comparison.txt", "w") as f:
        f.write(table)
    

    ### iter comparison ###
    for llm in ["llama3.1", "gpt4o-mini", "deepseek-v3"]:

        scores = get_all_scores(all_datasets,
                                f"iter_comparison_{llm}.csv")
        
        table = create_tex_table(scores,
                                table_cols,
                                dnames_ignored=[])
        
        with open(f"tex_tables/table_iter_comparison_{llm}.txt", "w") as f:
            f.write(table)

    ### variant comparison ###
    for model in ["synthloop", "reduced", "promptrefine"]:
        scores = get_all_scores(["adult", "travel"],
                                f"{model}_variants_comparison.csv")
        
        table = create_tex_table(scores,
                                 table_cols,
                                 dnames_ignored=[])
        
        with open(f"tex_tables/table_{model}_variants_comparison.txt", "w") as f:
            f.write(table)

    scores = get_all_scores(["adult", "thyroid"],
                            f"synthloop-refine_variants_comparison.csv")
    
    table = create_tex_table_variants(scores,
                                      ["Dataset", "Model", "Variant", "Utility TSTR", "Utility Combined", "precision", "recall", "Collisions"],
                                      dnames_ignored=[],
                                      model_order=["Original",
                                                   "synthloop_baseline", "synthloop_temp. 0.9", "synthloop_30 shots", "synthloop_cat first", "synthloop_num first", "synthloop_fshots feedback", "synthloop_sentence",
                                                   "promptrefine_baseline", "promptrefine_temp. 0.9", "promptrefine_30 shots", "promptrefine_cat first", "promptrefine_num first", "promptrefine_fshots feedback", "promptrefine_sentence"])

    with open("tex_tables/table_synthloop-refine_variants_comparison.txt", "w") as f:
        f.write(table)


    ### info weakness comparison ###
    for model in ["synthloop", "reduced", "promptrefine"]:
        scores = get_all_scores(["adult", "thyroid", "travel"],
                                f"{model}_info-weakness_comparison.csv")
        
        table = create_tex_table(scores,
                                 table_cols,
                                 dnames_ignored=[])
        
        with open(f"tex_tables/table_{model}_info-weakness_comparison.txt", "w") as f:
            f.write(table)

    scores = get_all_scores(["adult", "thyroid"],
                            f"synthloop-refine_info-weakness_comparison.csv")
    
    table = create_tex_table_infoweakness(scores,
                                          ["Dataset", "Model", "Variant", "Utility TSTR", "Utility Combined", "precision", "recall", "Collisions"],
                                          dnames_ignored=[],
                                          model_order=["Original",
                                                       "synthloop_info-full", "synthloop_info-weakness", "synthloop_noInfo-full", "synthloop_noInfo-weakness",
                                                       "promptrefine_info-full", "promptrefine_info-weakness", "promptrefine_noInfo-full", "promptrefine_noInfo-weakness"])
    
    with open("tex_tables/table_synthloop-refine_info-weakness_comparison.txt", "w") as f:
        f.write(table)
    














###########################################################################""



    # scores = pd.read_csv("thyroid_results.csv")
    # table = get_table_comp_models_dataset("thyroid", scores, scoring="roc_auc")
    # print(table)

    # scores = pd.read_csv("thyroid_results_synthloop_sentence.csv")
    # table = get_table_comp_models_dataset("adult", scores, scoring="roc_auc",
    #                                       model_order=["original", "synthloop_0", "synthloop_1", "synthloop_2"],
    #                                       model_tex=["Original", "iter 0", "iter 1", "iter 2"],
    #                                       )
    # print(table)


    # "adult_results_synthloop_csv_info_weakness.csv"
    # "thyroid_results_synthloop_csv_info_weakness.csv"
    # "adult_results_synthloop_csv_variants.csv"




    # scores = pd.read_csv("evaluation_gen_data/adult/info_weakness_csv_deepseek_temp0.9_shots30_comparison.csv")

    # fake_names = ["original",
    #               "info-full",
    #               "info-full-fs_fb",
    #               "info-weakness",
    #               "info-weakness-fs_fb",
    #               "no_info-full",
    #               "no_info-full-fs_fb",
    #               "no_info-weakness",
    #               "no_info-weakness-fs_fb"]
    
    # names = ["Original",
    #          "Info - Full",
    #          "Info - Full - FS Fb",
    #          "Info - Weakness",
    #          "Info - Weakness - FS Fb",
    #          "No Info - Full",
    #          "No Info - Full - FS Fb",
    #          "No Info - Weakness",
    #          "No Info - Weakness - FS Fb"]

    # table = get_table_comp_models_dataset(scores, scoring="roc_auc",
    #                                       model_order=fake_names,
    #                                       model_tex=names,)
    # print(table)

