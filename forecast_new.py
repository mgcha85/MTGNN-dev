# forecast_unified.py
import ast
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from safetensors.torch import load_file
from scipy.sparse import linalg  # noqa: F401 (for parity with original imports)

from net import gtnet
from config import NET_ROOT

plt.rcParams['savefig.dpi'] = 1200
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_random_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def map_name_to_abbreviation(name: str):
    """Optional abbreviation mapping (abb.txt: 'ABB: Full Name')"""
    abbr_path = os.path.join(os.getcwd(), "abb.txt")
    if not os.path.exists(abbr_path):
        return name
    abbreviation_dict = {}
    with open(abbr_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or ": " not in line:
                continue
            abbreviation, full_name = map(str.strip, line.split(': ', 1))
            abbreviation_dict[full_name] = abbreviation
    return abbreviation_dict.get(name, name)

def consistent_name(name: str):
    """Normalize names for plotting/saving consistency."""
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    # special case
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'
    if name in ('CAPTCHA', 'DNSSEC', 'RRAM'):
        return name
    # Title-case if not all caps
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2:
                result += word
            else:
                result += word[0].upper() + word[1:]
            if i < len(words) - 1:
                result += ' '
        return result
    # Mixed handling for all caps with acronyms
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word in ('MITM', 'SIEM'):
            result += word
        else:
            result += word[0] + word[1:].lower()
        if i < len(words) - 1:
            result += ' '
    return result

def exponential_smoothing(series, alpha: float):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result

def create_columns(file_name):
    col_name = []
    col_index = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name = col_name[1:]
        for i, c in enumerate(col_name):
            col_index[c] = i
    return col_name, col_index

def build_graph(file_name):
    graph = defaultdict(list)
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key_node = row[0]
            adjacent_nodes = [node for node in row[1:] if node]
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with', len(graph), 'attacks...')
    return graph

def zero_negative_curves(data, forecast, attack, solutions, index):
    a = data[:, index[attack]]
    f = forecast[:, index[attack]]
    a[a < 0] = 0
    f[f < 0] = 0
    for s in solutions:
        aa = data[:, index[s]]
        ff = forecast[:, index[s]]
        aa[aa < 0] = 0
        ff[ff < 0] = 0
    return data, forecast

def getClosestCurveLarger(c, forecast, confidence, attack, solutions, col):
    d = 999999999
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if col[j] not in solutions and col[j] != attack:
            continue
        if torch.mean(f) <= torch.mean(c):
            continue
        delta = torch.mean(f) - torch.mean(c)
        if delta < d:
            d = delta
            cc = f.clone()
            cc_conf = f_conf.clone()
    return cc, cc_conf

def getClosestCurveSmaller(c, forecast, confidence, attack, solutions, col):
    d = 999999999
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if col[j] not in solutions and col[j] != attack:
            continue
        if torch.mean(f) >= torch.mean(c):
            continue
        delta = torch.abs(torch.mean(f) - torch.mean(c))
        if delta < d:
            d = delta
            cc = f.clone()
            cc_conf = f_conf.clone()
    return cc, cc_conf

def save_data(data, forecast, confidence, variance, col):
    file_dir = os.path.join(NET_ROOT, 'model/Bayesian/forecast/data/')
    ensure_dir(file_dir)
    for i in range(data.shape[1]):
        d = data[:, i]
        f = forecast[:, i]
        c = confidence[:, i]
        v = variance[:, i]
        name = col[i]
        with open(os.path.join(file_dir, name.replace('/', '_') + '.txt'), 'w', encoding='utf-8') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')

def save_gap(out_root, forecast, attack, solutions, index):
    gap_dir = os.path.join(out_root, 'model/Bayesian/forecast/gap/')
    ensure_dir(gap_dir)
    out_path = os.path.join(gap_dir, consistent_name(attack).replace('/', '_') + '_gap.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Solution', '2023', '2024', '2025'])
        table = []
        a = forecast[:, index[attack]].tolist()
        a_reduced = [sum(a[i:i + 12]) / 12 for i in range(0, len(a), 12)]
        for s in solutions:
            row = [consistent_name(s)]
            f = forecast[:, index[s]].tolist()
            f_reduced = [sum(f[i:i + 12]) / 12 for i in range(0, len(f), 12)]
            gap = [x - y for x, y in zip(a_reduced, f_reduced)]
            row.extend(gap)
            table.append(row)
        sorted_table = sorted(table, key=lambda r: sum(r[-3:]))
        for row in sorted_table:
            writer.writerow(row)

def plot_forecast(out_root, data, forecast, confidence, attack, solutions, index, col, alarming=True):
    data, forecast = zero_negative_curves(data, forecast, attack, solutions, index)
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
               "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
               "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
               "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
               "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
               "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]

    plt.style.use("seaborn-dark")
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    # Attack
    counter = 0
    d = torch.cat((data[:, index[attack]], forecast[0:1, index[attack]]), dim=0)
    f = forecast[:, index[attack]]
    c = confidence[:, index[attack]]
    a = consistent_name(attack)
    ax.plot(range(len(d)), d, '-', color=colours[counter], label=a, linewidth=2)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, '-', color=colours[counter], linewidth=2)
    ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=colours[counter], alpha=0.6)
    f_attack = f.clone()
    counter += 1

    # Filter solutions by alarming
    if alarming:
        for s in list(solutions):
            f_s = forecast[:, index[s]]
            if torch.mean(f_s) >= torch.mean(f_attack):
                solutions.remove(s)

    # Solutions
    for s in solutions:
        d = torch.cat((data[:, index[s]], forecast[0:1, index[s]]), dim=0)
        f = forecast[:, index[s]]
        c = confidence[:, index[s]]
        s_name = consistent_name(s)
        ax.plot(range(len(d)), d, '-', color=colours[counter], label=s_name, linewidth=1)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, '-', color=colours[counter], linewidth=1)
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=colours[counter], alpha=0.6)
        if torch.mean(f_attack) > torch.mean(f):
            cc, cc_conf = getClosestCurveLarger(f, forecast, confidence, attack, solutions, col)
            if cc is not None:
                ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), cc - cc_conf, f + c, color=colours[counter], alpha=0.3)
        else:
            cc, cc_conf = getClosestCurveSmaller(f, forecast, confidence, attack, solutions, col)
            if cc is not None:
                ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), cc + cc_conf, f - c, color=colours[counter], alpha=0.3)
        counter += 1

    x = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']
    ax.set_xticks([6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174], x)
    ax.set_ylabel("Trend", fontsize=15)
    plt.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    plt.xticks(rotation=90, fontsize=13)
    ax.set_title(a, y=1.03, fontsize=18)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)

    images_dir = os.path.join(out_root, 'model/Bayesian/forecast/plots/')
    ensure_dir(images_dir)
    plt.savefig(os.path.join(images_dir, a.replace('/', '_') + '.png'), bbox_inches="tight")
    plt.savefig(os.path.join(images_dir, a.replace('/', '_') + ".pdf"), bbox_inches="tight", format='pdf')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def visualise_saliency_map(saliency_map, node_title, out_root=None, save_pdf=True):
    """Visualize a single concatenated saliency heatmap (36 x 10)"""
    smap = saliency_map.reshape(saliency_map.shape[1], saliency_map.shape[2])
    smap = (smap - smap.min()) / (smap.max() - smap.min())
    smap = smap.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(smap, cmap='YlGnBu', interpolation='nearest', aspect='auto')
    plt.colorbar()
    ttl = consistent_name(node_title)
    plt.title(ttl, fontsize=20)
    plt.xlabel('Past 10 timesteps', fontsize=20)
    plt.ylabel('Future 36 timesteps', fontsize=20)
    x_ticks = [f'N-{i}' for i in range(10, 0, -1)]
    y_ticks = [f'N+{i}' for i in range(1, 37)]
    plt.xticks(range(10), x_ticks)
    plt.yticks(range(36), y_ticks)

    if save_pdf and out_root:
        ensure_dir(out_root)
        out_pdf = os.path.join(out_root, ttl + '_SM.pdf')
        plt.savefig(out_pdf, format='pdf', bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# ----------------------------
# Main pipeline (unified)
# ----------------------------
def main(
    explain: bool = True,
    attention_plots: bool = True,   # requires model methods
    saliency_node_name: str | None = None,  # if None, skips saliency; else node name
    alarming: bool = True,
    smoothing_alpha: float = 0.1,
    runs: int = 10,
    P: int = 10,
    out_len: int = 36
):
    set_random_seed(123)

    data_file = f'{NET_ROOT}/data/Smoothed_CyberTrend_Forecasting_All.txt'
    model_file = f'{NET_ROOT}/model/Bayesian/model.safetensors'
    nodes_file = f'{NET_ROOT}/data/Smoothed_CyberTrend_Forecasting_All.csv'
    graph_file = f'{NET_ROOT}/data/graph.csv'
    hp_path = f"{NET_ROOT}/model/Bayesian/hp.txt"

    # Load data
    with open(data_file, 'r', encoding='utf-8') as fin:
        rawdat = np.loadtxt(fin, delimiter='\t')
    n, m = rawdat.shape

    # Columns and graph
    col, index = create_columns(nodes_file)
    graph = build_graph(graph_file)

    # Normalize globally (per-column max)
    scale = np.ones(m)
    dat = np.zeros(rawdat.shape)
    for i in range(m):
        mx = np.max(np.abs(rawdat[:, i]))
        scale[i] = mx if mx != 0 else 1.0
        dat[:, i] = rawdat[:, i] / scale[i]
    scale = torch.from_numpy(scale).to(device=device)

    print('data shape:', dat.shape)

    # Prepare X (last P months)
    X = torch.from_numpy(dat[-P:, :]).unsqueeze(0).unsqueeze(1).transpose(2, 3).to(dtype=torch.float, device=device)
    X.requires_grad = True  # for saliency

    # Load best HP
    with open(hp_path, "r", encoding='utf-8') as f:
        best_hp = ast.literal_eval(f.read())

    (gcn_depth, _lr, conv, res, skip, end,
     k, dropout, dilation_ex, node_dim,
     prop_alpha, tanh_alpha, layer, _) = best_hp

    # Build model and load weights
    model = gtnet(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=gcn_depth,
        num_nodes=m,
        device=device,
        predefined_A=None,
        static_feat=None,
        dropout=dropout,
        subgraph_size=k,
        node_dim=node_dim,
        dilation_exponential=dilation_ex,
        conv_channels=conv,
        residual_channels=res,
        skip_channels=skip,
        end_channels=end,
        seq_length=P,
        in_dim=1,
        out_dim=out_len,
        layers=layer,
        propalpha=prop_alpha,
        tanhalpha=tanh_alpha,
        layer_norm_affline=False
    ).to(device)

    state = load_file(model_file, device=device)
    model.load_state_dict(state)
    model.eval()

    # Monte Carlo forward passes
    outputs = []
    for _ in range(runs):
        with torch.no_grad():
            output = model(X)
            y_pred = output[-1, :, :, -1].clone()  # (out_len x m), original code uses [-1,:,:, -1]
        outputs.append(y_pred)
    outputs = torch.stack(outputs)  # (runs x out_len x m)

    dat_t = torch.from_numpy(dat).to(device=device)
    Y = torch.mean(outputs, dim=0)
    variance = torch.var(outputs, dim=0)
    std_dev = torch.std(outputs, dim=0)
    z = 1.96
    confidence = z * std_dev / torch.sqrt(torch.tensor(runs))

    # de-normalize
    dat_t *= scale
    Y *= scale
    variance *= scale
    confidence *= scale

    print('output shape:', Y.shape)

    # ----------------------------
    # Explainability (optional)
    # ----------------------------
    if explain:
        # Optional attention visualizations if the model implements them
        if attention_plots and hasattr(model, "visualize_attention_scores"):
            # Prepare pretty names / abbreviations
            names = [map_name_to_abbreviation(consistent_name(c)) for c in col]
            attacks_r = list(range(0, 16))
            all_attacks_r = list(range(0, 16)) + list(range(33, 42))
            tech_r = list(range(44, 142))
            attacks_mentions_r = list(range(16, 42))
            wars_r = list(range(42, 43))
            holidays_r = list(range(43, 44))
            try:
                model.visualize_attention_scores(names, all_attacks_r, tech_r, 'Attacks_PATs')
                model.visualize_attention_scores(names, attacks_r, attacks_mentions_r, 'Attacks_Mentions')
                model.visualize_attention_scores(names, all_attacks_r, wars_r, 'Attacks_Wars')
                model.visualize_attention_scores(names, all_attacks_r, holidays_r, 'Attacks_Holidays')
                model.visualize_attention_scores(names, all_attacks_r, all_attacks_r, 'Attacks_Attacks')
                model.visualize_attention_scores(names, tech_r, tech_r, 'PATs_PATs')
            except Exception as e:
                print("[warn] attention visualization skipped:", e)

        # Saliency heatmap for a specific node (36 future x 10 past)
        if saliency_node_name is not None and hasattr(model, "compute_saliency"):
            if saliency_node_name not in index:
                print(f"[warn] saliency node '{saliency_node_name}' not found; skipping saliency.")
            else:
                node_idx = index[saliency_node_name]
                time_saliency_maps = []
                saliency_map_36 = None
                for t in range(0, out_len):
                    try:
                        smap = model.compute_saliency(X, t, node_idx, True)
                    except Exception as e:
                        print("[warn] compute_saliency failed:", e)
                        smap = None
                    if smap is None:
                        time_saliency_maps = []
                        saliency_map_36 = None
                        break
                    time_saliency_maps.append(smap)
                    if saliency_map_36 is None:
                        saliency_map_36 = smap
                    else:
                        saliency_map_36 = torch.cat((saliency_map_36, smap), dim=1)
                if saliency_map_36 is not None:
                    sm_dir = os.path.join(NET_ROOT, "model/Bayesian/forecast/saliency/")
                    ensure_dir(sm_dir)
                    visualise_saliency_map(saliency_map_36, saliency_node_name, out_root=sm_dir, save_pdf=True)

    # ----------------------------
    # Save forecast data, plots, gap tables
    # ----------------------------
    # Save raw data/forecast/confidence/variance per node
    save_data(dat_t, Y, confidence, variance, col)

    # Combine data for normalization to [0,1] buckets for plot aesthetics
    all_full = torch.cat((dat_t, Y), dim=0)

    incident_max = -1e15
    mention_max = -1e15
    for i in range(all_full.shape[0]):
        for j in range(all_full.shape[1]):
            if 'WAR' in col[j] or 'Holiday' in col[j] or j in range(16, 32):
                continue
            if 'Mention' in col[j]:
                if all_full[i, j] > mention_max:
                    mention_max = all_full[i, j]
            else:
                if all_full[i, j] > incident_max:
                    incident_max = all_full[i, j]
    # Fallbacks to avoid division by zero
    if incident_max <= 0:
        incident_max = 1.0
    if mention_max <= 0:
        mention_max = 1.0

    all_n = torch.zeros_like(all_full)
    confidence_n = torch.zeros_like(confidence)
    u = 0
    for i in range(all_full.shape[0]):
        for j in range(all_full.shape[1]):
            if 'Mention' in col[j]:
                all_n[i, j] = all_full[i, j] / mention_max
            else:
                all_n[i, j] = all_full[i, j] / incident_max
            if i >= all_full.shape[0] - out_len:
                # scale confidence proportionally
                denom = all_full[i, j].item()
                ratio = (all_n[i, j].item() / denom) if denom != 0 else 0.0
                confidence_n[u, j] = confidence[u, j] * ratio
        if i >= all_full.shape[0] - out_len:
            u += 1

    smoothed_dat = torch.stack(exponential_smoothing(all_n, smoothing_alpha))
    smoothed_confidence = torch.stack(exponential_smoothing(confidence_n, smoothing_alpha))

    # Per-attack plots + gap tables
    for attack, solutions in graph.items():
        try:
            plot_forecast(NET_ROOT, smoothed_dat[:-out_len, ], smoothed_dat[-out_len:, ], smoothed_confidence, attack, list(solutions), index, col, alarming=alarming)
            save_gap(NET_ROOT, smoothed_dat[-out_len:, ], attack, solutions, index)
        except Exception as e:
            print(f"[warn] plotting/gap for '{attack}' failed:", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Forecast with optional Explainability")
    parser.add_argument("--no-explain", action="store_true", help="Disable explainability (attention/saliency)")
    parser.add_argument("--no-attn", action="store_true", help="Disable attention visualizations (if model supports)")
    parser.add_argument("--saliency-node", type=str, default=None, help="Node name for saliency heatmap (exact column name)")
    parser.add_argument("--no-alarming", action="store_true", help="Do not filter solutions against attack (plot all solutions)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Exponential smoothing alpha")
    parser.add_argument("--runs", type=int, default=10, help="Monte Carlo runs for Bayesian estimation")
    parser.add_argument("--P", type=int, default=10, help="Look-back window")
    parser.add_argument("--out_len", type=int, default=36, help="Forecast horizon")
    args = parser.parse_args()

    main(
        explain=(not args.no_explain),
        attention_plots=(not args.no_attn),
        saliency_node_name=args.saliency_node,
        alarming=(not args.no_alarming),
        smoothing_alpha=args.alpha,
        runs=args.runs,
        P=args.P,
        out_len=args.out_len
    )


## 설명가능성 끄기(기존 forecast.py와 유사 동작):
## python forecast_unified.py --no-explain