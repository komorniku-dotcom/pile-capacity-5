import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="Pile Geomechanical Model V10", layout="wide")
st.title("Pile Geomechanical Model - Soil Cross Section & Load Test Comparison")
st.markdown("---")

SOIL_COLORS = {
    "Fill": "#d5d8dc",
    "Silt": "#d4efdf",
    "Marl": "#fcf3cf",
    "Loose Sand": "#f9e79f",
    "Dense Sand": "#f5b041",
    "Clay": "#d98880",
    "Kurkar": "#d7bde2",
    "Rock": "#aeb6bf"
}

def calc_shear_stress(w, t_peak, t_res, w_p, w_r):
    if w <= 1e-9 or t_peak <= 0:
        return 0.0, "No Displacement"
    if w <= w_p:
        return (t_peak / w_p) * w, "Elastic"
    if w <= w_r:
        slope = (t_peak - t_res) / max((w_r - w_p), 1e-6)
        return t_peak - slope * (w - w_p), "Softening"
    return t_res, "Residual"

def calc_base_load(w_base_m, base_model_type, A, B_parab, q_ult_hyp, k_init_hyp, base_table_df):
    if w_base_m <= 1e-9:
        return 0.0
    if base_model_type == "Parabolic (B * sqrt(w))":
        return B_parab * np.sqrt(w_base_m)
    if base_model_type == "Hyperbolic (Classic Load Test)":
        q_b = w_base_m / ((1.0 / max(k_init_hyp, 1.0)) + (w_base_m / max(q_ult_hyp, 1.0)))
        return q_b * A
    df_sorted = base_table_df.dropna().sort_values(by="Base Settlement (mm)")
    w_mm_arr = df_sorted.get("Base Settlement (mm)").values
    p_kn_arr = df_sorted.get("Base Load (kN)").values
    return float(np.interp(w_base_m * 1000.0, w_mm_arr, p_kn_arr))

def run_pile_model_from_base(layers_df, E, D, w_base_m, base_model_type, B_parab, q_ult_hyp, k_init_hyp, base_table_df):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    num_layers = len(layers_df)
    depths = [0.0] + layers_df.get("Bottom Depth (m)").tolist()

    w_total_node = np.zeros(num_layers + 1)
    w_elastic_node = np.zeros(num_layers + 1)
    P_node = np.zeros(num_layers + 1)

    t_peak_seg = np.zeros(num_layers)
    t_res_seg = np.zeros(num_layers)
    t_mob_seg = np.zeros(num_layers)
    F_fric_seg = np.zeros(num_layers)
    dw_elastic_seg = np.zeros(num_layers)
    state_seg = [""] * num_layers

    w_total_node[num_layers] = w_base_m
    w_elastic_node[num_layers] = 0.0
    P_node[num_layers] = calc_base_load(
        w_base_m, base_model_type, A, B_parab, q_ult_hyp, k_init_hyp, base_table_df
    )

    for i in range(num_layers - 1, -1, -1):
        dz = max(depths[i + 1] - depths[i], 0.01)
        sig_h = layers_df.at[i, "Initial Pressuremeter σ'h (kPa)"]
        d_peak = layers_df.at[i, "Peak Friction δ_peak (°)"]
        w_p = layers_df.at[i, "Peak Disp w_peak (mm)"] / 1000.0
        d_res = layers_df.at[i, "Residual Friction δ_res (°)"]
        w_r = layers_df.at[i, "Residual Disp w_res (mm)"] / 1000.0

        t_p = sig_h * np.tan(np.radians(d_peak))
        t_r = sig_h * np.tan(np.radians(d_res))

        w_top_est = w_total_node[i + 1] + (P_node[i + 1] * dz) / (E * A)
        w_mid = 0.5 * (w_total_node[i + 1] + w_top_est)

        t_mob, state = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
        F_fric = U * dz * t_mob

        P_node[i] = P_node[i + 1] + F_fric
        P_mean = 0.5 * (P_node[i] + P_node[i + 1])

        dw_el = (P_mean * dz) / (E * A)
        dw_elastic_seg[i] = dw_el
        w_elastic_node[i] = w_elastic_node[i + 1] + dw_el
        w_total_node[i] = w_total_node[i + 1] + dw_el

        t_peak_seg[i] = t_p
        t_res_seg[i] = t_r
        t_mob_seg[i] = t_mob
        F_fric_seg[i] = F_fric
        state_seg[i] = state

    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_fric_seg)
    q_base = P_node[-1] / A

    return (
        depths, w_total_node, w_elastic_node, w_fric_node, P_node,
        t_peak_seg, t_res_seg, t_mob_seg, F_fric_seg, cum_fric_seg,
        dw_elastic_seg, q_base, state_seg
    )

def run_pile_model_partial_load(layers_df, E, D, target_P0):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    depths = [0.0] + layers_df.get("Bottom Depth (m)").tolist()
    n_s = len(layers_df)

    if target_P0 <= 1e-6:
        return (
            depths, np.zeros(n_s + 1), np.zeros(n_s + 1), np.zeros(n_s + 1), np.zeros(n_s + 1),
            np.zeros(n_s), np.zeros(n_s), np.zeros(n_s), np.zeros(n_s), np.zeros(n_s),
            np.zeros(n_s), 0.0, ["No Displacement"] * n_s
        )

    low_w, high_w = 0.0, 0.2
    for _ in range(60):
        mid_w = 0.5 * (low_w + high_w)
        w_curr = mid_w
        P_curr = target_P0
        for idx in range(n_s):
            dz_s = max(depths[idx + 1] - depths[idx], 0.01)
            sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
            t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"]))
            t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"]))
            w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
            w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0

            tm, _ = calc_shear_stress(w_curr, t_p, t_r, w_p, w_r)
            ff = U * dz_s * tm
            P_next = max(P_curr - ff, 0.0)
            P_avg = 0.5 * (P_curr + P_next)
            w_curr = max(w_curr - (P_avg * dz_s) / (E * A), 0.0)
            P_curr = P_next

        if P_curr > 0:
            low_w = mid_w
        else:
            high_w = mid_w

    w_total_node = np.zeros(n_s + 1)
    P_node = np.zeros(n_s + 1)
    t_peak_seg = np.zeros(n_s)
    t_res_seg = np.zeros(n_s)
    t_mob_seg = np.zeros(n_s)
    F_fric_seg = np.zeros(n_s)
    dw_elastic_seg = np.zeros(n_s)
    state_seg = [""] * n_s

    np.put(w_total_node, 0, 0.5 * (low_w + high_w))
    np.put(P_node, 0, target_P0)

    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"]))
        t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"]))
        w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
        w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0

        if P_node[idx] <= 1e-6:
            tm, st_txt = 0.0, "No Displacement"
            ff = 0.0
            P_node[idx + 1] = 0.0
            dw_el = 0.0
            w_total_node[idx + 1] = 0.0
        else:
            tm, st_txt = calc_shear_stress(w_total_node[idx], t_p, t_r, w_p, w_r)
            ff_max = U * dz_s * tm
            ff = min(P_node[idx], ff_max)
            if ff < ff_max and P_node[idx] > 0:
                tm = ff / (U * dz_s)
            P_node[idx + 1] = max(P_node[idx] - ff, 0.0)
            P_avg = 0.5 * (P_node[idx] + P_node[idx + 1])
            dw_el = (P_avg * dz_s) / (E * A)
            w_total_node[idx + 1] = max(w_total_node[idx] - dw_el, 0.0)

        t_peak_seg[idx] = t_p
        t_res_seg[idx] = t_r
        t_mob_seg[idx] = tm
        F_fric_seg[idx] = ff
        dw_elastic_seg[idx] = dw_el
        state_seg[idx] = st_txt

    w_elastic_node = np.append(np.cumsum(dw_elastic_seg[::-1])[::-1], 0.0)
    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_fric_seg)

    return (
        depths, w_total_node, w_elastic_node, w_fric_node, P_node,
        t_peak_seg, t_res_seg, t_mob_seg, F_fric_seg, cum_fric_seg,
        dw_elastic_seg, 0.0, state_seg
    )

def plot_pile_soil_cross_section(layers_df, depths, D, P_node, t_mob_seg, state_seg, q_base):
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    max_depth = depths[-1]
    fallback_colors = ["#fdebd0", "#fad7a0", "#f5cba7", "#edbb99", "#e59866", "#d5dbdb"]

    for i in range(len(layers_df)):
        z_top, z_bot = depths[i], depths[i + 1]
        dz = z_bot - z_top
        s_type = str(layers_df.at[i, "Soil Type"]).strip()
        sig_h = layers_df.at[i, "Initial Pressuremeter σ'h (kPa)"]

        color = SOIL_COLORS.get(s_type, fallback_colors[i % len(fallback_colors)])
        for key, val in SOIL_COLORS.items():
            if key.lower() in s_type.lower():
                color = val
                break

        rect_soil = patches.Rectangle((-3.0, z_top), 6.0, dz, facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.9)
        ax.add_patch(rect_soil)

        z_mid = 0.5 * (z_top + z_bot)
        ax.text(-2.85, z_mid, f"{s_type}\nσ'h = {sig_h:.0f} kPa", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#2c3e50")
        ax.text(2.85, z_mid, f"t = {t_mob_seg[i]:.1f} kPa\n[{state_seg[i]}]", va="center", ha="right", fontsize=8, color="#1a5276")

    pile_half_w = 0.45
    rect_pile = patches.Rectangle((-pile_half_w, 0.0), pile_half_w * 2, max_depth, facecolor="#7f8c8d", edgecolor="#2c3e50", linewidth=2, zorder=4)
    ax.add_patch(rect_pile)

    p_head_val = float(P_node.item(0))
    ax.annotate(
        f"P0 = {p_head_val:,.0f} kN",
        xy=(0, 0), xytext=(0, -max_depth * 0.12),
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#c0392b",
        arrowprops=dict(facecolor="#c0392b", edgecolor="#c0392b", width=2.5, headwidth=8)
    )

    ax.annotate(
        f"P_base = {P_node[-1]:,.0f} kN (q = {q_base:,.0f} kPa)",
        xy=(0, max_depth), xytext=(0, max_depth + max_depth * 0.13),
        ha="center", va="top", fontsize=8.5, fontweight="bold", color="#117864",
        arrowprops=dict(facecolor="#117864", edgecolor="#117864", width=2, headwidth=7)
    )

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(max_depth + max_depth * 0.18, -max_depth * 0.18)
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Pile Cross-Section (D = {D}m) & Soil Layers", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    return fig

def build_preset_df(preset_name, f_peak, f_res):
    cols = [
        "Bottom Depth (m)",
        "Soil Type",
        "Initial Pressuremeter σ'h (kPa)",
        "Peak Friction δ_peak (°)",
        "Peak Disp w_peak (mm)",
