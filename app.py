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
    w_mm_arr = df_sorted["Base Settlement (mm)"].values
    p_kn_arr = df_sorted["Base Load (kN)"].values
    return float(np.interp(w_base_m * 1000.0, w_mm_arr, p_kn_arr))

def run_pile_model_from_base(layers_df, E, D, w_base_m, base_model_type, B_parab, q_ult_hyp, k_init_hyp, base_table_df):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    num_layers = len(layers_df)
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()

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
        sig_h = layers_df.loc[i, "Initial Pressuremeter σ'h (kPa)"]
        d_peak = layers_df.loc[i, "Peak Friction δ_peak (°)"]
        w_p = layers_df.loc[i, "Peak Disp w_peak (mm)"] / 1000.0
        d_res = layers_df.loc[i, "Residual Friction δ_res (°)"]
        w_r = layers_df.loc[i, "Residual Disp w_res (mm)"] / 1000.0

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
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    n_s = len(layers_df)

    if target_P0 <= 1e-6:
        return (
            depths, np.zeros(n_s + 1), np.zeros(n_s + 1), np.zeros(n_s + 1), np.zeros(n_s + 1),
            np.zeros(n_s), np.zeros(n_s), np.zeros(n_s), np.zeros(n_s), np.zeros(n_s),
            np.zeros(n_s), 0.0, ["No Displacement"] * n_s
        )

    low_w, high_w = 0.0, 0.1
    for _ in range(55):
        mid_w = 0.5 * (low_w + high_w)
        w_curr = mid_w
        P_curr = target_P0
        for idx in range(n_s):
            dz_s = max(depths[idx + 1] - depths[idx], 0.01)
            sig_h = layers_df.loc[idx, "Initial Pressuremeter σ'h (kPa)"]
            t_p = sig_h * np.tan(np.radians(layers_df.loc[idx, "Peak Friction δ_peak (°)"]))
            t_r = sig_h * np.tan(np.radians(layers_df.loc[idx, "Residual Friction δ_res (°)"]))
            w_p = layers_df.loc[idx, "Peak Disp w_peak (mm)"] / 1000.0
            w_r = layers_df.loc[idx, "Residual Disp w_res (mm)"] / 1000.0
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

    w_total_node[0] = 0.5 * (low_w + high_w)
    P_node[0] = target_P0

    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.loc[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.loc[idx, "Peak Friction δ_peak (°)"]))
        t_r = sig_h * np.tan(np.radians(layers_df.loc[idx, "Residual Friction δ_res (°)"]))
        w_p = layers_df.loc[idx, "Peak Disp w_peak (mm)"] / 1000.0
        w_r = layers_df.loc[idx, "Residual Disp w_res (mm)"] / 1000.0

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
        s_type = str(layers_df.loc[i, "Soil Type"]).strip()
        sig_h = layers_df.loc[i, "Initial Pressuremeter σ'h (kPa)"]

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

    ax.annotate(
        f"P0 = {P_node[0]:,.0f} kN",
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
    presets = {
        "Multi-layer (Fill, Silt, Marl, Clay, Dense Sand)": pd.DataFrame({
            "Bottom Depth (m)": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
"Soil Type": ["Fill", "Fill", "Silt", "Silt", "Marl", "Marl", "Clay", "Dense Sand", "Dense Sand", "Dense Sand"],
"Initial Pressuremeter σ'h (kPa)": [40.0, 45.0, 55.0, 65.0, 85.0, 95.0, 105.0, 130.0, 145.0, 160.0],
"Peak Friction δ_peak (°)": [round(x * f_peak, 1) for x in [25.0, 25.0, 27.0, 27.0, 29.0, 29.0, 24.0, 35.8, 35.8, 35.8]],
"Peak Disp w_peak (mm)": [4.0, 4.0, 3.5, 3.5, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
"Residual Friction δ_res (°)": [round(x * f_res, 1) for x in [24.0, 24.0, 23.0, 23.0, 22.0, 22.0, 16.0, 30.1, 30.1, 30.1]],
"Residual Disp w_res (mm)": [9.0, 9.0, 8.5, 8.5, 9.0, 9.0, 10.0, 6.0, 6.0, 6.0]
}),
"Dense Sand": pd.DataFrame({
"Bottom Depth (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Dense Sand"] * 10,
"Initial Pressuremeter σ'h (kPa)": [80.0 + i * 10.0 for i in range(10)],
"Peak Friction δ_peak (°)": [round(35.8 * f_peak, 1)] * 10,
"Peak Disp w_peak (mm)": [2.0] * 10,
"Residual Friction δ_res (°)": [round(30.1 * f_res, 1)] * 10,
"Residual Disp w_res (mm)": [6.0] * 10
}),
"Loose Sand": pd.DataFrame({
"Bottom Depth (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Loose Sand"] * 10,
"Initial Pressuremeter σ'h (kPa)": [50.0 + i * 5.0 for i in range(10)],
"Peak Friction δ_peak (°)": [round(28.0 * f_peak, 1)] * 10,
"Peak Disp w_peak (mm)": [3.5] * 10,
"Residual Friction δ_res (°)": [round(26.5 * f_res, 1)] * 10,
"Residual Disp w_res (mm)": [8.0] * 10
}),
"Marl": pd.DataFrame({
"Bottom Depth (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Marl"] * 10,
"Initial Pressuremeter σ'h (kPa)": [75.0 + i * 8.0 for i in range(10)],
"Peak Friction δ_peak (°)": [round(29.0 * f_peak, 1)] * 10,
"Peak Disp w_peak (mm)": [3.0] * 10,
"Residual Friction δ_res (°)": [round(22.0 * f_res, 1)] * 10,
"Residual Disp w_res (mm)": [9.0] * 10
}),
"Silt": pd.DataFrame({
"Bottom Depth (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Silt"] * 10,
"Initial Pressuremeter σ'h (kPa)": [55.0 + i * 6.0 for i in range(10)],
"Peak Friction δ_peak (°)": [round(27.0 * f_peak, 1)] * 10,
"Peak Disp w_peak (mm)": [3.5] * 10,
"Residual Friction δ_res (°)": [round(23.0 * f_res, 1)] * 10,
"Residual Disp w_res (mm)": [8.5] * 10
}),
"Clay": pd.DataFrame({
"Bottom Depth (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Clay"] * 10,
"Initial Pressuremeter σ'h (kPa)": [60.0 + i * 5.0 for i in range(10)],
"Peak Friction δ_peak (°)": [round(24.0 * f_peak, 1)] * 10,
"Peak Disp w_peak (mm)": [3.0] * 10,
"Residual Friction δ_res (°)": [round(16.0 * f_res, 1)] * 10,
"Residual Disp w_res (mm)": [10.0] * 10
})
}
return presets[preset_name]

# Sidebar Inputs

st.sidebar.header("1. Pile Properties & Soil Preset")
pile_mat = st.sidebar.selectbox("Pile Material", ["Concrete", "Steel", "HDPE Plastic", "Wood"])
default_E = {"Concrete": 30000000, "Steel": 210000000, "HDPE Plastic": 1500000, "Wood": 11000000}[pile_mat]
E = st.sidebar.number_input("Elastic Modulus E (kPa)", value=default_E, step=1000000)
D = st.sidebar.number_input("Pile Diameter D (m)", min_value=0.2, max_value=3.0, value=0.6, step=0.05)
soil_preset = st.sidebar.selectbox(
"Soil Profile Preset",
[
"Multi-layer (Fill, Silt, Marl, Clay, Dense Sand)",
"Dense Sand",
"Loose Sand",
"Clay",
"Marl",
"Silt"
]
)
f_peak, f_res = {"Concrete": (1.0, 1.0), "Steel": (0.67, 0.76), "HDPE Plastic": (0.45, 0.55), "Wood": (0.85, 0.85)}[pile_mat]
default_layers = build_preset_df(soil_preset, f_peak, f_res)
st.sidebar.markdown("---")
st.sidebar.header("2. Load Range & Base Resistance")
P_max_user = st.sidebar.number_input(
"Max Load for Curve P_max (kN):",
min_value=100.0, max_value=50000.0, value=2500.0, step=100.0
)
base_model_type = st.sidebar.radio(
"Base Resistance Model:",
["Hyperbolic (Classic Load Test)", "Parabolic (B * sqrt(w))", "Custom Load Test Table"]
)
B_parab, q_ult_hyp, k_init_hyp = 1500.0, 3500.0, 150000.0
default_base_table = pd.DataFrame({
"Base Settlement (mm)": [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
"Base Load (kN)": [0.0, 180.0, 380.0, 620.0, 900.0, 1150.0, 1300.0]
})
if base_model_type == "Hyperbolic (Classic Load Test)":
    q_ult_hyp = st.sidebar.number_input("Ultimate Base Stress q_ult (kPa)", value=3500.0, step=250.0)
    k_init_hyp = st.sidebar.number_input("Initial Base Stiffness k_init (kPa/m)", value=150000.0, step=10000.0)
elif base_model_type == "Parabolic (B * sqrt(w))":
    B_parab = st.sidebar.number_input("Parabolic Base Factor B (kN/m^0.5)", value=1500.0, step=100.0)
else:
    st.sidebar.write("Edit Base Load Test Table:")
    default_base_table = st.sidebar.data_editor(default_base_table, num_rows="dynamic", hide_index=True)

# Main Panel

st.subheader("1. Soil Layers Input Table (Editable)")
edited_df = st.data_editor(
default_layers,
num_rows="dynamic",
use_container_width=True,
hide_index=True,
column_config={
"Soil Type": st.column_config.SelectboxColumn(
"Soil Type",
options=list(SOIL_COLORS.keys()),
required=True
)
}
)
with st.expander("Optional: Field Load Test Data for Comparison on Graph"):
    field_test_df = st.data_editor(
    pd.DataFrame({
    "Measured Head Load P0 (kN)": [0.0, 400.0, 800.0, 1200.0, 1600.0],
    "Measured Head Settlement w0 (mm)": [0.0, 1.2, 3.1, 6.5, 14.0]
    }),
    num_rows="dynamic", use_container_width=True, hide_index=True
    )
layers_df = edited_df.dropna(subset=["Bottom Depth (m)", "Initial Pressuremeter σ'h (kPa)"]).sort_values(by="Bottom Depth (m)").reset_index(drop=True)
if len(layers_df) > 0:
    res_zero_base = run_pile_model_from_base(layers_df, E, D, 0.0, base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table)
    F_fric_at_zero_base = float(res_zero_base[4][0])
    P0_list, w0_tot_list, w0_el_list, w0_fr_list, Pbase_list, cum_fric_list = [], [], [], [], [], []
if F_fric_at_zero_base > 0:
    partial_loads = np.linspace(0.0, min(F_fric_at_zero_base, P_max_user), 35)
for p_val in partial_loads:
    _, wt, we, wf, Pn, _, _, _, _, cf, _, _, _ = run_pile_model_partial_load(layers_df, E, D, p_val)
    P0_list.append(float(Pn[0]))
    w0_tot_list.append(float(wt[0] * 1000.0))
    w0_el_list.append(float(we[0] * 1000.0))
    w0_fr_list.append(float(wf[0] * 1000.0))
    Pbase_list.append(0.0)
    cum_fric_list.append(float(cf[-1]))
if P_max_user > F_fric_at_zero_base:
    wb = 0.0002
    step_wb = 0.0005
for _ in range(300):
    _, wt, we, wf, Pn, _, _, _, _, cf, _, _, _ = run_pile_model_from_base(
    layers_df, E, D, wb, base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table
    )
    p_curr = float(Pn[0])
    P0_list.append(p_curr)
    w0_tot_list.append(float(wt[0] * 1000.0))
    w0_el_list.append(float(we[0] * 1000.0))
    w0_fr_list.append(float(wf[0] * 1000.0))
    Pbase_list.append(float(Pn[-1]))
    cum_fric_list.append(float(cf[-1]))
if p_curr >= P_max_user or wb >= 0.5:
    break
    wb += step_wb
if wb > 0.02:
    step_wb = 0.002
if wb > 0.10:
    step_wb = 0.005
    st.markdown("---")
    st.subheader("2. Pile Performance Evaluation")
    target_P0 = st.slider(
    "Select Head Load P0 (kN) [10 kN ≈ 1 ton]:",
    min_value=0.0,
    max_value=float(P_max_user),
    value=min(round(F_fric_at_zero_base * 0.9, 1), float(P_max_user)),
    step=10.0
    )
if target_P0 <= F_fric_at_zero_base:
    (depths, w_total_node, w_elastic_node, w_fric_node, P_node,
    t_peak_seg, t_res_seg, t_mob_seg, F_fric_seg, cum_fric_seg,
    dw_elastic_seg, q_base, state_seg) = run_pile_model_partial_load(layers_df, E, D, target_P0)
else:
    low_wb, high_wb = 0.0, 0.5
for _ in range(50):
    mid_wb = 0.5 * (low_wb + high_wb)
    res_tmp = run_pile_model_from_base(layers_df, E, D, mid_wb, base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table)
if res_tmp[4][0] < target_P0:
    low_wb = mid_wb
else:
    high_wb = mid_wb
    (depths, w_total_node, w_elastic_node, w_fric_node, P_node,
    t_peak_seg, t_res_seg, t_mob_seg, F_fric_seg, cum_fric_seg,
    dw_elastic_seg, q_base, state_seg) = run_pile_model_from_base(
    layers_df, E, D, 0.5 * (low_wb + high_wb), base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Head Load P0", f"{P_node[0]:,.1f} kN", f"{P_node[0]/10.0:,.1f} ton")
    m2.metric("Total Head Disp w0", f"{w_total_node[0]*1000:.2f} mm")
    m3.metric("Base Load P_base", f"{P_node[-1]:,.1f} kN", f"q_base: {q_base:,.0f} kPa")
    m4.metric("Max Friction (q_base=0)", f"{F_fric_at_zero_base:,.1f} kN")
    st.info(
    f"Head Displacement Breakdown: "
    f"Elastic = *{w_elastic_node[0]1000:.2f} mm | "
    f"Friction/Soil = *{w_fric_node[0]1000:.2f} mm | "
    f"Total = *{w_total_node[0]1000:.2f} mm"
    )
    c_vis, c_curve = st.columns([1.0, 1.45])
    with c_vis:
    fig_soil = plot_pile_soil_cross_section(layers_df, depths, D, P_node, t_mob_seg, state_seg, q_base)
    st.pyplot(fig_soil)
    with c_curve:
    fig_ls, ax_a = plt.subplots(figsize=(6.8, 5.2))
    ax_a.plot(w0_tot_list, P0_list, label="Model: Total Disp (w_total)", color="#1f497d", linewidth=2.4)
    ax_a.plot(w0_el_list, P0_list, "--", label="Model: Elastic Shortening (w_elastic)", color="#2e75b6")
    ax_a.plot(w0_fr_list, P0_list, "-.", label="Model: Friction/Base Disp (w_fric)", color="#e67e22")
    clean_field = field_test_df.dropna()
if len(clean_field) > 0:
    ax_a.plot(
    clean_field["Measured Head Settlement w0 (mm)"],
    clean_field["Measured Head Load P0 (kN)"],
    "o-", color="#8e44ad", linewidth=1.8, markersize=6, label="Field Load Test (Measured)"
    )
    ax_a.axhline(F_fric_at_zero_base, color="green", linestyle=":", label=f"Friction Limit q_base=0 ({F_fric_at_zero_base:.0f} kN)")
    ax_a.scatter([w_total_node[0] * 1000.0], [P_node[0]], color="red", s=80, zorder=6, label="Current Evaluation Point")
    ax_a.set_ylim(0, P_max_user * 1.05)
    ax_a.set_xlabel("Pile Head Settlement w0 (mm)")
    ax_a.set_ylabel("Pile Head Load P0 (kN)")
    ax_a.set_title(f"Load-Settlement Curve (0 to {P_max_user:,.0f} kN)")
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_ls)
    tab1, tab2 = st.tabs(["Depth Profiles (Disp, Shear, Axial Load)", "Detailed Layer Table & CSV Export"])
    with tab1:
    fig_prof, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.3))
    mid_depths = [0.5 * (depths[i] + depths[i + 1]) for i in range(len(layers_df))]
    ax1.plot(w_total_node * 1000.0, depths, marker="o", label="Total", color="#1f497d")
    ax1.plot(w_elastic_node * 1000.0, depths, "--", label="Elastic", color="#2e75b6")
    ax1.plot(w_fric_node * 1000.0, depths, "-.", label="Friction/Base", color="#e67e22")
    ax1.set_title("Displacement Profile (mm)")
    ax1.set_xlabel("Displacement (mm)")
    ax1.set_ylabel("Depth (m)")
    ax1.invert_yaxis()
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2.plot(t_peak_seg, mid_depths, "--", label="Peak", color="gray")
    ax2.plot(t_res_seg, mid_depths, ":", label="Residual", color="red")
    ax2.plot(t_mob_seg, mid_depths, marker="s", label="Mobilized", color="#27ae60")
    ax2.set_title("Shear Stress Profile (kPa)")
    ax2.set_xlabel("Shear Stress (kPa)")
    ax2.invert_yaxis()
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax3.plot(P_node, depths, marker="^", color="#8e44ad")
    ax3.set_title("Axial Load Distribution (kN)")
    ax3.set_xlabel("Axial Load (kN)")
    ax3.invert_yaxis()
    ax3.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_prof)
    with tab2:
    results_df = pd.DataFrame({
    "Layer": range(1, len(layers_df) + 1),
    "Soil Type": layers_df["Soil Type"],
    "Depth (m)": [f"{depths[i]:.1f} - {depths[i+1]:.1f}" for i in range(len(layers_df))],
    "Total Disp (mm)": w_total_node[:-1] * 1000.0,
    "Elastic Shortening (mm)": dw_elastic_seg * 1000.0,
    "Friction Disp (mm)": w_fric_node[:-1] * 1000.0,
    "Peak Shear (kPa)": t_peak_seg,
    "Residual Shear (kPa)": t_res_seg,
    "Mobilized Shear (kPa)": t_mob_seg,
    "Layer Friction (kN)": F_fric_seg,
    "Axial Load at Top (kN)": P_node[:-1],
    "State": state_seg
    })
    st.dataframe(results_df.style.format({
    "Total Disp (mm)": "{:.2f}",
    "Elastic Shortening (mm)": "{:.2f}",
    "Friction Disp (mm)": "{:.2f}",
    "Peak Shear (kPa)": "{:.1f}",
    "Residual Shear (kPa)": "{:.1f}",
    "Mobilized Shear (kPa)": "{:.1f}",
    "Layer Friction (kN)": "{:.1f}",
    "Axial Load at Top (kN)": "{:,.1f}"
    }), use_container_width=True, hide_index=True)
    csv_out = results_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
    label="Download Full Results Report (CSV)",
    data=csv_out,
    file_name=f"pile_analysis_V10_D{D}.csv",
    mime="text/csv"
    )
