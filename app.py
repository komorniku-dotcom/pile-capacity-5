import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pile Geomechanical Model V11", layout="wide")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

st.markdown(
    """
    <style>
    h1, h2, h3, .stSubheader, .prof-header {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .prof-header {
        color: #2e86c1;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    </style>
    <div class="prof-header">Prof. Uri Komornik</div>
    """,
    unsafe_allow_html=True,
)


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
    "Rock": "#aeb6bf",
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


def run_pile_model_partial_load(layers_df, E, D, target_P0):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    n_s = len(layers_df)

    if target_P0 <= 1e-6:
        return (
            depths,
            np.zeros(n_s + 1),
            np.zeros(n_s + 1),
            np.zeros(n_s + 1),
            np.zeros(n_s + 1),
            np.zeros(n_s),
            np.zeros(n_s),
            np.zeros(n_s),
            np.zeros(n_s),
            np.zeros(n_s),
            np.zeros(n_s),
            0.0,
            ["No Displacement"] * n_s,
        )

    low_w, high_w = 0.0, 0.25
    for _ in range(65):
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

            w_bot_est = max(w_curr - (P_curr * dz_s) / (E * A), 0.0)
            w_mid = 0.5 * (w_curr + w_bot_est)
            tm, _ = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
            ff = U * dz_s * tm
            P_next = max(P_curr - ff, 0.0)
            P_avg = 0.5 * (P_curr + P_next)
            w_curr = max(w_curr - (P_avg * dz_s) / (E * A), 0.0)
            P_curr = P_next

        if P_curr > 1e-4:
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
            w_bot_est = max(w_total_node[idx] - (P_node[idx] * dz_s) / (E * A), 0.0)
            w_mid = 0.5 * (w_total_node[idx] + w_bot_est)
            tm, st_txt = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
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
    q_base = float(P_node[-1]) / A

    return (
        depths,
        w_total_node,
        w_elastic_node,
        w_fric_node,
        P_node,
        t_peak_seg,
        t_res_seg,
        t_mob_seg,
        F_fric_seg,
        cum_fric_seg,
        dw_elastic_seg,
        q_base,
        state_seg,
    )


def find_max_friction_load_zero_base(layers_df, E, D):
    U = np.pi * D
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    n_s = len(layers_df)

    max_possible_P = 0.0
    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"]))
        max_possible_P += U * dz_s * t_p

    if max_possible_P <= 0:
        return 0.0

    low_P, high_P = 0.0, max_possible_P * 1.2
    for _ in range(60):
        mid_P = 0.5 * (low_P + high_P)
        res = run_pile_model_partial_load(layers_df, E, D, mid_P)
        P_node_res = res[4]
        if float(P_node_res[-1]) <= 1e-3:
            low_P = mid_P
        else:
            high_P = mid_P

    return low_P


def run_pile_model_from_base(
    layers_df, E, D, w_base_m, base_model_type, B_parab, q_ult_hyp, k_init_hyp, base_table_df
):
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
    q_base = float(P_node[-1]) / A

    return (
        depths,
        w_total_node,
        w_elastic_node,
        w_fric_node,
        P_node,
        t_peak_seg,
        t_res_seg,
        t_mob_seg,
        F_fric_seg,
        cum_fric_seg,
        dw_elastic_seg,
        q_base,
        state_seg,
    )


def plot_pile_soil_cross_section(layers_df, depths, D, P_node, t_mob_seg, state_seg, q_base):
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    max_depth = depths[-1]
    fallback_colors = ("#fdebd0", "#fad7a0", "#f5cba7", "#edbb99", "#e59866", "#d5dbdb")

    for i in range(len(layers_df)):
        z_top, z_bot = depths[i], depths[i + 1]
        dz = z_bot - z_top
        s_type = str(layers_df.at[i, "Soil Type"]).strip()
        sig_h = layers_df.at[i, "Initial Pressuremeter σ'h (kPa)"]

        color = SOIL_COLORS.get(s_type, fallback_colors[i % len(fallback_colors)])
