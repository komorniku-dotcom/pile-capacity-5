import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pile Geomechanical Model - Tension / Uplift", layout="wide")

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

st.title("Pile Geomechanical Model (Tension / Uplift) - Soil Cross Section & Load Test")
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


def run_tension_model_partial_load(layers_df, E, D, gamma_pile, f_tension_opt, target_T0):
    """
    Solves partial uplift load (T_base = 0 and w_base = 0) from head downwards.
    Includes pure physical self-weight dW = gamma_pile * A * dz resisting uplift.
    """
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    n_s = len(layers_df)

    if target_T0 <= 1e-6:
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
            np.zeros(n_s),
            ["No Displacement"] * n_s,
        )

    low_w, high_w = 0.0, 0.25
    for _ in range(65):
        mid_w = 0.5 * (low_w + high_w)
        w_curr = mid_w
        T_curr = target_T0
        for idx in range(n_s):
            dz_s = max(depths[idx + 1] - depths[idx], 0.01)
            sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
            t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"])) * f_tension_opt
            t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"])) * f_tension_opt
            w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
            w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0

            dW = gamma_pile * A * dz_s
            w_bot_est = max(w_curr - (T_curr * dz_s) / (E * A), 0.0)
            w_mid = 0.5 * (w_curr + w_bot_est)
            tm, _ = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
            ff = U * dz_s * tm
            T_next = max(T_curr - ff - dW, 0.0)
            T_avg = 0.5 * (T_curr + T_next)
            w_curr = max(w_curr - (T_avg * dz_s) / (E * A), 0.0)
            T_curr = T_next

        if T_curr > 1e-4:
            low_w = mid_w
        else:
            high_w = mid_w

    w_total_node = np.zeros(n_s + 1)
    T_node = np.zeros(n_s + 1)
    t_peak_seg = np.zeros(n_s)
    t_res_seg = np.zeros(n_s)
    t_mob_seg = np.zeros(n_s)
    F_fric_seg = np.zeros(n_s)
    W_seg = np.zeros(n_s)
    dw_elastic_seg = np.zeros(n_s)
    state_seg = [""] * n_s

    w_total_node[0] = 0.5 * (low_w + high_w)
    T_node[0] = target_T0

    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"])) * f_tension_opt
        t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"])) * f_tension_opt
        w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
        w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0
        dW = gamma_pile * A * dz_s

        if T_node[idx] <= 1e-6:
            tm, st_txt = 0.0, "No Displacement"
            ff = 0.0
            T_node[idx + 1] = 0.0
            dw_el = 0.0
            w_total_node[idx + 1] = 0.0
        else:
            w_bot_est = max(w_total_node[idx] - (T_node[idx] * dz_s) / (E * A), 0.0)
            w_mid = 0.5 * (w_total_node[idx] + w_bot_est)
            tm, st_txt = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
            ff_max = U * dz_s * tm
            rem_after_w = max(T_node[idx] - dW, 0.0)
            ff = min(rem_after_w, ff_max)
            if ff < ff_max and rem_after_w > 0:
                tm = ff / (U * dz_s)
            elif rem_after_w == 0:
                tm = 0.0
                st_txt = "Self-Weight Only"
            T_node[idx + 1] = max(T_node[idx] - ff - dW, 0.0)
            T_avg = 0.5 * (T_node[idx] + T_node[idx + 1])
            dw_el = (T_avg * dz_s) / (E * A)
            w_total_node[idx + 1] = max(w_total_node[idx] - dw_el, 0.0)

        t_peak_seg[idx] = t_p
        t_res_seg[idx] = t_r
        t_mob_seg[idx] = tm
        F_fric_seg[idx] = ff
        W_seg[idx] = dW
        dw_elastic_seg[idx] = dw_el
        state_seg[idx] = st_txt

    w_elastic_node = np.append(np.cumsum(dw_elastic_seg[::-1])[::-1], 0.0)
    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_fric_seg)

    return (
        depths,
        w_total_node,
        w_elastic_node,
        w_fric_node,
        T_node,
        t_peak_seg,
        t_res_seg,
        t_mob_seg,
        F_fric_seg,
        cum_fric_seg,
        dw_elastic_seg,
        W_seg,
        state_seg,
    )


def find_max_uplift_at_zero_base_disp(layers_df, E, D, gamma_pile, f_tension_opt):
    """Finds the threshold head uplift load T0 right before the pile toe begins moving upward (w_base = 0+)."""
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    n_s = len(layers_df)

    max_possible_T = 0.0
    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"])) * f_tension_opt
        max_possible_T += U * dz_s * t_p + gamma_pile * A * dz_s

    if max_possible_T <= 0:
        return 0.0

    low_T, high_T = 0.0, max_possible_T * 1.2
    for _ in range(60):
        mid_T = 0.5 * (low_T + high_T)
        res = run_tension_model_partial_load(layers_df, E, D, gamma_pile, f_tension_opt, mid_T)
        T_node_res = res[4]
        if float(T_node_res[-1]) <= 1e-3:
            low_T = mid_T
        else:
            high_T = mid_T

    return low_T


def run_tension_model_from_base(layers_df, E, D, gamma_pile, f_tension_opt, w_base_up_m):
    """
    Solves full-pile uplift from bottom up for a prescribed upward toe displacement w_base_up_m > 0.
    Boundary condition at z = L: T_base = 0 (no base resistance in tension).
    """
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    num_layers = len(layers_df)
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()

    w_total_node = np.zeros(num_layers + 1)
    w_elastic_node = np.zeros(num_layers + 1)
    T_node = np.zeros(num_layers + 1)

    t_peak_seg = np.zeros(num_layers)
    t_res_seg = np.zeros(num_layers)
    t_mob_seg = np.zeros(num_layers)
    F_fric_seg = np.zeros(num_layers)
    W_seg = np.zeros(num_layers)
    dw_elastic_seg = np.zeros(num_layers)
    state_seg = [""] * num_layers

    w_total_node[num_layers] = w_base_up_m
    w_elastic_node[num_layers] = 0.0
    T_node[num_layers] = 0.0

    for i in range(num_layers - 1, -1, -1):
        dz = max(depths[i + 1] - depths[i], 0.01)
        sig_h = layers_df.at[i, "Initial Pressuremeter σ'h (kPa)"]
        d_peak = layers_df.at[i, "Peak Friction δ_peak (°)"]
        w_p = layers_df.at[i, "Peak Disp w_peak (mm)"] / 1000.0
        d_res = layers_df.at[i, "Residual Friction δ_res (°)"]
        w_r = layers_df.at[i, "Residual Disp w_res (mm)"] / 1000.0

        t_p = sig_h * np.tan(np.radians(d_peak)) * f_tension_opt
        t_r = sig_h * np.tan(np.radians(d_res)) * f_tension_opt
        dW = gamma_pile * A * dz

        w_top_est = w_total_node[i + 1] + (T_node[i + 1] * dz) / (E * A)
        w_mid = 0.5 * (w_total_node[i + 1] + w_top_est)

        t_mob, state = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
        F_fric = U * dz * t_mob

        T_node[i] = T_node[i + 1] + F_fric + dW
        T_mean = 0.5 * (T_node[i] + T_node[i + 1])

        dw_el = (T_mean * dz) / (E * A)
        dw_elastic_seg[i] = dw_el
        w_elastic_node[i] = w_elastic_node[i + 1] + dw_el
        w_total_node[i] = w_total_node[i + 1] + dw_el

        t_peak_seg[i] = t_p
        t_res_seg[i] = t_r
        t_mob_seg[i] = t_mob
        F_fric_seg[i] = F_fric
        W_seg[i] = dW
        state_seg[i] = state

    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_fric_seg)

    return (
        depths,
        w_total_node,
        w_elastic_node,
        w_fric_node,
        T_node,
        t_peak_seg,
        t_res_seg,
        t_mob_seg,
        F_fric_seg,
        cum_fric_seg,
        dw_elastic_seg,
        W_seg,
        state_seg,
    )


def plot_tension_pile_cross_section(layers_df, depths, D, T_node, t_mob_seg, state_seg, total_weight_kn):
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    max_depth = depths[-1]
def run_tension_model_partial_load(layers_df, E, D, gamma_pile, f_tension_opt, target_T0):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    n_s = len(layers_df)

    if target_T0 <= 1e-6:
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
            np.zeros(n_s),
            ["No Displacement"] * n_s,
        )

    low_w, high_w = 0.0, 0.25
    for _ in range(65):
        mid_w = 0.5 * (low_w + high_w)
        w_curr = mid_w
        T_curr = target_T0
        for idx in range(n_s):
            dz_s = max(depths[idx + 1] - depths[idx], 0.01)
            sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
            t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"])) * f_tension_opt
            t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"])) * f_tension_opt
            w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
            w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0

            dW = gamma_pile * A * dz_s
            w_bot_est = max(w_curr - (T_curr * dz_s) / (E * A), 0.0)
            w_mid = 0.5 * (w_curr + w_bot_est)
            tm, _ = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
            ff = U * dz_s * tm
            T_next = max(T_curr - ff - dW, 0.0)
            T_avg = 0.5 * (T_curr + T_next)
            w_curr = max(w_curr - (T_avg * dz_s) / (E * A), 0.0)
            T_curr = T_next

        if T_curr > 1e-4:
            low_w = mid_w
        else:
            high_w = mid_w

    w_total_node = np.zeros(n_s + 1)
    T_node = np.zeros(n_s + 1)
    t_peak_seg = np.zeros(n_s)
    t_res_seg = np.zeros(n_s)
    t_mob_seg = np.zeros(n_s)
    F_fric_seg = np.zeros(n_s)
    W_seg = np.zeros(n_s)
    dw_elastic_seg = np.zeros(n_s)
    state_seg = [""] * n_s

    w_total_node[0] = 0.5 * (low_w + high_w)
    T_node[0] = target_T0

    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"])) * f_tension_opt
        t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"])) * f_tension_opt
        w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
        w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0
        dW = gamma_pile * A * dz_s

        if T_node[idx] <= 1e-6:
            tm, st_txt = 0.0, "No Displacement"
            ff = 0.0
            T_node[idx + 1] = 0.0
            dw_el = 0.0
            w_total_node[idx + 1] = 0.0
        else:
            w_bot_est = max(w_total_node[idx] - (T_node[idx] * dz_s) / (E * A), 0.0)
            w_mid = 0.5 * (w_total_node[idx] + w_bot_est)
            tm, st_txt = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
            ff_max = U * dz_s * tm
            rem_after_w = max(T_node[idx] - dW, 0.0)
            ff = min(rem_after_w, ff_max)
            if ff < ff_max and rem_after_w > 0:
                tm = ff / (U * dz_s)
            elif rem_after_w == 0:
                tm = 0.0
                st_txt = "Self-Weight Only"
            T_node[idx + 1] = max(T_node[idx] - ff - dW, 0.0)
            T_avg = 0.5 * (T_node[idx] + T_node[idx + 1])
            dw_el = (T_avg * dz_s) / (E * A)
            w_total_node[idx + 1] = max(w_total_node[idx] - dw_el, 0.0)

        t_peak_seg[idx] = t_p
        t_res_seg[idx] = t_r
        t_mob_seg[idx] = tm
        F_fric_seg[idx] = ff
        W_seg[idx] = dW
        dw_elastic_seg[idx] = dw_el
        state_seg[idx] = st_txt

    w_elastic_node = np.append(np.cumsum(dw_elastic_seg[::-1])[::-1], 0.0)
    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_fric_seg)

    return (
        depths,
        w_total_node,
        w_elastic_node,
        w_fric_node,
        T_node,
        t_peak_seg,
        t_res_seg,
        t_mob_seg,
        F_fric_seg,
        cum_fric_seg,
        dw_elastic_seg,
        W_seg,
        state_seg,
    )


def find_max_uplift_at_zero_base_disp(layers_df, E, D, gamma_pile, f_tension_opt):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    n_s = len(layers_df)

    max_possible_T = 0.0
    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"])) * f_tension_opt
        max_possible_T += U * dz_s * t_p + gamma_pile * A * dz_s

    if max_possible_T <= 0:
        return 0.0

    low_T, high_T = 0.0, max_possible_T * 1.2
    for _ in range(60):
        mid_T = 0.5 * (low_T + high_T)
        res = run_tension_model_partial_load(layers_df, E, D, gamma_pile, f_tension_opt, mid_T)
        T_node_res = res[4]
        if float(T_node_res[-1]) <= 1e-3:
            low_T = mid_T
        else:
            high_T = mid_T

    return low_T


def build_preset_df(preset_name, f_peak, f_res):
    cols = (
        "Bottom Depth (m)",
        "Soil Type",
        "Initial Pressuremeter σ'h (kPa)",
        "Peak Friction δ_peak (°)",
        "Peak Disp w_peak (mm)",
        "Residual Friction δ_res (°)",
        "Residual Disp w_res (mm)",
    )
    if preset_name == "Multi-layer (Fill, Silt, Marl, Clay, Dense Sand)":
        rows = (
            (1.0, "Fill", 40.0, round(25.0 * f_peak, 1), 4.0, round(24.0 * f_res, 1), 9.0),
            (2.0, "Fill", 45.0, round(25.0 * f_peak, 1), 4.0, round(24.0 * f_res, 1), 9.0),
            (3.0, "Silt", 55.0, round(27.0 * f_peak, 1), 3.5, round(23.0 * f_res, 1), 8.5),
            (4.0, "Silt", 65.0, round(27.0 * f_peak, 1), 3.5, round(23.0 * f_res, 1), 8.5),
            (5.0, "Marl", 85.0, round(29.0 * f_peak, 1), 3.0, round(22.0 * f_res, 1), 9.0),
            (6.0, "Marl", 95.0, round(29.0 * f_peak, 1), 3.0, round(22.0 * f_res, 1), 9.0),
            (7.0, "Clay", 105.0, round(24.0 * f_peak, 1), 3.0, round(16.0 * f_res, 1), 10.0),
            (8.0, "Dense Sand", 130.0, round(35.8 * f_peak, 1), 2.0, round(30.1 * f_res, 1), 6.0),
            (9.0, "Dense Sand", 145.0, round(35.8 * f_peak, 1), 2.0, round(30.1 * f_res, 1), 6.0),
            (10.0, "Dense Sand", 160.0, round(35.8 * f_peak, 1), 2.0, round(30.1 * f_res, 1), 6.0),
        )
        return pd.DataFrame.from_records(rows, columns=cols)

    params = {
        "Dense Sand": (80.0, 10.0, 35.8, 2.0, 30.1, 6.0),
        "Loose Sand": (50.0, 5.0, 28.0, 3.5, 26.5, 8.0),
        "Marl": (75.0, 8.0, 29.0, 3.0, 22.0, 9.0),
        "Silt": (55.0, 6.0, 27.0, 3.5, 23.0, 8.5),
        "Clay": (60.0, 5.0, 24.0, 3.0, 16.0, 10.0),
    }
    s0, ds, dp, wp, dr, wr = params.get(preset_name)
    rows = tuple(
        (float(i + 1), preset_name, s0 + i * ds, round(dp * f_peak, 1), wp, round(dr * f_res, 1), wr)
        for i in range(10)
    )
    return pd.DataFrame.from_records(rows, columns=cols)


st.sidebar.header("1. Pile Properties & Density")
pile_mat = st.sidebar.selectbox("Pile Material", ("Concrete", "Steel", "HDPE Plastic", "Wood", "Composite / Custom"))
mat_props = {
    "Concrete": (30000000, 24.0, 1.0, 1.0),
    "Steel": (210000000, 78.5, 0.67, 0.76),
    "HDPE Plastic": (1500000, 9.5, 0.45, 0.55),
    "Wood": (11000000, 8.0, 0.85, 0.85),
    "Composite / Custom": (45000000, 30.0, 1.0, 1.0),
}
default_E, default_gamma, f_peak, f_res = mat_props.get(pile_mat)
E = st.sidebar.number_input("Elastic Modulus E (kPa)", value=default_E, step=1000000)
gamma_pile = st.sidebar.number_input("Unit Weight γ_pile (kN/m³)", min_value=0.0, max_value=120.0, value=float(default_gamma), step=0.5)
D = st.sidebar.number_input("Pile Diameter D (m)", min_value=0.2, max_value=3.0, value=0.6, step=0.05)
L_pile = st.sidebar.number_input("Pile Length L (m)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)

soil_preset = st.sidebar.selectbox(
    "Soil Profile Preset",
    ("Multi-layer (Fill, Silt, Marl, Clay, Dense Sand)", "Dense Sand", "Loose Sand", "Clay", "Marl", "Silt"),
)

default_layers = build_preset_df(soil_preset, f_peak, f_res)
default_layers["Bottom Depth (m)"] = (default_layers["Bottom Depth (m)"] * (L_pile / 10.0)).round(2)

st.sidebar.markdown("---")
st.sidebar.header("2. Uplift Load Range & Physics Options")
T_max_user = st.sidebar.number_input("Max Uplift Load T_max (kN):", min_value=100.0, max_value=50000.0, value=3000.0, step=100.0)

use_tension_factor = st.sidebar.checkbox("Optional: Apply Tension Friction Factor (Default = Pure Physics 1.0)", value=False)
f_tension_opt = st.sidebar.slider("Tension Friction Multiplier α_t:", 0.5, 1.0, 0.8, 0.05) if use_tension_factor else 1.0

st.subheader("1. Soil Layers Input Table (Editable)")
edited_df = st.data_editor(
    default_layers,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Soil Type": st.column_config.SelectboxColumn("Soil Type", options=list(SOIL_COLORS.keys()), required=True)
    },
)

with st.expander("Optional: Field Uplift Load Test Data for Comparison on Graph"):
    field_rows = ((0.0, 0.0), (300.0, 1.0), (600.0, 2.8), (900.0, 6.2), (1100.0, 14.5))
    field_test_df = st.data_editor(
        pd.DataFrame.from_records(field_rows, columns=("Measured Uplift Load T0 (kN)", "Measured Head Uplift w0 (mm)")),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

layers_df = (
    edited_df.dropna(subset=["Bottom Depth (m)", "Initial Pressuremeter σ'h (kPa)"])
    .sort_values(by="Bottom Depth (m)")
    .reset_index(drop=True)
)
if len(layers_df) > 0:
    T_threshold_zero_toe = find_max_uplift_at_zero_base_disp(layers_df, E, D, gamma_pile, f_tension_opt)
    T0_list, w0_tot_list, w0_el_list, w0_fr_list, cum_fric_list = [], [], [], [], []
    partial_loads = np.linspace(0.0, min(T_threshold_zero_toe, T_max_user), 40) if T_threshold_zero_toe > 0 else []

    for t_val in partial_loads:
        _, wt, we, wf, Tn, _, _, _, _, cf, _, _, _ = run_tension_model_partial_load(
            layers_df, E, D, gamma_pile, f_tension_opt, t_val
        )
        T0_list.append(float(Tn.item(0)))
        w0_tot_list.append(float(wt.item(0) * 1000.0))
        w0_el_list.append(float(we.item(0) * 1000.0))
        w0_fr_list.append(float(wf.item(0) * 1000.0))
        cum_fric_list.append(float(cf.item(-1)))

    wb_grid = np.linspace(0.0001, 0.08, 220)
    peak_uplift_capacity = max(T0_list) if len(T0_list) > 0 else 0.0
    for wb in wb_grid:
        _, wt, we, wf, Tn, _, _, _, _, cf, _, _, _ = run_tension_model_from_base(
            layers_df, E, D, gamma_pile, f_tension_opt, wb
        )
        t_curr = float(Tn.item(0))
        w_curr_mm = float(wt.item(0) * 1000.0)
        peak_uplift_capacity = max(peak_uplift_capacity, t_curr)
        T0_list.append(t_curr)
        w0_tot_list.append(w_curr_mm)
        w0_el_list.append(float(we.item(0) * 1000.0))
        w0_fr_list.append(float(wf.item(0) * 1000.0))
        cum_fric_list.append(float(cf.item(-1)))
        if t_curr >= T_max_user and wb > 0.01:
            break

    st.markdown("---")
    st.subheader("2. Interactive Uplift Evaluation & Simulation Results")

    target_T0 = st.slider(
        "Select Evaluation Uplift Head Load T0 (kN):",
        min_value=0.0,
        max_value=float(T_max_user),
        value=min(round(T_threshold_zero_toe * 0.5, 1), float(T_max_user)),
        step=10.0,
    )

    if target_T0 <= T_threshold_zero_toe:
        (
            depths,
            w_total_node,
            w_elastic_node,
            w_fric_node,
            T_node,
            t_peak_seg,
            t_res_seg,
            t_mob_seg,
            F_fric_seg,
            cum_fric_seg,
            dw_elastic_seg,
            W_seg,
            state_seg,
        ) = run_tension_model_partial_load(layers_df, E, D, gamma_pile, f_tension_opt, target_T0)
    else:
        low_wb, high_wb = 0.0, 0.05
        for _ in range(55):
            mid_wb = 0.5 * (low_wb + high_wb)
            res_tmp = run_tension_model_from_base(layers_df, E, D, gamma_pile, f_tension_opt, mid_wb)
            if float(res_tmp[4].item(0)) < target_T0:
                low_wb = mid_wb
            else:
                high_wb = mid_wb

        (
            depths,
            w_total_node,
            w_elastic_node,
            w_fric_node,
            T_node,
            t_peak_seg,
            t_res_seg,
            t_mob_seg,
            F_fric_seg,
            cum_fric_seg,
            dw_elastic_seg,
            W_seg,
            state_seg,
        ) = run_tension_model_from_base(layers_df, E, D, gamma_pile, f_tension_opt, 0.5 * (low_wb + high_wb))

    t0_eval = float(T_node.item(0))
    w0_tot_eval = float(w_total_node.item(0) * 1000.0)
    w0_el_eval = float(w_elastic_node.item(0) * 1000.0)
    w0_fr_eval = float(w_fric_node.item(0) * 1000.0)
    total_weight_kn = float(np.sum(W_seg))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Uplift Head Load T0", f"{t0_eval:,.1f} kN", f"{t0_eval/10.0:,.1f} ton")
    m2.metric("Total Head Uplift w0", f"{w0_tot_eval:.2f} mm")
    m3.metric("Pile Self-Weight W", f"{total_weight_kn:,.1f} kN", f"γ = {gamma_pile:.1f} kN/m³")
    m4.metric("Peak Uplift Capacity T_ult", f"{peak_uplift_capacity:,.1f} kN", f"w_base=0 limit: {T_threshold_zero_toe:,.0f} kN")

    if target_T0 > t0_eval + 5.0:
        st.error(f"כשל שליפה (הרס מלא במתיחה): כוח השליפה המבוקש ({target_T0:,.0f} kN) עובר את תסבולת השליפה המרבית של הכלונס ({peak_uplift_capacity:,.1f} kN).")

    st.info(
        f"Head Uplift Breakdown: Elastic Elongation = {w0_el_eval:.2f} mm | Interface/Toe Uplift = {w0_fr_eval:.2f} mm | Total Uplift = {w0_tot_eval:.2f} mm"
    )

    c_vis, c_curve = st.columns([1.0, 1.45])
    fig_soil = plot_tension_pile_cross_section(layers_df, depths, D, T_node, t_mob_seg, state_seg, total_weight_kn)
    c_vis.pyplot(fig_soil)

    fig_ls, ax_a = plt.subplots(figsize=(6.8, 5.2))
    ax_a.plot(w0_tot_list, T0_list, label="Model: Total Uplift (w_total)", color="#1f497d", linewidth=2.4)
    ax_a.plot(w0_el_list, T0_list, "--", label="Model: Elastic Elongation (w_elastic)", color="#2e75b6")
    ax_a.plot(w0_fr_list, T0_list, "-.", label="Model: Interface/Toe Slip (w_fric)", color="#e67e22")

    clean_field = field_test_df.dropna()
    if len(clean_field) > 0:
        ax_a.plot(
            clean_field["Measured Head Uplift w0 (mm)"],
            clean_field["Measured Uplift Load T0 (kN)"],
            "o-",
            color="#8e44ad",
            linewidth=1.8,
            markersize=6,
            label="Field Uplift Test (Measured)",
        )

    ax_a.axhline(total_weight_kn, color="brown", linestyle=":", label=f"Pile Self-Weight W ({total_weight_kn:.1f} kN)")
    ax_a.axhline(peak_uplift_capacity, color="green", linestyle=":", label=f"Peak Uplift Capacity ({peak_uplift_capacity:.0f} kN)")
    ax_a.scatter([w0_tot_eval], [t0_eval], color="red", s=80, zorder=6, label="Current Evaluation Point")
    ax_a.set_ylim(0, max(T_max_user, peak_uplift_capacity) * 1.08)
    ax_a.set_xlim(0, 50)
    ax_a.set_xlabel("Pile Head Upward Displacement w0 (mm)", fontname="Times New Roman")
    ax_a.set_ylabel("Pile Head Uplift Load T0 (kN)", fontname="Times New Roman")
    ax_a.set_title("Tension / Uplift Load-Displacement Curve", fontname="Times New Roman", fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=8)
    plt.tight_layout()
    c_curve.pyplot(fig_ls)

    st.markdown("---")
    st.subheader("2B. 10-Increment Simulated Uplift Load Test (Instrumented Pile)")

    col_sim1, _ = st.columns([1.2, 1.8])
    with col_sim1:
        sim_max_load = st.number_input(
            "Enter Maximum Uplift Load for 10-Step Simulation (kN):",
            min_value=50.0,
            max_value=50000.0,
            value=float(T_max_user),
            step=100.0,
            key="sim_tension_max_input",
        )
    run_10_steps_btn = st.button("Run 10-Step Uplift Test Simulation", type="primary")
    fig_10_for_excel = None
    fig_10w_for_excel = None
    fig_10s_for_excel = None

    if run_10_steps_btn or "sim_t10_df" in st.session_state:
        A_pile = np.pi * (D ** 2) / 4.0
        step_loads = np.linspace(0.0, sim_max_load, 11)
        sim_records = []
        step_profiles = []

        for step_idx, t_req in enumerate(step_loads):
            if t_req <= T_threshold_zero_toe:
                res_s = run_tension_model_partial_load(layers_df, E, D, gamma_pile, f_tension_opt, t_req)
                failed_step = False
            else:
                low_wb, high_wb = 0.0, 0.05
                for _ in range(55):
                    mid_wb = 0.5 * (low_wb + high_wb)
                    res_tmp = run_tension_model_from_base(layers_df, E, D, gamma_pile, f_tension_opt, mid_wb)
                    if float(res_tmp[4].item(0)) < t_req:
                        low_wb = mid_wb
                    else:
                        high_wb = mid_wb
                res_s = run_tension_model_from_base(layers_df, E, D, gamma_pile, f_tension_opt, 0.5 * (low_wb + high_wb))
                failed_step = t_req > float(res_s[4].item(0)) + 5.0

            d_s, wt_s, we_s, wf_s, Tn_s = res_s[0], res_s[1], res_s[2], res_s[3], res_s[4]
            t_act = float(Tn_s.item(0))
            wt_mm = float(wt_s.item(0) * 1000.0)
            we_mm = float(we_s.item(0) * 1000.0)
            wf_mm = float(wf_s.item(0) * 1000.0)
            st_msg = "OK (Partial)" if t_req <= T_threshold_zero_toe else ("Pullout Failure!" if failed_step else "OK (Full Mobilization)")

            sim_records.append({
                "Step": step_idx,
                "Target Uplift (kN)": round(float(t_req), 1),
                "Achieved Uplift T0 (kN)": round(t_act, 1),
                "Head Uplift w0 (mm)": round(wt_mm, 2),
                "Elastic Elong. (mm)": round(we_mm, 2),
                "Interface/Toe Slip (mm)": round(wf_mm, 2),
                "Status": st_msg,
            })

            if not failed_step:
                step_profiles.append({
                    "step": step_idx,
                    "T0": t_act,
                    "depths": d_s,
                    "w_mm": wt_s * 1000.0,
                    "sigma_kPa": Tn_s / A_pile,
                    "T_kN": Tn_s,
                })

        st.session_state["sim_t10_df"] = pd.DataFrame(sim_records)
        st.session_state["sim_t10_profiles"] = step_profiles

        sim_t10_df = st.session_state["sim_t10_df"]
        c_tbl10, c_plt10 = st.columns([1.2, 1.2])
        with c_tbl10:
            st.dataframe(sim_t10_df, use_container_width=True, hide_index=True)

        with c_plt10:
            fig_10, ax_10 = plt.subplots(figsize=(6.2, 4.5))
            ax_10.plot(T0_list, w0_tot_list, color="#aeb6bf", linestyle="--", linewidth=1.5, label="Continuous Model Curve")
            ax_10.plot(
                sim_t10_df["Achieved Uplift T0 (kN)"],
                sim_t10_df["Head Uplift w0 (mm)"],
                "s-",
                color="#d35400",
                linewidth=2.2,
                markersize=7,
                label="10-Step Simulated Uplift",
            )
            ax_10.set_xlabel("Pile Head Uplift Load T0 (kN)", fontname="Times New Roman")
            ax_10.set_ylabel("Pile Head Upward Disp w0 (mm)", fontname="Times New Roman")
            ax_10.set_title("10-Step Simulated Uplift Test", fontname="Times New Roman", fontweight="bold")
            ax_10.grid(True, linestyle="--", alpha=0.5)
            ax_10.legend(fontsize=8.5)
        if len(step_profiles) > 0:
            st.markdown("#### Depth Profiles Across Pre-Failure Uplift Steps (Including Pile Toe)")
            col_10w, col_10s = st.columns(2)
            cmap_steps = plt.get_cmap("viridis", len(step_profiles))

            with col_10w:
                fig_10w, ax_10w = plt.subplots(figsize=(6.2, 5.2))
                for idx_p, prof in enumerate(step_profiles):
                    ax_10w.plot(
                        prof["w_mm"],
                        prof["depths"],
                        marker="o",
                        markersize=4.5,
                        linewidth=1.8,
                        color=cmap_steps(idx_p),
                        label=f"Step {prof['step']}: T0={prof['T0']:.0f} kN",
                    )
                ax_10w.invert_yaxis()
                ax_10w.set_xlabel("Upward Axial Displacement w(z) [mm]", fontweight="bold", fontname="Times New Roman")
                ax_10w.set_ylabel("Depth z [m]", fontweight="bold", fontname="Times New Roman")
                ax_10w.set_title("Upward Displacement Along Pile & Toe", fontweight="bold", fontname="Times New Roman")
                ax_10w.grid(True, linestyle="--", alpha=0.5)
                ax_10w.legend(fontsize=7.5, loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_10w)
                fig_10w_for_excel = fig_10w

            with col_10s:
                fig_10s, ax_10s = plt.subplots(figsize=(6.2, 5.2))
                for idx_p, prof in enumerate(step_profiles):
                    ax_10s.plot(
                        prof["sigma_kPa"],
                        prof["depths"],
                        marker="^",
                        markersize=4.5,
                        linewidth=1.8,
                        color=cmap_steps(idx_p),
                        label=f"Step {prof['step']}: T0={prof['T0']:.0f} kN",
                    )
                ax_10s.invert_yaxis()
                ax_10s.set_xlabel("Tensile Axial Stress σ_t(z) = T(z)/A [kPa]", fontweight="bold", fontname="Times New Roman")
                ax_10s.set_ylabel("Depth z [m]", fontweight="bold", fontname="Times New Roman")
                ax_10s.set_title("Tensile Stress Along Pile & Toe (T_base = 0)", fontweight="bold", fontname="Times New Roman")
                ax_10s.grid(True, linestyle="--", alpha=0.5)
                ax_10s.legend(fontsize=7.5, loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_10s)
                fig_10s_for_excel = fig_10s

    st.subheader("3. Combined Depth Profile: Tensile Load & Upward Displacement Along Pile")
    fig_comb, ax_load = plt.subplots(figsize=(10.5, 5.2))

    for i in range(len(layers_df)):
        z_top, z_bot = depths[i], depths[i + 1]
        s_type = str(layers_df.at[i, "Soil Type"]).strip()
        bg_col = SOIL_COLORS.get(s_type, "#f2f4f4")
        ax_load.axhspan(z_top, z_bot, facecolor=bg_col, alpha=0.35)
        ax_load.text(
            max(float(T_node.item(0)) * 0.02, 5.0),
            0.5 * (z_top + z_bot),
            f"{s_type} (τ={t_mob_seg.item(i):.1f} kPa | dW={W_seg.item(i):.1f} kN)",
            va="center",
            ha="left",
            fontsize=8,
            color="#2c3e50",
            alpha=0.85,
        )

    line1 = ax_load.plot(T_node, depths, marker="^", markersize=7, linewidth=2.6, color="#8e44ad", label="Tensile Axial Load T(z) [kN]")
    ax_load.set_xlabel("Tensile Axial Load Along Pile T(z) [kN]", color="#8e44ad", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_load.tick_params(axis="x", labelcolor="#8e44ad")
    ax_load.set_ylabel("Depth z [m]", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_load.set_ylim(depths[-1], 0.0)
    ax_load.set_xlim(0.0, max(float(T_node.item(0)) * 1.15, 50.0))
    ax_load.grid(True, linestyle="--", alpha=0.45)

    ax_disp = ax_load.twiny()
    line2 = ax_disp.plot(
        w_total_node * 1000.0, depths, marker="o", markersize=6, linestyle="-.", linewidth=2.4, color="#1f497d", label="Upward Displacement w(z) [mm]"
    )
    ax_disp.set_xlabel("Upward Displacement Along Pile w(z) [mm]", color="#1f497d", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_disp.tick_params(axis="x", labelcolor="#1f497d")
    ax_disp.set_xlim(0.0, max(float(w_total_node.item(0)) * 1000.0 * 1.15, 1.0))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_load.legend(lines, labels, loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax_load.set_title(
        f"Uplift Load Transfer & Displacement Distribution (T0 = {t0_eval:,.1f} kN | w0 = {w0_tot_eval:.2f} mm)",
        fontsize=11.5,
        fontweight="bold",
        fontname="Times New Roman",
        pad=14,
    )
    plt.tight_layout()
    st.pyplot(fig_comb)

    st.markdown("---")
    st.subheader("4. Direct Shear Curves (τ - Δx) & Current Layer Operating Points")
    fig_ds, ax_ds = plt.subplots(figsize=(10.5, 5.2))
    cmap = plt.get_cmap("tab10")

    max_w_plot = max(
        float(layers_df["Residual Disp w_res (mm)"].max()) * 1.4,
        float(np.max(w_total_node) * 1000.0 * 1.1),
        12.0,
    )
    w_grid_mm = np.linspace(0.0, max_w_plot, 250)

    for i in range(len(layers_df)):
        s_type = str(layers_df.at[i, "Soil Type"]).strip()
        z_top, z_bot = depths[i], depths[i + 1]
        t_p = float(t_peak_seg.item(i))
        t_r = float(t_res_seg.item(i))
        w_p = float(layers_df.at[i, "Peak Disp w_peak (mm)"]) / 1000.0
        w_r = float(layers_df.at[i, "Residual Disp w_res (mm)"]) / 1000.0

        tau_curve = [calc_shear_stress(wm / 1000.0, t_p, t_r, w_p, w_r)[0] for wm in w_grid_mm]
        line_col = cmap(i % 10)
        lbl = f"L{i+1}: {s_type} ({z_top:.1f}-{z_bot:.1f}m) | τ_mob={t_mob_seg.item(i):.1f} kPa"
        ax_ds.plot(w_grid_mm, tau_curve, color=line_col, linewidth=2.0, label=lbl)

        w_op_mm = 0.5 * (float(w_total_node.item(i)) + float(w_total_node.item(i + 1))) * 1000.0
        t_op = float(t_mob_seg.item(i))
        ax_ds.scatter([w_op_mm], [t_op], color="red", edgecolor="black", s=65, zorder=6)

    ax_ds.scatter([], [], color="red", edgecolor="black", s=65, label="Current Operating Point (Layer Mid-Depth)")
    ax_ds.set_xlabel("Interface Shear Displacement Δx [mm]", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_ds.set_ylabel("Mobilized Shear Stress τ [kPa]", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_ds.set_title("Direct Shear Stress-Displacement Curves by Soil Layer", fontsize=11.5, fontweight="bold", fontname="Times New Roman")
    ax_ds.set_xlim(0.0, max_w_plot)
    ax_ds.set_ylim(bottom=0.0)
    ax_ds.grid(True, linestyle="--", alpha=0.45)
    ax_ds.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8.5, frameon=True)
    plt.tight_layout()
    st.pyplot(fig_ds)

    st.markdown("---")
    st.subheader("5. Export Full Tension Engineering Report to Excel (Print-Ready)")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        summary_rows = (
            ("Analysis Mode", "Tension / Uplift (Pure Physics)"),
            ("Pile Material", pile_mat),
            ("Elastic Modulus E (kPa)", f"{E:,.0f}"),
            ("Unit Weight γ_pile (kN/m³)", f"{gamma_pile:.1f}"),
            ("Total Pile Self-Weight W (kN)", f"{total_weight_kn:,.1f}"),
            ("Pile Diameter D (m)", f"{D:.2f}"),
            ("Pile Length L (m)", f"{depths[-1]:.2f}"),
            ("Soil Profile Preset", soil_preset),
            ("Evaluation Uplift Load T0 (kN)", f"{t0_eval:,.1f}"),
            ("Total Head Uplift w0 (mm)", f"{w0_tot_eval:.2f}"),
            ("Elastic Elongation (mm)", f"{w0_el_eval:.2f}"),
            ("Interface/Toe Uplift (mm)", f"{w0_fr_eval:.2f}"),
            ("Peak Uplift Capacity T_ult (kN)", f"{peak_uplift_capacity:,.1f}"),
        )
        summary_df = pd.DataFrame.from_records(summary_rows, columns=("Parameter", "Value"))
        summary_df.to_excel(writer, sheet_name="Summary & Results", startrow=2, index=False)

        export_layers_df = pd.DataFrame({
            "Layer": range(1, len(layers_df) + 1),
            "Soil Type": layers_df["Soil Type"],
            "Depth (m)": [f"{depths[i]:.1f} - {depths[i+1]:.1f}" for i in range(len(layers_df))],
            "Upward Disp (mm)": w_total_node[:-1] * 1000.0,
            "Elastic Elong. (mm)": dw_elastic_seg * 1000.0,
            "Segment Weight dW (kN)": W_seg,
            "Peak Shear (kPa)": t_peak_seg,
            "Residual Shear (kPa)": t_res_seg,
            "Mobilized Shear (kPa)": t_mob_seg,
            "Layer Friction (kN)": F_fric_seg,
            "Tensile Load at Top (kN)": T_node[:-1],
            "State": state_seg,
        })
        export_layers_df.to_excel(writer, sheet_name="Summary & Results", startrow=18, index=False)

        ws_res = writer.sheets["Summary & Results"]
        prof_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#2e86c1", "font_name": "Times New Roman"})
        title_fmt = workbook.add_format({"bold": True, "font_size": 13, "font_color": "#1f497d", "font_name": "Times New Roman"})

        ws_res.write(0, 0, "Prof. Uri Komornik", prof_fmt)
        ws_res.write(1, 0, "Tension / Uplift Pile Geomechanical Report", title_fmt)
        ws_res.set_column("A:B", 28)
        ws_res.set_column("C:L", 16)
        ws_res.set_landscape()
        ws_res.fit_to_pages(1, 0)

        layers_df.to_excel(writer, sheet_name="Soil Input Data", startrow=2, index=False)
        ws_in = writer.sheets["Soil Input Data"]
        ws_in.write(0, 0, "Prof. Uri Komornik", prof_fmt)
        ws_in.write(1, 0, "Input Soil Profile & Pressuremeter Parameters", title_fmt)
        ws_in.set_column("A:G", 22)

