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
