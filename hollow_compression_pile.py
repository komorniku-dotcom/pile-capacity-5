import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pile Geomechanical Model - Hollow Compression Pile", layout="wide")

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

st.title("Pile Geomechanical Model (Hollow Pile - Compression) - Soil Plug & Load Test")
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


def get_hollow_geometry(D_out, t_wall):
    D_in = max(D_out - 2.0 * t_wall, 0.0)
    A_gross = np.pi * (D_out ** 2) / 4.0
    A_inner = np.pi * (D_in ** 2) / 4.0
    A_wall = max(A_gross - A_inner, 1e-6)
    U_out = np.pi * D_out
    U_in = np.pi * D_in
    return D_in, A_gross, A_inner, A_wall, U_out, U_in


def run_hollow_compression_partial(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, target_P0):
    """
    Solves partial compression load (P_base = 0, w_base = 0) from head downwards
    using annular wall cross-section A_wall and external + internal plug friction.
    """
    D_in, A_gross, A_inner, A_wall, U_out, U_in = get_hollow_geometry(D_out, t_wall)
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    L_total = depths[-1]
    z_plug_top = max(L_total - H_plug, 0.0) if end_mode == "Open-Ended (Soil Plug)" else L_total
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
            np.zeros(n_s),
            ["No Displacement"] * n_s,
            0.0,
            "No Displacement",
        )

    low_w, high_w = 0.0, 0.25
    for _ in range(65):
        mid_w = 0.5 * (low_w + high_w)
        w_curr = mid_w
        P_curr = target_P0
        for idx in range(n_s):
            z_top, z_bot = depths[idx], depths[idx + 1]
            dz_s = max(z_bot - z_top, 0.01)
            plug_len_seg = max(0.0, z_bot - max(z_top, z_plug_top))

            sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
            t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"]))
            t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"]))
            w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
            w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0

            w_bot_est = max(w_curr - (P_curr * dz_s) / (E * A_wall), 0.0)
            w_mid = 0.5 * (w_curr + w_bot_est)
            tm, _ = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)

            ff_out = U_out * dz_s * tm
            ff_in = U_in * plug_len_seg * tm
            ff_tot = ff_out + ff_in

            P_next = max(P_curr - ff_tot, 0.0)
            P_avg = 0.5 * (P_curr + P_next)
            w_curr = max(w_curr - (P_avg * dz_s) / (E * A_wall), 0.0)
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
    F_out_seg = np.zeros(n_s)
    F_in_seg = np.zeros(n_s)
    dw_elastic_seg = np.zeros(n_s)
    state_seg = [""] * n_s

    w_total_node[0] = 0.5 * (low_w + high_w)
    P_node[0] = target_P0

    for idx in range(n_s):
        z_top, z_bot = depths[idx], depths[idx + 1]
        dz_s = max(z_bot - z_top, 0.01)
        plug_len_seg = max(0.0, z_bot - max(z_top, z_plug_top))

        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"]))
        t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"]))
        w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
        w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0

        if P_node[idx] <= 1e-6:
            tm, st_txt = 0.0, "No Displacement"
            f_out, f_in = 0.0, 0.0
            P_node[idx + 1] = 0.0
            dw_el = 0.0
            w_total_node[idx + 1] = 0.0
        else:
            w_bot_est = max(w_total_node[idx] - (P_node[idx] * dz_s) / (E * A_wall), 0.0)
            w_mid = 0.5 * (w_total_node[idx] + w_bot_est)
            tm, st_txt = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)

            f_out_max = U_out * dz_s * tm
            f_in_max = U_in * plug_len_seg * tm
            f_tot_max = f_out_max + f_in_max

            if P_node[idx] < f_tot_max and f_tot_max > 0:
                ratio = P_node[idx] / f_tot_max
                f_out = f_out_max * ratio
                f_in = f_in_max * ratio
                tm = tm * ratio
            else:
                f_out = f_out_max
                f_in = f_in_max

            P_node[idx + 1] = max(P_node[idx] - (f_out + f_in), 0.0)
            P_avg = 0.5 * (P_node[idx] + P_node[idx + 1])
            dw_el = (P_avg * dz_s) / (E * A_wall)
            w_total_node[idx + 1] = max(w_total_node[idx] - dw_el, 0.0)

        t_peak_seg[idx] = t_p
        t_res_seg[idx] = t_r
        t_mob_seg[idx] = tm
        F_out_seg[idx] = f_out
        F_in_seg[idx] = f_in
        dw_elastic_seg[idx] = dw_el
        state_seg[idx] = st_txt

    w_elastic_node = np.append(np.cumsum(dw_elastic_seg[::-1])[::-1], 0.0)
    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_out_seg + F_in_seg)

    return (
        depths,
        w_total_node,
        w_elastic_node,
        w_fric_node,
        P_node,
        t_peak_seg,
        t_res_seg,
        t_mob_seg,
        F_out_seg,
        F_in_seg,
        cum_fric_seg,
        dw_elastic_seg,
        state_seg,
        0.0,
        "No Base Movement",
    )


def find_max_load_at_zero_base_disp(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base):
    D_in, A_gross, A_inner, A_wall, U_out, U_in = get_hollow_geometry(D_out, t_wall)
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    L_total = depths[-1]
    z_plug_top = max(L_total - H_plug, 0.0) if end_mode == "Open-Ended (Soil Plug)" else L_total

    max_possible_P = 0.0
    for idx in range(len(layers_df)):
        z_top, z_bot = depths[idx], depths[idx + 1]
        dz_s = max(z_bot - z_top, 0.01)
        plug_len_seg = max(0.0, z_bot - max(z_top, z_plug_top))
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"]))
        max_possible_P += (U_out * dz_s + U_in * plug_len_seg) * t_p

    if max_possible_P <= 0:
        return 0.0

    low_P, high_P = 0.0, max_possible_P * 1.2
    for _ in range(60):
        mid_P = 0.5 * (low_P + high_P)
        res = run_hollow_compression_partial(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, mid_P)
        if float(res[4][-1]) <= 1e-3:
            low_P = mid_P
        else:
            high_P = mid_P

    return low_P


def run_hollow_compression_from_base(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, w_base_m):
    """
    Solves full hollow pile compression from base upwards for a prescribed w_base_m > 0.
    Physical plugging check: compares Unplugged (ring bearing + internal plug shear) vs Plugged (gross bearing).
    """
    D_in, A_gross, A_inner, A_wall, U_out, U_in = get_hollow_geometry(D_out, t_wall)
    num_layers = len(layers_df)
    depths = [0.0] + layers_df["Bottom Depth (m)"].tolist()
    L_total = depths[-1]
    z_plug_top = max(L_total - H_plug, 0.0) if end_mode == "Open-Ended (Soil Plug)" else L_total

    w_total_node = np.zeros(num_layers + 1)
    w_elastic_node = np.zeros(num_layers + 1)
    P_node = np.zeros(num_layers + 1)

    t_peak_seg = np.zeros(num_layers)
    t_res_seg = np.zeros(num_layers)
    t_mob_seg = np.zeros(num_layers)
    F_out_seg = np.zeros(num_layers)
    F_in_seg = np.zeros(num_layers)
    dw_elastic_seg = np.zeros(num_layers)
    state_seg = [""] * num_layers

    w_total_node[num_layers] = w_base_m
    w_elastic_node[num_layers] = 0.0

    q_mob = min(k_base * w_base_m, q_ult)

    if end_mode == "Closed-Ended (Sealed Toe)":
        P_base = A_gross * q_mob
        plug_status = "Closed-Ended (Full Gross Toe Bearing)"
        include_inner_friction = False
    elif end_mode == "Pure Annular Ring (Unplugged)":
        P_base = A_wall * q_mob
        plug_status = "Unplugged (Annular Wall Bearing Only)"
        include_inner_friction = False
    else:
        # Open-Ended with Soil Plug: compare Unplugged (ring + plug friction) vs Plugged (full gross bearing)
        P_ring = A_wall * q_mob
        P_gross = A_gross * q_mob

