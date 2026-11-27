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
                # Calculate maximum available plug resistance along H_plug
        F_plug_avail = 0.0
        for i in range(num_layers - 1, -1, -1):
            z_top, z_bot = depths[i], depths[i + 1]
            plug_len_seg = max(0.0, z_bot - max(z_top, z_plug_top))
            if plug_len_seg > 0:
                sig_h = layers_df.at[i, "Initial Pressuremeter σ'h (kPa)"]
                t_p = sig_h * np.tan(np.radians(layers_df.at[i, "Peak Friction δ_peak (°)"]))
                t_r = sig_h * np.tan(np.radians(layers_df.at[i, "Residual Friction δ_res (°)"]))
                w_p = layers_df.at[i, "Peak Disp w_peak (mm)"] / 1000.0
                w_r = layers_df.at[i, "Residual Disp w_res (mm)"] / 1000.0
                tm_est, _ = calc_shear_stress(w_base_m, t_p, t_r, w_p, w_r)
                F_plug_avail += U_in * plug_len_seg * tm_est

        if P_ring + F_plug_avail >= P_gross:
            P_base = P_gross
            plug_status = "Plugged Mode (Plug Holds -> Full Gross Bearing)"
            include_inner_friction = False
        else:
            P_base = P_ring
            plug_status = "Unplugged Mode (Plug Slips -> Ring + Inner Shear)"
            include_inner_friction = True

    P_node[num_layers] = P_base

    for i in range(num_layers - 1, -1, -1):
        z_top, z_bot = depths[i], depths[i + 1]
        dz = max(z_bot - z_top, 0.01)
        plug_len_seg = max(0.0, z_bot - max(z_top, z_plug_top)) if include_inner_friction else 0.0

        sig_h = layers_df.at[i, "Initial Pressuremeter σ'h (kPa)"]
        d_peak = layers_df.at[i, "Peak Friction δ_peak (°)"]
        w_p = layers_df.at[i, "Peak Disp w_peak (mm)"] / 1000.0
        d_res = layers_df.at[i, "Residual Friction δ_res (°)"]
        w_r = layers_df.at[i, "Residual Disp w_res (mm)"] / 1000.0

        t_p = sig_h * np.tan(np.radians(d_peak))
        t_r = sig_h * np.tan(np.radians(d_res))

        w_top_est = w_total_node[i + 1] + (P_node[i + 1] * dz) / (E * A_wall)
        w_mid = 0.5 * (w_total_node[i + 1] + w_top_est)

        t_mob, state = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
        F_out = U_out * dz * t_mob
        F_in = U_in * plug_len_seg * t_mob

        P_node[i] = P_node[i + 1] + F_out + F_in
        P_mean = 0.5 * (P_node[i] + P_node[i + 1])

        dw_el = (P_mean * dz) / (E * A_wall)
        dw_elastic_seg[i] = dw_el
        w_elastic_node[i] = w_elastic_node[i + 1] + dw_el
        w_total_node[i] = w_total_node[i + 1] + dw_el

        t_peak_seg[i] = t_p
        t_res_seg[i] = t_r
        t_mob_seg[i] = t_mob
        F_out_seg[i] = F_out
        F_in_seg[i] = F_in
        state_seg[i] = state

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
        P_base,
        plug_status,
    )

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


st.sidebar.header("1. Hollow Pile Geometry & Material")
pile_mat = st.sidebar.selectbox("Pile Material", ("Steel Pipe", "Spun Concrete", "HDPE Pipe", "Composite / Custom"))
mat_props = {
    "Steel Pipe": (210000000, 0.67, 0.76),
    "Spun Concrete": (35000000, 1.0, 1.0),
    "HDPE Pipe": (1500000, 0.45, 0.55),
    "Composite / Custom": (50000000, 1.0, 1.0),
}
default_E, f_peak, f_res = mat_props.get(pile_mat)
E = st.sidebar.number_input("Elastic Modulus E (kPa)", value=default_E, step=1000000)

D_out = st.sidebar.number_input("Outer Diameter D_out (m)", min_value=0.2, max_value=3.0, value=0.6, step=0.05)
t_wall_mm = st.sidebar.number_input("Wall Thickness t_wall (mm)", min_value=2.0, max_value=500.0, value=16.0, step=1.0)
t_wall = t_wall_mm / 1000.0
L_pile = st.sidebar.number_input("Pile Length L (m)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)

D_in, A_gross, A_inner, A_wall, U_out, U_in = get_hollow_geometry(D_out, t_wall)
st.sidebar.caption(f"D_in = {D_in:.3f} m | A_wall = {A_wall:.4f} m² ({A_wall/A_gross*100:.1f}% of gross area)")

st.sidebar.markdown("---")
st.sidebar.header("2. Toe & Soil Plug Condition")
end_mode = st.sidebar.selectbox(
    "Pile Toe & Plug Mode",
    ("Open-Ended (Soil Plug)", "Closed-Ended (Sealed Toe)", "Pure Annular Ring (Unplugged)"),
)
H_plug = 0.0
if end_mode == "Open-Ended (Soil Plug)":
    H_plug = st.sidebar.number_input(
        "Internal Soil Plug Height H_plug (m from toe):",
        min_value=0.0,
        max_value=float(L_pile),
        value=min(6.0, float(L_pile)),
        step=0.5,
    )

soil_preset = st.sidebar.selectbox(
    "Soil Profile Preset",
    ("Multi-layer (Fill, Silt, Marl, Clay, Dense Sand)", "Dense Sand", "Loose Sand", "Clay", "Marl", "Silt"),
)
default_layers = build_preset_df(soil_preset, f_peak, f_res)
default_layers["Bottom Depth (m)"] = (default_layers["Bottom Depth (m)"] * (L_pile / 10.0)).round(2)

st.sidebar.markdown("---")
st.sidebar.header("3. Base Bearing & Load Range")
q_ult = st.sidebar.number_input("Ultimate End Bearing q_ult (kPa)", min_value=100.0, max_value=50000.0, value=6000.0, step=200.0)
k_base = st.sidebar.number_input("Base Stiffness k_base (kPa/m)", min_value=1000.0, max_value=2000000.0, value=250000.0, step=10000.0)
P_max_user = st.sidebar.number_input("Max Head Load P_max (kN):", min_value=100.0, max_value=50000.0, value=4000.0, step=100.0)

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

with st.expander("Optional: Field Compression Load Test Data for Comparison on Graph"):
    field_rows = ((0.0, 0.0), (500.0, 1.5), (1000.0, 4.2), (1500.0, 9.5), (2000.0, 19.0))
    field_test_df = st.data_editor(
        pd.DataFrame.from_records(field_rows, columns=("Measured Head Load P0 (kN)", "Measured Head Settlement w0 (mm)")),
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
    P_threshold_zero_toe = find_max_load_at_zero_base_disp(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base)
    P0_list, w0_tot_list, w0_el_list, w0_fr_list, Pbase_list, cum_fric_list = [], [], [], [], [], []
    partial_loads = np.linspace(0.0, min(P_threshold_zero_toe, P_max_user), 40) if P_threshold_zero_toe > 0 else []
    for p_val in partial_loads:
        res_p = run_hollow_compression_partial(
            layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, p_val
        )
        P0_list.append(float(res_p[4].item(0)))
        w0_tot_list.append(float(res_p[1].item(0) * 1000.0))
        w0_el_list.append(float(res_p[2].item(0) * 1000.0))
        w0_fr_list.append(float(res_p[3].item(0) * 1000.0))
        Pbase_list.append(float(res_p[13]))
        cum_fric_list.append(float(res_p[10].item(-1)))

    wb_grid = np.linspace(0.0001, 0.08, 220)
    peak_capacity = max(P0_list) if len(P0_list) > 0 else 0.0
    for wb in wb_grid:
        res_b = run_hollow_compression_from_base(
            layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, wb
        )
        p_curr = float(res_b[4].item(0))
        w_curr_mm = float(res_b[1].item(0) * 1000.0)
        peak_capacity = max(peak_capacity, p_curr)
        P0_list.append(p_curr)
        w0_tot_list.append(w_curr_mm)
        w0_el_list.append(float(res_b[2].item(0) * 1000.0))
        w0_fr_list.append(float(res_b[3].item(0) * 1000.0))
        Pbase_list.append(float(res_b[13]))
        cum_fric_list.append(float(res_b[10].item(-1)))
        if p_curr >= P_max_user and wb > 0.01:
            break


    st.markdown("---")
    st.subheader("2. Interactive Hollow Pile Evaluation & Plugging Analysis")

    target_P0 = st.slider(
        "Select Evaluation Head Load P0 (kN):",
        min_value=0.0,
        max_value=float(P_max_user),
        value=min(round(P_threshold_zero_toe * 0.6, 1), float(P_max_user)),
        step=10.0,
    )

    if target_P0 <= P_threshold_zero_toe:
        (
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
            P_base_eval,
            plug_status_eval,
        ) = run_hollow_compression_partial(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, target_P0)
    else:
        low_wb, high_wb = 0.0, 0.08
        for _ in range(55):
            mid_wb = 0.5 * (low_wb + high_wb)
            res_tmp = run_hollow_compression_from_base(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, mid_wb)
            if float(res_tmp[4].item(0)) < target_P0:
                low_wb = mid_wb
            else:
                high_wb = mid_wb

        (
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
            P_base_eval,
            plug_status_eval,
        ) = run_hollow_compression_from_base(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, 0.5 * (low_wb + high_wb))

    p0_eval = float(P_node.item(0))
    w0_tot_eval = float(w_total_node.item(0) * 1000.0)
    w0_el_eval = float(w_elastic_node.item(0) * 1000.0)
    w0_fr_eval = float(w_fric_node.item(0) * 1000.0)
    wall_stress_mpa = (p0_eval / A_wall) / 1000.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Head Load P0", f"{p0_eval:,.1f} kN", f"Wall Stress: {wall_stress_mpa:.1f} MPa")
    m2.metric("Total Head Settlement w0", f"{w0_tot_eval:.2f} mm")
    m3.metric("Mobilized Base Load P_base", f"{P_base_eval:,.1f} kN", f"Shaft: {p0_eval - P_base_eval:,.0f} kN")
    m4.metric("Ultimate Capacity P_ult", f"{peak_capacity:,.1f} kN", f"w_base=0 limit: {P_threshold_zero_toe:,.0f} kN")

    if target_P0 > p0_eval + 5.0:
        st.error(f"כשל בלחיצה: העומס המבוקש ({target_P0:,.0f} kN) עובר את תסבולת הכלונס החלול ({peak_capacity:,.1f} kN).")

    st.info(
        f"Plug Status: **{plug_status_eval}** | Elastic Wall Shortening = {w0_el_eval:.2f} mm | Soil/Base Settlement = {w0_fr_eval:.2f} mm"
    )

    c_vis, c_curve = st.columns([1.0, 1.45])
    with c_vis:
        fig_soil = plot_hollow_pile_cross_section(layers_df, depths, D_out, t_wall, end_mode, H_plug, P_node, t_mob_seg, state_seg)
        st.pyplot(fig_soil)

    with c_curve:
        fig_ls, ax_a = plt.subplots(figsize=(6.8, 5.2))
        ax_a.plot(w0_tot_list, P0_list, label="Model: Total Settlement (w_total)", color="#1f497d", linewidth=2.4)
        ax_a.plot(w0_el_list, P0_list, "--", label="Model: Elastic Wall Shortening (w_elastic)", color="#2e75b6")
        ax_a.plot(w0_fr_list, P0_list, "-.", label="Model: Shaft/Toe Slip (w_fric + w_base)", color="#e67e22")

        clean_field = field_test_df.dropna()
        if len(clean_field) > 0:
            ax_a.plot(
                clean_field["Measured Head Settlement w0 (mm)"],
                clean_field["Measured Head Load P0 (kN)"],
                "o-",
                color="#8e44ad",
                linewidth=1.8,
                markersize=6,
                label="Field Load Test (Measured)",
            )

        ax_a.axhline(peak_capacity, color="green", linestyle=":", label=f"Ultimate Capacity ({peak_capacity:.0f} kN)")
        ax_a.scatter([w0_tot_eval], [p0_eval], color="red", s=80, zorder=6, label="Current Evaluation Point")
        ax_a.set_ylim(0, max(P_max_user, peak_capacity) * 1.08)
        ax_a.set_xlim(0, 50)
        ax_a.set_xlabel("Pile Head Settlement w0 (mm)", fontname="Times New Roman")
        ax_a.set_ylabel("Pile Head Load P0 (kN)", fontname="Times New Roman")
        ax_a.set_title("Hollow Pile Load-Settlement Curve", fontname="Times New Roman", fontweight="bold")
        ax_a.grid(True, linestyle="--", alpha=0.5)
        ax_a.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_ls)

    st.markdown("---")
    st.subheader("2B. 10-Increment Simulated Compression Load Test (Instrumented Hollow Pile)")

    col_sim1, _ = st.columns([1.2, 1.8])
    with col_sim1:
        sim_max_load = st.number_input(
            "Enter Maximum Load for 10-Step Simulation (kN):",
            min_value=50.0,
            max_value=50000.0,
            value=float(P_max_user),
            step=100.0,
            key="sim_hollow_max_input",
        )
    run_10_steps_btn = st.button("Run 10-Step Hollow Pile Test Simulation", type="primary")
    fig_10_for_excel = None
    fig_10w_for_excel = None
    fig_10s_for_excel = None

    if run_10_steps_btn or "sim_h10_df" in st.session_state:
        step_loads = np.linspace(0.0, sim_max_load, 11)
        sim_records = []
        step_profiles = []

        for step_idx, p_req in enumerate(step_loads):
            if p_req <= P_threshold_zero_toe:
                res_s = run_hollow_compression_partial(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, p_req)
                failed_step = False
            else:
                low_wb, high_wb = 0.0, 0.08
                for _ in range(55):
                    mid_wb = 0.5 * (low_wb + high_wb)
                    res_tmp = run_hollow_compression_from_base(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, mid_wb)
                    if float(res_tmp[4].item(0)) < p_req:
                        low_wb = mid_wb
                    else:
                        high_wb = mid_wb
                res_s = run_hollow_compression_from_base(layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, 0.5 * (low_wb + high_wb))
                failed_step = p_req > float(res_s[4].item(0)) + 5.0

            d_s, wt_s, we_s, wf_s, Pn_s = res_s[0], res_s[1], res_s[2], res_s[3], res_s[4]
            p_act = float(Pn_s.item(0))
            wt_mm = float(wt_s.item(0) * 1000.0)
            we_mm = float(we_s.item(0) * 1000.0)
            st_msg = "OK (Partial)" if p_req <= P_threshold_zero_toe else ("Failure!" if failed_step else str(res_s[14]))

            sim_records.append({
                "Step": step_idx,
                "Target Load (kN)": round(float(p_req), 1),
                "Achieved Load P0 (kN)": round(p_act, 1),
                "Head Settlement w0 (mm)": round(wt_mm, 2),
                "Elastic Shortening (mm)": round(we_mm, 2),
                "Base Load P_base (kN)": round(float(res_s[13]), 1),
                "Status / Plug": st_msg,
            })

            if not failed_step:
                step_profiles.append({
                    "step": step_idx,
                    "P0": p_act,
                    "depths": d_s,
                    "w_mm": wt_s * 1000.0,
                    "sigma_wall_kPa": Pn_s / A_wall,
                    "P_kN": Pn_s,
                })

        st.session_state["sim_h10_df"] = pd.DataFrame(sim_records)
        st.session_state["sim_h10_profiles"] = step_profiles

        sim_h10_df = st.session_state["sim_h10_df"]
        c_tbl10, c_plt10 = st.columns([1.2, 1.2])
        with c_tbl10:
            st.dataframe(sim_h10_df, use_container_width=True, hide_index=True)

        with c_plt10:
            fig_10, ax_10 = plt.subplots(figsize=(6.2, 4.5))
            ax_10.plot(P0_list, w0_tot_list, color="#aeb6bf", linestyle="--", linewidth=1.5, label="Continuous Model Curve")
            ax_10.plot(
                sim_h10_df["Achieved Load P0 (kN)"],
                sim_h10_df["Head Settlement w0 (mm)"],
                "s-",
                color="#d35400",
                linewidth=2.2,
                markersize=7,
                label="10-Step Simulated Test",
            )
            ax_10.set_xlabel("Pile Head Load P0 (kN)", fontname="Times New Roman")
            ax_10.set_ylabel("Pile Head Settlement w0 (mm)", fontname="Times New Roman")
            ax_10.set_title("10-Step Simulated Hollow Pile Load Test", fontname="Times New Roman", fontweight="bold")
            ax_10.grid(True, linestyle="--", alpha=0.5)
            ax_10.legend(fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig_10)
            fig_10_for_excel = fig_10
        if len(step_profiles) > 0:
            st.markdown("#### Depth Profiles Across Pre-Failure Load Steps (Including Pile Toe)")
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
                        label=f"Step {prof['step']}: P0={prof['P0']:.0f} kN",
                    )
                ax_10w.invert_yaxis()
                ax_10w.set_xlabel("Axial Displacement w(z) [mm]", fontweight="bold", fontname="Times New Roman")
                ax_10w.set_ylabel("Depth z [m]", fontweight="bold", fontname="Times New Roman")
                ax_10w.set_title("Displacement Along Hollow Pile & Toe", fontweight="bold", fontname="Times New Roman")
                ax_10w.grid(True, linestyle="--", alpha=0.5)
                ax_10w.legend(fontsize=7.5, loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_10w)
                fig_10w_for_excel = fig_10w

            with col_10s:
                fig_10s, ax_10s = plt.subplots(figsize=(6.2, 5.2))
                for idx_p, prof in enumerate(step_profiles):
                    ax_10s.plot(
                        prof["sigma_wall_kPa"],
                        prof["depths"],
                        marker="^",
                        markersize=4.5,
                        linewidth=1.8,
                        color=cmap_steps(idx_p),
                        label=f"Step {prof['step']}: P0={prof['P0']:.0f} kN",
                    )
                ax_10s.invert_yaxis()
                ax_10s.set_xlabel("Wall Axial Stress σ_wall(z) = P(z)/A_wall [kPa]", fontweight="bold", fontname="Times New Roman")
                ax_10s.set_ylabel("Depth z [m]", fontweight="bold", fontname="Times New Roman")
                ax_10s.set_title("Wall Stress Along Hollow Pile & Toe", fontweight="bold", fontname="Times New Roman")
                ax_10s.grid(True, linestyle="--", alpha=0.5)
                ax_10s.legend(fontsize=7.5, loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_10s)
                fig_10s_for_excel = fig_10s

    st.subheader("3. Combined Depth Profile: Axial Load & Settlement Along Hollow Pile")
    fig_comb, ax_load = plt.subplots(figsize=(10.5, 5.2))

    for i in range(len(layers_df)):
        z_top, z_bot = depths[i], depths[i + 1]
        s_type = str(layers_df.at[i, "Soil Type"]).strip()
        bg_col = SOIL_COLORS.get(s_type, "#f2f4f4")
        ax_load.axhspan(z_top, z_bot, facecolor=bg_col, alpha=0.35)
        ax_load.text(
            max(float(P_node.item(0)) * 0.02, 5.0),
            0.5 * (z_top + z_bot),
            f"{s_type} (τ={t_mob_seg.item(i):.1f} kPa | F_in={F_in_seg.item(i):.0f} kN)",
            va="center",
            ha="left",
            fontsize=8,
            color="#2c3e50",
            alpha=0.85,
        )

    line1 = ax_load.plot(P_node, depths, marker="s", markersize=7, linewidth=2.6, color="#c0392b", label="Axial Load P(z) [kN]")
    ax_load.set_xlabel("Axial Load Along Hollow Pile P(z) [kN]", color="#c0392b", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_load.tick_params(axis="x", labelcolor="#c0392b")
    ax_load.set_ylabel("Depth z [m]", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_load.set_ylim(depths[-1], 0.0)
    ax_load.set_xlim(0.0, max(float(P_node.item(0)) * 1.15, 50.0))
    ax_load.grid(True, linestyle="--", alpha=0.45)

    ax_disp = ax_load.twiny()
    line2 = ax_disp.plot(
        w_total_node * 1000.0, depths, marker="o", markersize=6, linestyle="-.", linewidth=2.4, color="#1f497d", label="Settlement w(z) [mm]"
    )
    ax_disp.set_xlabel("Settlement Along Hollow Pile w(z) [mm]", color="#1f497d", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_disp.tick_params(axis="x", labelcolor="#1f497d")
    ax_disp.set_xlim(0.0, max(float(w_total_node.item(0)) * 1000.0 * 1.15, 1.0))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_load.legend(lines, labels, loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax_load.set_title(
        f"Hollow Pile Load Transfer & Settlement (P0 = {p0_eval:,.1f} kN | P_base = {P_base_eval:,.1f} kN | w0 = {w0_tot_eval:.2f} mm)",
        fontsize=11,
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
    st.subheader("5. Export Full Hollow Pile Engineering Report to Excel (Print-Ready)")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        summary_rows = (
            ("Analysis Mode", "Hollow Pile - Compression (Pure Physics)"),
            ("Pile Material", pile_mat),
            ("Elastic Modulus E (kPa)", f"{E:,.0f}"),
            ("Outer Diameter D_out (m)", f"{D_out:.3f}"),
            ("Wall Thickness t_wall (mm)", f"{t_wall_mm:.1f}"),
            ("Inner Diameter D_in (m)", f"{D_in:.3f}"),
            ("Annular Wall Area A_wall (m²)", f"{A_wall:.4f}"),
            ("End & Plug Condition", end_mode),
            ("Internal Soil Plug Height H_plug (m)", f"{H_plug:.2f}"),
            ("Active Plug Behavior", plug_status_eval),
            ("Evaluation Head Load P0 (kN)", f"{p0_eval:,.1f}"),
            ("Mobilized Base Load P_base (kN)", f"{P_base_eval:,.1f}"),
            ("Total Head Settlement w0 (mm)", f"{w0_tot_eval:.2f}"),
            ("Ultimate Capacity P_ult (kN)", f"{peak_capacity:,.1f}"),
        )
        summary_df = pd.DataFrame.from_records(summary_rows, columns=("Parameter", "Value"))
        summary_df.to_excel(writer, sheet_name="Summary & Results", startrow=2, index=False)

        export_layers_df = pd.DataFrame({
            "Layer": range(1, len(layers_df) + 1),
            "Soil Type": layers_df["Soil Type"],
            "Depth (m)": [f"{depths[i]:.1f} - {depths[i+1]:.1f}" for i in range(len(layers_df))],
            "Displacement (mm)": w_total_node[:-1] * 1000.0,
            "Elastic Shortening (mm)": dw_elastic_seg * 1000.0,
            "Mobilized Shear (kPa)": t_mob_seg,
            "Outer Friction (kN)": F_out_seg,
            "Inner Plug Friction (kN)": F_in_seg,
            "Axial Load at Top (kN)": P_node[:-1],
            "Wall Stress (MPa)": (P_node[:-1] / A_wall) / 1000.0,
            "State": state_seg,
        })
        export_layers_df.to_excel(writer, sheet_name="Summary & Results", startrow=19, index=False)

        ws_res = writer.sheets["Summary & Results"]
        prof_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#2e86c1", "font_name": "Times New Roman"})
        title_fmt = workbook.add_format({"bold": True, "font_size": 13, "font_color": "#1f497d", "font_name": "Times New Roman"})

        ws_res.write(0, 0, "Prof. Uri Komornik", prof_fmt)
        ws_res.write(1, 0, "Hollow Pile (Compression) Geomechanical Report", title_fmt)
        ws_res.set_column("A:B", 30)
        ws_res.set_column("C:L", 16)
        ws_res.set_landscape()
        ws_res.fit_to_pages(1, 0)

        layers_df.to_excel(writer, sheet_name="Soil Input Data", startrow=2, index=False)
        ws_in = writer.sheets["Soil Input Data"]
        ws_in.write(0, 0, "Prof. Uri Komornik", prof_fmt)
        ws_in.write(1, 0, "Input Soil Profile & Pressuremeter Parameters", title_fmt)
        curve_df = pd.DataFrame({
            "Head Load P0 (kN)": P0_list,
            "Total Head Settlement w0 (mm)": w0_tot_list,
            "Elastic Shortening (mm)": w0_el_list,
            "Shaft/Base Slip (mm)": w0_fr_list,
            "Base Load P_base (kN)": Pbase_list,
            "Cumulative Shaft Friction (kN)": cum_fric_list,
        })
        curve_df.to_excel(writer, sheet_name="Load-Settlement Curve", startrow=2, index=False)
        ws_curve = writer.sheets["Load-Settlement Curve"]
        ws_curve.write(0, 0, "Prof. Uri Komornik", prof_fmt)
        ws_curve.write(1, 0, "Calculated Hollow Pile Load-Settlement Curve", title_fmt)
        ws_curve.set_column("A:F", 24)

        if "sim_h10_df" in st.session_state:
            st.session_state["sim_h10_df"].to_excel(writer, sheet_name="10-Step Load Sim", startrow=2, index=False)
            ws_sim = writer.sheets["10-Step Load Sim"]
            ws_sim.write(0, 0, "Prof. Uri Komornik", prof_fmt)
            ws_sim.write(1, 0, "10-Increment Simulated Hollow Pile Load Test", title_fmt)
            ws_sim.set_column("A:G", 20)

        if "sim_h10_profiles" in st.session_state and len(st.session_state["sim_h10_profiles"]) > 0:
            prof_rows = []
            for prof in st.session_state["sim_h10_profiles"]:
                for z_val, w_val, sig_val, p_val in zip(prof["depths"], prof["w_mm"], prof["sigma_wall_kPa"], prof["P_kN"]):
                    prof_rows.append({
                        "Step": prof["step"],
                        "Head Load P0 (kN)": round(prof["P0"], 1),
                        "Depth z (m)": round(float(z_val), 2),
                        "Displacement w(z) (mm)": round(float(w_val), 3),
                        "Wall Stress σ_wall(z) (kPa)": round(float(sig_val), 1),
                        "Axial Load P(z) (kN)": round(float(p_val), 1),
                    })
            pd.DataFrame(prof_rows).to_excel(writer, sheet_name="10-Step Depth Profiles", startrow=2, index=False)
            ws_p10 = writer.sheets["10-Step Depth Profiles"]
            ws_p10.write(0, 0, "Prof. Uri Komornik", prof_fmt)
            ws_p10.write(1, 0, "Depth Profiles Across Pre-Failure Load Steps", title_fmt)
            ws_p10.set_column("A:F", 22)

        ws_charts = workbook.add_worksheet("Engineering Charts")
        ws_charts.write(0, 0, "Prof. Uri Komornik", prof_fmt)
        ws_charts.write(1, 0, "Hollow Pile (Compression) Graphical Output", title_fmt)
        ws_charts.set_landscape()
        ws_charts.fit_to_pages(1, 5)

        chart_list = [
            (fig_soil, "A4", 0.62),
            (fig_ls, "J4", 0.62),
            (fig_comb, "A30", 0.62),
            (fig_ds, "A56", 0.62),
        ]
        if fig_10_for_excel is not None:
            chart_list.append((fig_10_for_excel, "A82", 0.62))
        if fig_10w_for_excel is not None:
            chart_list.append((fig_10w_for_excel, "A108", 0.62))
        if fig_10s_for_excel is not None:
            chart_list.append((fig_10s_for_excel, "J108", 0.62))

        for idx_c, (fig_obj, cell_pos, scale_val) in enumerate(chart_list):
            img_buf = io.BytesIO()
            fig_obj.savefig(img_buf, format="png", dpi=180, bbox_inches="tight")
            img_buf.seek(0)
            ws_charts.insert_image(
                cell_pos,
                f"chart_hc_{idx_c}.png",
                {
                    "image_data": img_buf,
                    "x_scale": scale_val,
                    "y_scale": scale_val,
                },
            )

    st.download_button(
        label="Download Full Hollow Pile Excel Report (.xlsx)",
        data=excel_buffer.getvalue(),
        file_name=f"Hollow_Compression_Pile_D{D_out}_t{t_wall_mm:.0f}mm.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

