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
        _, wt, we, wf, Pn, _, _, _, _, _, cf, _, _, pb, _ = run_hollow_compression_partial(
            layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, p_val
        )
        P0_list.append(float(Pn.item(0)))
        w0_tot_list.append(float(wt.item(0) * 1000.0))
        w0_el_list.append(float(we.item(0) * 1000.0))
        w0_fr_list.append(float(wf.item(0) * 1000.0))
        Pbase_list.append(float(pb))
        cum_fric_list.append(float(cf.item(-1)))

    wb_grid = np.linspace(0.0001, 0.08, 220)
    peak_capacity = max(P0_list) if len(P0_list) > 0 else 0.0
    for wb in wb_grid:
        _, wt, we, wf, Pn, _, _, _, _, _, cf, _, _, pb, _ = run_hollow_compression_from_base(
            layers_df, E, D_out, t_wall, end_mode, H_plug, q_ult, k_base, wb
        )
        p_curr = float(Pn.item(0))
        w_curr_mm = float(wt.item(0) * 1000.0)
        peak_capacity = max(peak_capacity, p_curr)
        P0_list.append(p_curr)
        w0_tot_list.append(w_curr_mm)
        w0_el_list.append(float(we.item(0) * 1000.0))
        w0_fr_list.append(float(wf.item(0) * 1000.0))
        Pbase_list.append(float(pb))
        cum_fric_list.append(float(cf.item(-1)))
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

