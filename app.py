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
        color: #1f497d;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 4px;
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
    w_mm_arr = df_sorted.get("Base Settlement (mm)").values
    p_kn_arr = df_sorted.get("Base Load (kN)").values
    return float(np.interp(w_base_m * 1000.0, w_mm_arr, p_kn_arr))


def run_pile_model_partial_load(layers_df, E, D, target_P0):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    depths = [0.0] + layers_df.get("Bottom Depth (m)").tolist()
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

    np.put(w_total_node, 0, 0.5 * (low_w + high_w))
    np.put(P_node, 0, target_P0)

    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.at[idx, "Initial Pressuremeter σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.at[idx, "Peak Friction δ_peak (°)"]))
        t_r = sig_h * np.tan(np.radians(layers_df.at[idx, "Residual Friction δ_res (°)"]))
        w_p = layers_df.at[idx, "Peak Disp w_peak (mm)"] / 1000.0
        w_r = layers_df.at[idx, "Residual Disp w_res (mm)"] / 1000.0

        if P_node.item(idx) <= 1e-6:
            tm, st_txt = 0.0, "No Displacement"
            ff = 0.0
            np.put(P_node, idx + 1, 0.0)
            dw_el = 0.0
            np.put(w_total_node, idx + 1, 0.0)
        else:
            w_bot_est = max(w_total_node.item(idx) - (P_node.item(idx) * dz_s) / (E * A), 0.0)
            w_mid = 0.5 * (w_total_node.item(idx) + w_bot_est)
            tm, st_txt = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
            ff_max = U * dz_s * tm
            ff = min(P_node.item(idx), ff_max)
            if ff < ff_max and P_node.item(idx) > 0:
                tm = ff / (U * dz_s)
            np.put(P_node, idx + 1, max(P_node.item(idx) - ff, 0.0))
            P_avg = 0.5 * (P_node.item(idx) + P_node.item(idx + 1))
            dw_el = (P_avg * dz_s) / (E * A)
            np.put(w_total_node, idx + 1, max(w_total_node.item(idx) - dw_el, 0.0))

        np.put(t_peak_seg, idx, t_p)
        np.put(t_res_seg, idx, t_r)
        np.put(t_mob_seg, idx, tm)
        np.put(F_fric_seg, idx, ff)
        np.put(dw_elastic_seg, idx, dw_el)
        state_seg[idx] = st_txt

    w_elastic_node = np.append(np.cumsum(dw_elastic_seg[::-1])[::-1], 0.0)
    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_fric_seg)
    q_base = float(P_node.item(-1)) / A

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
    depths = [0.0] + layers_df.get("Bottom Depth (m)").tolist()
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
        if float(P_node_res.item(-1)) <= 1e-3:
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

    np.put(w_total_node, num_layers, w_base_m)
    np.put(w_elastic_node, num_layers, 0.0)
    np.put(
        P_node,
        num_layers,
        calc_base_load(w_base_m, base_model_type, A, B_parab, q_ult_hyp, k_init_hyp, base_table_df),
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

        w_top_est = w_total_node.item(i + 1) + (P_node.item(i + 1) * dz) / (E * A)
        w_mid = 0.5 * (w_total_node.item(i + 1) + w_top_est)

        t_mob, state = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
        F_fric = U * dz * t_mob

        np.put(P_node, i, P_node.item(i + 1) + F_fric)
        P_mean = 0.5 * (P_node.item(i) + P_node.item(i + 1))

        dw_el = (P_mean * dz) / (E * A)
        np.put(dw_elastic_seg, i, dw_el)
        np.put(w_elastic_node, i, w_elastic_node.item(i + 1) + dw_el)
        np.put(w_total_node, i, w_total_node.item(i + 1) + dw_el)

        np.put(t_peak_seg, i, t_p)
        np.put(t_res_seg, i, t_r)
        np.put(t_mob_seg, i, t_mob)
        np.put(F_fric_seg, i, F_fric)
        state_seg[i] = state

    w_fric_node = np.maximum(w_total_node - w_elastic_node, 0.0)
    cum_fric_seg = np.cumsum(F_fric_seg)
    q_base = float(P_node.item(-1)) / A

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
        for key, val in SOIL_COLORS.items():
            if key.lower() in s_type.lower():
                color = val
                break

        rect_soil = patches.Rectangle((-3.0, z_top), 6.0, dz, facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.9)
        ax.add_patch(rect_soil)

        z_mid = 0.5 * (z_top + z_bot)
        ax.text(-2.85, z_mid, f"{s_type}\nσ'h = {sig_h:.0f} kPa", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#2c3e50")
        ax.text(2.85, z_mid, f"t = {t_mob_seg.item(i):.1f} kPa\n[{state_seg[i]}]", va="center", ha="right", fontsize=8, color="#1a5276")

    pile_half_w = 0.45
    rect_pile = patches.Rectangle((-pile_half_w, 0.0), pile_half_w * 2, max_depth, facecolor="#7f8c8d", edgecolor="#2c3e50", linewidth=2, zorder=4)
    ax.add_patch(rect_pile)

    p_head_val = float(P_node.item(0))
    ax.annotate(
        f"P0 = {p_head_val:,.0f} kN",
        xy=(0, 0),
        xytext=(0, -max_depth * 0.12),
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#c0392b",
        arrowprops=dict(facecolor="#c0392b", edgecolor="#c0392b", width=2.5, headwidth=8),
    )

    ax.annotate(
        f"P_base = {float(P_node.item(-1)):,.0f} kN (q = {q_base:,.0f} kPa)",
        xy=(0, max_depth),
        xytext=(0, max_depth + max_depth * 0.13),
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#117864",
        arrowprops=dict(facecolor="#117864", edgecolor="#117864", width=2, headwidth=7),
    )

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(max_depth + max_depth * 0.18, -max_depth * 0.18)
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)", fontname="Times New Roman")
    ax.set_title(f"Pile Cross-Section (D = {D}m, L = {max_depth:.1f}m)", fontsize=11, fontweight="bold", fontname="Times New Roman")

    plt.tight_layout()
    return fig


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


st.sidebar.header("1. Pile Properties & Soil Preset")
pile_mat = st.sidebar.selectbox("Pile Material", ("Concrete", "Steel", "HDPE Plastic", "Wood"))
e_map = {"Concrete": 30000000, "Steel": 210000000, "HDPE Plastic": 1500000, "Wood": 11000000}
default_E = e_map.get(pile_mat)
E = st.sidebar.number_input("Elastic Modulus E (kPa)", value=default_E, step=1000000)
D = st.sidebar.number_input("Pile Diameter D (m)", min_value=0.2, max_value=3.0, value=0.6, step=0.05)
L_pile = st.sidebar.number_input("Pile Length L (m)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)

soil_preset = st.sidebar.selectbox(
    "Soil Profile Preset",
    ("Multi-layer (Fill, Silt, Marl, Clay, Dense Sand)", "Dense Sand", "Loose Sand", "Clay", "Marl", "Silt"),
)

f_map = {"Concrete": (1.0, 1.0), "Steel": (0.67, 0.76), "HDPE Plastic": (0.45, 0.55), "Wood": (0.85, 0.85)}
f_peak, f_res = f_map.get(pile_mat)
default_layers = build_preset_df(soil_preset, f_peak, f_res)
default_layers["Bottom Depth (m)"] = (default_layers["Bottom Depth (m)"] * (L_pile / 10.0)).round(2)

st.sidebar.markdown("---")
st.sidebar.header("2. Load Range & Base Resistance")
P_max_user = st.sidebar.number_input("Max Load for Curve P_max (kN):", min_value=100.0, max_value=50000.0, value=5000.0, step=100.0)

base_model_type = st.sidebar.radio(
    "Base Resistance Model:",
    ("Hyperbolic (Classic Load Test)", "Parabolic (B * sqrt(w))", "Custom Load Test Table"),
)

B_parab, q_ult_hyp, k_init_hyp = 1500.0, 3500.0, 150000.0
base_rows = ((0.0, 0.0), (2.0, 180.0), (5.0, 380.0), (10.0, 620.0), (20.0, 900.0), (40.0, 1150.0), (80.0, 1300.0))
default_base_table = pd.DataFrame.from_records(base_rows, columns=("Base Settlement (mm)", "Base Load (kN)"))

if base_model_type == "Hyperbolic (Classic Load Test)":
    q_ult_hyp = st.sidebar.number_input("Ultimate Base Stress q_ult (kPa)", value=3500.0, step=250.0)
    k_init_hyp = st.sidebar.number_input("Initial Base Stiffness k_init (kPa/m)", value=150000.0, step=10000.0)
elif base_model_type == "Parabolic (B * sqrt(w))":
    B_parab = st.sidebar.number_input("Parabolic Base Factor B (kN/m^0.5)", value=1500.0, step=100.0)
else:
    st.sidebar.write("Edit Base Load Test Table:")
    default_base_table = st.sidebar.data_editor(default_base_table, num_rows="dynamic", hide_index=True)

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

with st.expander("Optional: Field Load Test Data for Comparison on Graph"):
    field_rows = ((0.0, 0.0), (400.0, 1.2), (800.0, 3.1), (1200.0, 6.5), (1600.0, 14.0))
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
    F_fric_at_zero_base = find_max_friction_load_zero_base(layers_df, E, D)
    P0_list, w0_tot_list, w0_el_list, w0_fr_list, Pbase_list, cum_fric_list = [], [], [], [], [], []
    partial_loads = []

    if F_fric_at_zero_base > 0:
        partial_loads = np.linspace(0.0, min(F_fric_at_zero_base, P_max_user), 40)

    for p_val in partial_loads:
        _, wt, we, wf, Pn, _, _, _, _, cf, _, _, _ = run_pile_model_partial_load(layers_df, E, D, p_val)
        P0_list.append(float(Pn.item(0)))
        w0_tot_list.append(float(wt.item(0) * 1000.0))
        w0_el_list.append(float(we.item(0) * 1000.0))
        w0_fr_list.append(float(wf.item(0) * 1000.0))
        Pbase_list.append(0.0)
        cum_fric_list.append(float(cf.item(-1)))

    if P_max_user > F_fric_at_zero_base:
        wb = 0.0001
        step_wb = 0.0004
        for _ in range(300):
            _, wt, we, wf, Pn, _, _, _, _, cf, _, _, _ = run_pile_model_from_base(
                layers_df, E, D, wb, base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table
            )
            p_curr = float(Pn.item(0))
            if p_curr > F_fric_at_zero_base:
                P0_list.append(p_curr)
                w0_tot_list.append(float(wt.item(0) * 1000.0))
                w0_el_list.append(float(we.item(0) * 1000.0))
                w0_fr_list.append(float(wf.item(0) * 1000.0))
                Pbase_list.append(float(Pn.item(-1)))
                cum_fric_list.append(float(cf.item(-1)))

            if p_curr >= P_max_user or wb >= 0.5:
                break
            wb += step_wb
            if wb > 0.02:
                step_wb = 0.002
            if wb > 0.10:
                step_wb = 0.005

    st.markdown("---")
    st.subheader("2. Interactive Load Evaluation & Simulation Results")
    target_P0 = st.slider(
        "Select Evaluation Head Load P0 (kN):",
        min_value=0.0,
        max_value=float(P_max_user),
        value=min(round(F_fric_at_zero_base * 0.5, 1), float(P_max_user)),
        step=10.0,
    )

       if target_P0 <= F_fric_at_zero_base:
        (
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
        ) = run_pile_model_partial_load(layers_df, E, D, target_P0)
    else:
        low_wb, high_wb = 0.0, 0.5
        for _ in range(55):
            mid_wb = 0.5 * (low_wb + high_wb)
            res_tmp = run_pile_model_from_base(
                layers_df, E, D, mid_wb, base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table
            )
            P_node_tmp = res_tmp[4]
            if float(P_node_tmp.item(0)) < target_P0:
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
            F_fric_seg,
            cum_fric_seg,
            dw_elastic_seg,
            q_base,
            state_seg,
        ) = run_pile_model_from_base(
            layers_df, E, D, 0.5 * (low_wb + high_wb), base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table
        )

    p0_eval = float(P_node.item(0))
    w0_tot_eval = float(w_total_node.item(0) * 1000.0)
    w0_el_eval = float(w_elastic_node.item(0) * 1000.0)
    w0_fr_eval = float(w_fric_node.item(0) * 1000.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Head Load P0", f"{p0_eval:,.1f} kN", f"{p0_eval/10.0:,.1f} ton")
    m2.metric("Total Head Disp w0", f"{w0_tot_eval:.2f} mm")
    m3.metric("Base Load P_base", f"{float(P_node.item(-1)):,.1f} kN", f"q_base: {q_base:,.0f} kPa")
    m4.metric("Friction Load (q_base=0)", f"{F_fric_at_zero_base:,.1f} kN")

    if target_P0 > p0_eval + 5.0:
        st.error(f"כשל (הרס מלא): העומס המבוקש ({target_P0:,.0f} kN) גבוה מתסבולת ההרס של הכלונס ({p0_eval:,.1f} kN).")

    st.info(
        f"Head Displacement Breakdown: Elastic = {w0_el_eval:.2f} mm | Friction/Soil = {w0_fr_eval:.2f} mm | Total = {w0_tot_eval:.2f} mm"
    )

    c_vis, c_curve = st.columns([1.0, 1.45])
    fig_soil = plot_pile_soil_cross_section(layers_df, depths, D, P_node, t_mob_seg, state_seg, q_base)
    c_vis.pyplot(fig_soil)

    fig_ls, ax_a = plt.subplots(figsize=(6.8, 5.2))
    ax_a.plot(w0_tot_list, P0_list, label="Model: Total Disp (w_total)", color="#1f497d", linewidth=2.4)
    ax_a.plot(w0_el_list, P0_list, "--", label="Model: Elastic Shortening (w_elastic)", color="#2e75b6")
    ax_a.plot(w0_fr_list, P0_list, "-.", label="Model: Friction/Base Disp (w_fric)", color="#e67e22")

    clean_field = field_test_df.dropna()
    if len(clean_field) > 0:
        ax_a.plot(
            clean_field.get("Measured Head Settlement w0 (mm)"),
            clean_field.get("Measured Head Load P0 (kN)"),
            "o-",
            color="#8e44ad",
            linewidth=1.8,
            markersize=6,
            label="Field Load Test (Measured)",
        )

    ax_a.axhline(F_fric_at_zero_base, color="green", linestyle=":", label=f"Friction Limit q_base=0 ({F_fric_at_zero_base:.0f} kN)")
    ax_a.scatter([w0_tot_eval], [p0_eval], color="red", s=80, zorder=6, label="Current Evaluation Point")
    ax_a.set_ylim(0, P_max_user * 1.05)
    ax_a.set_xlabel("Pile Head Settlement w0 (mm)", fontname="Times New Roman")
    ax_a.set_ylabel("Pile Head Load P0 (kN)", fontname="Times New Roman")
    ax_a.set_title(f"Load-Settlement Curve (0 to {P_max_user:,.0f} kN)", fontname="Times New Roman", fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=8)
    plt.tight_layout()
    c_curve.pyplot(fig_ls)

    st.markdown("---")
    st.subheader("3. Combined Depth Profile: Axial Load & Settlement Along the Pile")
    fig_comb, ax_load = plt.subplots(figsize=(10.5, 5.2))

    for i in range(len(layers_df)):
        z_top, z_bot = depths[i], depths[i + 1]
        s_type = str(layers_df.at[i, "Soil Type"]).strip()
        bg_col = SOIL_COLORS.get(s_type, "#f2f4f4")
        ax_load.axhspan(z_top, z_bot, facecolor=bg_col, alpha=0.35)
        ax_load.text(
            max(float(P_node.item(0)) * 0.02, 5.0),
            0.5 * (z_top + z_bot),
            f"{s_type} (τ={t_mob_seg.item(i):.1f} kPa)",
            va="center",
            ha="left",
            fontsize=8,
            color="#2c3e50",
            alpha=0.85,
        )

    line1 = ax_load.plot(P_node, depths, marker="^", markersize=7, linewidth=2.6, color="#8e44ad", label="Axial Load P(z) [kN]")
    ax_load.set_xlabel("Axial Load Along Pile P(z) [kN]", color="#8e44ad", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_load.tick_params(axis="x", labelcolor="#8e44ad")
    ax_load.set_ylabel("Depth z [m]", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_load.set_ylim(depths[-1], 0.0)
    ax_load.set_xlim(0.0, max(float(P_node.item(0)) * 1.15, 50.0))
    ax_load.grid(True, linestyle="--", alpha=0.45)

    ax_disp = ax_load.twiny()
    line2 = ax_disp.plot(
        w_total_node * 1000.0, depths, marker="o", markersize=6, linestyle="-.", linewidth=2.4, color="#1f497d", label="Total Settlement w(z) [mm]"
    )
    ax_disp.set_xlabel("Settlement Along Pile w(z) [mm]", color="#1f497d", fontweight="bold", fontsize=10.5, fontname="Times New Roman")
    ax_disp.tick_params(axis="x", labelcolor="#1f497d")
    ax_disp.set_xlim(0.0, max(float(w_total_node.item(0)) * 1000.0 * 1.15, 1.0))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_load.legend(lines, labels, loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax_load.set_title(
        f"Load Transfer & Settlement Distribution Along Pile (P0 = {p0_eval:,.1f} kN | w0 = {w0_tot_eval:.2f} mm)",
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
        float(layers_df.get("Residual Disp w_res (mm)").max()) * 1.4,
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
    st.subheader("5. Export Full Engineering Report to Excel (Print-Ready)")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        summary_rows = (
            ("Pile Material", pile_mat),
            ("Elastic Modulus E (kPa)", f"{E:,.0f}"),
            ("Pile Diameter D (m)", f"{D:.2f}"),
            ("Pile Length L (m)", f"{depths[-1]:.2f}"),
            ("Soil Profile Preset", soil_preset),
            ("Evaluation Head Load P0 (kN)", f"{p0_eval:,.1f}"),
            ("Total Head Settlement w0 (mm)", f"{w0_tot_eval:.2f}"),
            ("Elastic Shortening (mm)", f"{w0_el_eval:.2f}"),
            ("Friction/Base Settlement (mm)", f"{w0_fr_eval:.2f}"),
            ("Base Load P_base (kN)", f"{float(P_node.item(-1)):,.1f}"),
            ("Base Stress q_base (kPa)", f"{q_base:,.1f}"),
            ("Max Friction Load q_base=0 (kN)", f"{F_fric_at_zero_base:,.1f}"),
        )
        summary_df = pd.DataFrame.from_records(summary_rows, columns=("Parameter", "Value"))
        summary_df.to_excel(writer, sheet_name="Summary & Results", startrow=2, index=False)

        export_layers_df = pd.DataFrame({
            "Layer": range(1, len(layers_df) + 1),
            "Soil Type": layers_df.get("Soil Type"),
            "Depth (m)": [f"{depths[i]:.1f} - {depths[i+1]:.1f}" for i in range(len(layers_df))],
            "Total Disp (mm)": w_total_node[:-1] * 1000.0,
            "Elastic Shortening (mm)": dw_elastic_seg * 1000.0,
            "Friction Disp (mm)": w_fric_node[:-1] * 1000.0,
            "Peak Shear (kPa)": t_peak_seg,
            "Residual Shear (kPa)": t_res_seg,
            "Mobilized Shear (kPa)": t_mob_seg,
            "Layer Friction (kN)": F_fric_seg,
            "Axial Load at Top (kN)": P_node[:-1],
            "State": state_seg,
        })
        export_layers_df.to_excel(writer, sheet_name="Summary & Results", startrow=17, index=False)

        ws_res = writer.sheets.get("Summary & Results")
        prof_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#1f497d", "font_name": "Times New Roman"})
        
