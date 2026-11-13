import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="מודל כלונס גיאומכני מלא V9.2", layout="wide")
st.title("🏗️ מודל גיאומכני רב-שכבתי לכלונס – חתך שכבות באנגלית והשוואה לניסיון העמסה")
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
        return 0.0, "ללא תזוזה"
    elif w <= w_p:
        return (t_peak / w_p) * w, "אלסטי"
    elif w <= w_r:
        slope = (t_peak - t_res) / max((w_r - w_p), 1e-6)
        return t_peak - slope * (w - w_p), "ריכוך"
    else:
        return t_res, "משתייר"

def calc_base_load(w_base_m, base_model_type, A, B_parab, q_ult_hyp, k_init_hyp, base_table_df):
    if w_base_m <= 1e-9:
        return 0.0
    if base_model_type == "פרבולי (B * sqrt(w))":
        return B_parab * np.sqrt(w_base_m)
    elif base_model_type == "היפרבולי (ניסיון העמסה קלאסי)":
        q_b = w_base_m / ((1.0 / max(k_init_hyp, 1.0)) + (w_base_m / max(q_ult_hyp, 1.0)))
        return q_b * A
    else:
        df_sorted = base_table_df.dropna().sort_values(by="שקיעת בסיס (mm)")
        w_mm_arr = df_sorted["שקיעת בסיס (mm)"].values
        p_kn_arr = df_sorted["עומס בקצה (kN)"].values
        return float(np.interp(w_base_m * 1000.0, w_mm_arr, p_kn_arr))

def run_pile_model_from_base(layers_df, E, D, w_base_m, base_model_type, B_parab, q_ult_hyp, k_init_hyp, base_table_df):
    A = np.pi * (D ** 2) / 4.0
    U = np.pi * D
    num_layers = len(layers_df)
    depths = [0.0] + layers_df["עומק תחתית שכבה (m)"].tolist()

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
        sig_h = layers_df.loc[i, "לחץ פרסיומטר התחלתי σ'h (kPa)"]
        d_peak = layers_df.loc[i, "זווית חיכוך שיא δ_peak (°)"]
        w_p = layers_df.loc[i, "תזוזת שיא w_peak (mm)"] / 1000.0
        d_res = layers_df.loc[i, "זווית משתיירת δ_res (°)"]
        w_r = layers_df.loc[i, "תזוזת התייצבות w_res (mm)"] / 1000.0

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
    depths = [0.0] + layers_df["עומק תחתית שכבה (m)"].tolist()
    n_s = len(layers_df)

    if target_P0 <= 1e-6:
        return (
            depths, np.zeros(n_s + 1), np.zeros(n_s + 1), np.zeros(n_s + 1), np.zeros(n_s + 1),
            np.zeros(n_s), np.zeros(n_s), np.zeros(n_s), np.zeros(n_s), np.zeros(n_s),
            np.zeros(n_s), 0.0, ["ללא תזוזה"] * n_s
        )

    low_w, high_w = 0.0, 0.1
    for _ in range(55):
        mid_w = 0.5 * (low_w + high_w)
        w_curr = mid_w
        P_curr = target_P0
        for idx in range(n_s):
            dz_s = max(depths[idx + 1] - depths[idx], 0.01)
            sig_h = layers_df.loc[idx, "לחץ פרסיומטר התחלתי σ'h (kPa)"]
            t_p = sig_h * np.tan(np.radians(layers_df.loc[idx, "זווית חיכוך שיא δ_peak (°)"]))
            t_r = sig_h * np.tan(np.radians(layers_df.loc[idx, "זווית משתיירת δ_res (°)"]))
            w_p = layers_df.loc[idx, "תזוזת שיא w_peak (mm)"] / 1000.0
            w_r = layers_df.loc[idx, "תזוזת התייצבות w_res (mm)"] / 1000.0
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

    w_total_node = 0.5 * (low_w + high_w)
    P_node = target_P0

    for idx in range(n_s):
        dz_s = max(depths[idx + 1] - depths[idx], 0.01)
        sig_h = layers_df.loc[idx, "לחץ פרסיומטר התחלתי σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.loc[idx, "זווית חיכוך שיא δ_peak (°)"]))
        t_r = sig_h * np.tan(np.radians(layers_df.loc[idx, "זווית משתיירת δ_res (°)"]))
        w_p = layers_df.loc[idx, "תזוזת שיא w_peak (mm)"] / 1000.0
        w_r = layers_df.loc[idx, "תזוזת התייצבות w_res (mm)"] / 1000.0

        if P_node[idx] <= 1e-6:
            tm, st_txt = 0.0, "ללא תזוזה"
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
        sig_h = layers_df.loc[i, "לחץ פרסיומטר התחלתי σ'h (kPa)"]

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
        f"P₀ = {P_node:,.0f} kN",
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
    ax.set_ylabel("עומק (m)")
    ax.set_title(f"חתך כלונס (D = {D}m) ושכבות קרקע", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    return fig

# --- ממשק קלט ---
col_inputs, col_results = st.columns([1.25, 1.75])

with col_inputs:
    st.header("1. תכונות הכלונס וברירת מחדל")
    c1, c2 = st.columns(2)
    with c1:
        pile_mat = st.selectbox("חומר הכלונס", ["בטון מחוספס", "פלדה / ברזל", "פלסטיק (HDPE)", "עץ"])
default_E = {"בטון מחוספס": 30000000, "פלדה / ברזל": 210000000, "פלסטיק (HDPE)": 1500000, "עץ": 11000000}[pile_mat]
E = st.number_input("מודול אלסטיות E (kPa)", value=default_E, step=1000000)
with c2:
    D = st.number_input("קוטר הכלונס D (m)", min_value=0.2, max_value=3.0, value=0.6, step=0.05)
    soil_preset = st.selectbox(
"תבנית פרופיל קרקע להתחלה",
[
"Multi-layer Profile (Fill, Silt, Marl, Clay, Dense Sand)",
"Dense Sand",
"Loose Sand",
"Clay",
"Marl",
"Silt"
]
)
f_peak, f_res = {"בטון מחוספס": (1.0, 1.0), "פלדה / ברזל": (0.67, 0.76), "פלסטיק (HDPE)": (0.45, 0.55), "עץ": (0.85, 0.85)}[pile_mat]
if soil_preset == "Multi-layer Profile (Fill, Silt, Marl, Clay, Dense Sand)":
default_layers = pd.DataFrame({
"עומק תחתית שכבה (m)": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
"Soil Type": ["Fill", "Fill", "Silt", "Silt", "Marl", "Marl", "Clay", "Dense Sand", "Dense Sand", "Dense Sand"],
"לחץ פרסיומטר התחלתי σ'h (kPa)": [40.0, 45.0, 55.0, 65.0, 85.0, 95.0, 105.0, 130.0, 145.0, 160.0],
"זווית חיכוך שיא δ_peak (°)": [
round(25.0 * f_peak, 1), round(25.0 * f_peak, 1),
round(27.0 * f_peak, 1), round(27.0 * f_peak, 1),
round(29.0 * f_peak, 1), round(29.0 * f_peak, 1),
round(24.0 * f_peak, 1),
round(35.8 * f_peak, 1), round(35.8 * f_peak, 1), round(35.8 * f_peak, 1)
],
"תזוזת שיא w_peak (mm)": [4.0, 4.0, 3.5, 3.5, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
"זווית משתיירת δ_res (°)": [
round(24.0 * f_res, 1), round(24.0 * f_res, 1),
round(23.0 * f_res, 1), round(23.0 * f_res, 1),
round(22.0 * f_res, 1), round(22.0 * f_res, 1),
round(16.0 * f_res, 1),
round(30.1 * f_res, 1), round(30.1 * f_res, 1), round(30.1 * f_res, 1)
],
"תזוזת התייצבות w_res (mm)": [9.0, 9.0, 8.5, 8.5, 9.0, 9.0, 10.0, 6.0, 6.0, 6.0]
})
elif soil_preset == "Dense Sand":
default_layers = pd.DataFrame({
"עומק תחתית שכבה (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Dense Sand"] * 10,
"לחץ פרסיומטר התחלתי σ'h (kPa)": [80.0 + i * 10.0 for i in range(10)],
"זווית חיכוך שיא δ_peak (°)": [round(35.8 * f_peak, 1)] * 10,
"תזוזת שיא w_peak (mm)": [2.0] * 10,
"זווית משתיירת δ_res (°)": [round(30.1 * f_res, 1)] * 10,
"תזוזת התייצבות w_res (mm)": [6.0] * 10
})
elif soil_preset == "Loose Sand":
default_layers = pd.DataFrame({
"עומק תחתית שכבה (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Loose Sand"] * 10,
"לחץ פרסיומטר התחלתי σ'h (kPa)": [50.0 + i * 5.0 for i in range(10)],
"זווית חיכוך שיא δ_peak (°)": [round(28.0 * f_peak, 1)] * 10,
"תזוזת שיא w_peak (mm)": [3.5] * 10,
"זווית משתיירת δ_res (°)": [round(26.5 * f_res, 1)] * 10,
"תזוזת התייצבות w_res (mm)": [8.0] * 10
})
elif soil_preset == "Marl":
default_layers = pd.DataFrame({
"עומק תחתית שכבה (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Marl"] * 10,
"לחץ פרסיומטר התחלתי σ'h (kPa)": [75.0 + i * 8.0 for i in range(10)],
"זווית חיכוך שיא δ_peak (°)": [round(29.0 * f_peak, 1)] * 10,
"תזוזת שיא w_peak (mm)": [3.0] * 10,
"זווית משתיירת δ_res (°)": [round(22.0 * f_res, 1)] * 10,
"תזוזת התייצבות w_res (mm)": [9.0] * 10
})
elif soil_preset == "Silt":
default_layers = pd.DataFrame({
"עומק תחתית שכבה (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Silt"] * 10,
"לחץ פרסיומטר התחלתי σ'h (kPa)": [55.0 + i * 6.0 for i in range(10)],
"זווית חיכוך שיא δ_peak (°)": [round(27.0 * f_peak, 1)] * 10,
"תזוזת שיא w_peak (mm)": [3.5] * 10,
"זווית משתיירת δ_res (°)": [round(23.0 * f_res, 1)] * 10,
"תזוזת התייצבות w_res (mm)": [8.5] * 10
})
else:
default_layers = pd.DataFrame({
"עומק תחתית שכבה (m)": [float(i) for i in range(1, 11)],
"Soil Type": ["Clay"] * 10,
"לחץ פרסיומטר התחלתי σ'h (kPa)": [60.0 + i * 5.0 for i in range(10)],
"זווית חיכוך שיא δ_peak (°)": [round(24.0 * f_peak, 1)] * 10,
"תזוזת שיא w_peak (mm)": [3.0] * 10,
"זווית משתיירת δ_res (°)": [round(16.0 * f_res, 1)] * 10,
"תזוזת התייצבות w_res (mm)": [10.0] * 10
})
st.markdown("---")
st.header("2. טווח עומסים להשוואה ותגובת קצה")
P_max_user = st.number_input(
"עומס מקסימלי להצגה בגרף ניסיון העמסה P_max (kN):",
min_value=100.0, max_value=50000.0, value=2500.0, step=100.0
)
base_model_type = st.radio(
"מודל להתנהגות קצה הכלונס:",
["היפרבולי (ניסיון העמסה קלאסי)", "פרבולי (B * sqrt(w))", "טבלת ניסיון העמסה חופשית (קלט מפעיל)"]
)
B_parab, q_ult_hyp, k_init_hyp = 1500.0, 3500.0, 150000.0
default_base_table = pd.DataFrame({
"שקיעת בסיס (mm)": [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
"עומס בקצה (kN)": [0.0, 180.0, 380.0, 620.0, 900.0, 1150.0, 1300.0]
})
if base_model_type == "היפרבולי (ניסיון העמסה קלאסי)":
cb1, cb2 = st.columns(2)
with cb1:
q_ult_hyp = st.number_input("מאמץ קצה גבולי q_ult (kPa)", value=3500.0, step=250.0)
with cb2:
k_init_hyp = st.number_input("קשיחות קצה התחלתית k_init (kPa/m)", value=150000.0, step=10000.0)
elif base_model_type == "פרבולי (B * sqrt(w))":
B_parab = st.number_input("מקדם קשיחות בסיס פרבולי B (kN/m^0.5)", value=1500.0, step=100.0)
else:
default_base_table = st.data_editor(default_base_table, num_rows="dynamic", use_container_width=True, hide_index=True)
with st.expander("📌 אופציונלי: הזנת נקודות מניסיון העמסה אמיתי להשוואה על הגרף"):
field_test_df = st.data_editor(
pd.DataFrame({
"עומס מדוד בראש P₀ (kN)": [0.0, 400.0, 800.0, 1200.0, 1600.0],
"שקיעה מדודה בראש w₀ (mm)": [0.0, 1.2, 3.1, 6.5, 14.0]
}),
num_rows="dynamic", use_container_width=True, hide_index=True
)
st.markdown("---")
st.header("3. נתוני שכבות הקרקע (ניתנים לעריכה)")
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
layers_df = edited_df.dropna(subset=["עומק תחתית שכבה (m)", "לחץ פרסיומטר התחלתי σ'h (kPa)"]).sort_values(by="עומק תחתית שכבה (m)").reset_index(drop=True)

--- חישוב ותצוגת תוצאות ---

if len(layers_df) > 0:
res_zero_base = run_pile_model_from_base(layers_df, E, D, 0.0, base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table)
F_fric_at_zero_base = float(res_zero_base)
P0_list, w0_tot_list, w0_el_list, w0_fr_list, Pbase_list, cum_fric_list = [], [], [], [], [], []
if F_fric_at_zero_base > 0:
partial_loads = np.linspace(0.0, min(F_fric_at_zero_base, P_max_user), 35)
for p_val in partial_loads:
_, wt, we, wf, Pn, _, _, _, _, cf, _, _, _ = run_pile_model_partial_load(layers_df, E, D, p_val)
P0_list.append(float(Pn))
w0_tot_list.append(float(wt * 1000.0))
w0_el_list.append(float(we * 1000.0))
w0_fr_list.append(float(wf * 1000.0))
Pbase_list.append(0.0)
cum_fric_list.append(float(cf[-1]))
if P_max_user > F_fric_at_zero_base:
wb = 0.0002
step_wb = 0.0005
for _ in range(300):
_, wt, we, wf, Pn, _, _, _, _, cf, _, _, _ = run_pile_model_from_base(
layers_df, E, D, wb, base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table
)
p_curr = float(Pn)
P0_list.append(p_curr)
w0_tot_list.append(float(wt * 1000.0))
w0_el_list.append(float(we * 1000.0))
w0_fr_list.append(float(wf * 1000.0))
Pbase_list.append(float(Pn[-1]))
cum_fric_list.append(float(cf[-1]))
if p_curr >= P_max_user or wb >= 0.5:
break
wb += step_wb
if wb > 0.02:
step_wb = 0.002
if wb > 0.10:
step_wb = 0.005
with col_results:
st.header("4. ניתוח מצב הכלונס ותוצאות")
target_P0 = st.slider(
"בחר עומס לבדיקה מפורטת בראש הכלונס P₀ (kN) [10 kN ≈ 1 טון]:",
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
if res_tmp < target_P0:
low_wb = mid_wb
else:
high_wb = mid_wb
(depths, w_total_node, w_elastic_node, w_fric_node, P_node,
t_peak_seg, t_res_seg, t_mob_seg, F_fric_seg, cum_fric_seg,
dw_elastic_seg, q_base, state_seg) = run_pile_model_from_base(
layers_df, E, D, 0.5 * (low_wb + high_wb), base_model_type, B_parab, q_ult_hyp, k_init_hyp, default_base_table
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("עומס בראש P₀", f"{P_node:,.1f} kN", f"{P_node/10.0:,.1f} טון")
m2.metric("תזוזה טוטאלית בראש", f"{w_total_node*1000:.2f} mm")
m3.metric("עומס בקצה P_base", f"{P_node[-1]:,.1f} kN", f"לחץ: {q_base:,.0f} kPa")
m4.metric("גבול חיכוך ללא קצה (q_base=0)", f"{F_fric_at_zero_base:,.1f} kN")
st.info(
f"הפרדת תזוזות בראש הכלונס: "
f"אלסטית = {w_elastic_node*1000:.2f} mm | "
f"חיכוך/קרקע = {w_fric_node*1000:.2f} mm | "
f"טוטאלית = {w_total_node*1000:.2f} mm"
)
c_vis, c_curve = st.columns([1.0, 1.45])
with c_vis:
fig_soil = plot_pile_soil_cross_section(layers_df, depths, D, P_node, t_mob_seg, state_seg, q_base)
st.pyplot(fig_soil)
with c_curve:
fig_ls, ax_a = plt.subplots(figsize=(6.8, 5.2))
ax_a.plot(w0_tot_list, P0_list, label="מודל: תזוזה טוטאלית (w_total)", color="#1f497d", linewidth=2.4)
ax_a.plot(w0_el_list, P0_list, "--", label="מודל: התכווצות אלסטית (w_elastic)", color="#2e75b6")
ax_a.plot(w0_fr_list, P0_list, "-.", label="מודל: תזוזת חיכוך/קרקע (w_fric)", color="#e67e22")
clean_field = field_test_df.dropna()
if len(clean_field) > 0:
ax_a.plot(
clean_field["שקיעה מדודה בראש w₀ (mm)"],
clean_field["עומס מדוד בראש P₀ (kN)"],
"o-", color="#8e44ad", linewidth=1.8, markersize=6, label="ניסיון העמסה מדוד (שטח)"
)
ax_a.axhline(F_fric_at_zero_base, color="green", linestyle=":", label=f"גבול חיכוך ללא קצה ({F_fric_at_zero_base:.0f} kN)")
ax_a.scatter([w_total_node * 1000.0], [P_node], color="red", s=80, zorder=6, label="נקודת בדיקה נוכחית")
ax_a.set_ylim(0, P_max_user * 1.05)
ax_a.set_xlabel("תזוזה בראש הכלונס w₀ (mm)")
ax_a.set_ylabel("עומס בראש P₀ (kN)")
ax_a.set_title(f"עקומת עומס-תזוזה מ-0 עד {P_max_user:,.0f} kN")
ax_a.grid(True, linestyle="--", alpha=0.5)
ax_a.legend(fontsize=8)
plt.tight_layout()
st.pyplot(fig_ls)
tab1, tab2 = st.tabs(["📉 פרופילי עומק (תזוזות, מאמצים וכוחות)", "📋 טבלת שכבות מפורטת וייצוא"])
with tab1:
fig_prof, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.3))
mid_depths = [0.5 * (depths[i] + depths[i + 1]) for i in range(len(layers_df))]
ax1.plot(w_total_node * 1000.0, depths, marker="o", label="טוטאלית", color="#1f497d")
ax1.plot(w_elastic_node * 1000.0, depths, "--", label="אלסטית", color="#2e75b6")
ax1.plot(w_fric_node * 1000.0, depths, "-.", label="חיכוך/בסיס", color="#e67e22")
ax1.set_title("פרופיל תזוזות (mm)")
ax1.set_xlabel("תזוזה (mm)")
ax1.set_ylabel("עומק (m)")
ax1.invert_yaxis()
ax1.legend(fontsize=8)
ax1.grid(True, linestyle="--", alpha=0.5)
ax2.plot(t_peak_seg, mid_depths, "--", label="שיא", color="gray")
ax2.plot(t_res_seg, mid_depths, ":", label="משתייר", color="red")
ax2.plot(t_mob_seg, mid_depths, marker="s", label="בפועל", color="#27ae60")
ax2.set_title("מאמצי חיכוך (kPa)")
ax2.set_xlabel("מאמץ גזירה (kPa)")
ax2.invert_yaxis()
ax2.legend(fontsize=8)
ax2.grid(True, linestyle="--", alpha=0.5)
ax3.plot(P_node, depths, marker="^", color="#8e44ad")
ax3.set_title("כוח צירי לאורך הכלונס (kN)")
ax3.set_xlabel("כוח צירי (kN)")
ax3.invert_yaxis()
ax3.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
st.pyplot(fig_prof)
with tab2:
results_df = pd.DataFrame({
"שכבה": range(1, len(layers_df) + 1),
"Soil Type": layers_df["Soil Type"],
"עומק (m)": [f"{depths[i]:.1f} - {depths[i+1]:.1f}" for i in range(len(layers_df))],
"תזוזה טוטאלית (mm)": w_total_node[:-1] * 1000.0,
"התכווצות אלסטית (mm)": dw_elastic_seg * 1000.0,
"תזוזת חיכוך/ממשק (mm)": w_fric_node[:-1] * 1000.0,
"מאמץ שיא (kPa)": t_peak_seg,
"מאמץ משתייר (kPa)": t_res_seg,
"מאמץ בפועל (kPa)": t_mob_seg,
"כוח חיכוך שכבה (kN)": F_fric_seg,
"כוח צירי בראש שכבה (kN)": P_node[:-1],
"מצב השכבה": state_seg
})
st.dataframe(results_df.style.format({
"תזוזה טוטאלית (mm)": "{:.2f}",
"התכווצות אלסטית (mm)": "{:.2f}",
"תזוזת חיכוך/ממשק (mm)": "{:.2f}",
"מאמץ שיא (kPa)": "{:.1f}",
"מאמץ משתייר (kPa)": "{:.1f}",
"מאמץ בפועל (kPa)": "{:.1f}",
"כוח חיכוך שכבה (kN)": "{:.1f}",
"כוח צירי בראש שכבה (kN)": "{:,.1f}"
}), use_container_width=True, hide_index=True)
csv_out = results_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
label="📥 הורד דוח תוצאות מלא לאקסל (CSV)",
data=csv_out,
file_name=f"pile_analysis_V9_D{D}.csv",
mime="text/csv"
)
