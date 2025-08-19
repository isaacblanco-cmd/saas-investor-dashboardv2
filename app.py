
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="SaaS Investor Dashboard", page_icon="📈", layout="wide")

# -------------------- Helpers --------------------
DEFAULT_MULTIPLES = {
    "Academy/Starter": 8.0,
    "Basic": 8.0,
    "Advance": 9.0,
    "Pro": 10.0,
}

def fmt_eur(x: float) -> str:
    try:
        return f"{x:,.0f} €".replace(",", ".")
    except Exception:
        return str(x)

@st.cache_data
def read_book(file):
    """Read Excel/CSV robustly. Tries openpyxl, then pandas auto-detect, with clear error messages."""
    name = file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(file)
        return {"Data": df, "Prices": None}
    try:
        # Prefer openpyxl
        return pd.read_excel(file, sheet_name=None, engine="openpyxl")
    except Exception as e1:
        try:
            # Fallback: let pandas choose an engine if available
            return pd.read_excel(file, sheet_name=None)
        except Exception as e2:
            st.error("No se pudo leer el Excel. Asegúrate de que **openpyxl** está instalado en el entorno (requirements.txt).")
            raise e2

def ensure_cols(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en 'Data': {missing}")
        st.stop()

def enrich_data(df_data: pd.DataFrame, df_prices: pd.DataFrame | None):
    df = df_data.copy()
    # Normalize columns
    rename_map = {
        "Real MRR (optional €)": "Real MRR (optional €)",
        "Active Customers (optional)": "Active Customers (optional)",
        "New Customers": "New Customers",
        "Lost Customers": "Lost Customers",
        "Plan": "Plan",
        "Date": "Date",
    }
    df = df.rename(columns=rename_map)
    ensure_cols(df, ["Date","Plan","New Customers","Lost Customers"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Plan","Date"]).reset_index(drop=True)

    # Prices / Multiples
    if df_prices is not None:
        prices = df_prices.set_index("Plan")
        price_map = prices["Price MRR (€)"].to_dict() if "Price MRR (€)" in prices else {}
        multiple_map = prices["Multiple (x ARR)"].to_dict() if "Multiple (x ARR)" in prices else {}
    else:
        price_map, multiple_map = {}, {}

    for p in df["Plan"].unique():
        price_map.setdefault(p, 0.0)
        multiple_map.setdefault(p, DEFAULT_MULTIPLES.get(p, 8.0))

    df["Price MRR (€)"] = df["Plan"].map(price_map)
    df["Multiple (x ARR)"] = df["Plan"].map(multiple_map)

    # -------- FIXED Active Used rollforward with index-preserving assignment --------
    df["Active Used"] = np.nan
    for plan, g in df.sort_values(["Plan","Date"]).groupby("Plan"):
        prev = 0.0
        out = []
        for idx, row in g.iterrows():
            override = row.get("Active Customers (optional)")
            if pd.notna(override):
                val = float(override)
            else:
                val = max(prev + float(row["New Customers"]) - float(row["Lost Customers"]), 0.0)
            out.append(val)
            prev = val
        df.loc[g.index, "Active Used"] = out
    df["Active Used"] = df["Active Used"].astype(float)

    # MRR Calculated & Real Used
    df["MRR Calculated (€)"] = df["Active Used"] * df["Price MRR (€)"]
    if "Real MRR (optional €)" in df.columns:
        df["Real MRR used (€)"] = df["Real MRR (optional €)"].where(
            pd.notna(df["Real MRR (optional €)"]),
            df["MRR Calculated (€)"]
        )
    else:
        df["Real MRR used (€)"] = df["MRR Calculated (€)"]

    # Previous month context
    df["Prev Real MRR (€)"] = df.groupby("Plan")["Real MRR used (€)"].shift(1).fillna(0.0)
    df["Prev Active"] = df.groupby("Plan")["Active Used"].shift(1).fillna(0.0)
    df["Prev ARPU (€)"] = np.where(
        df["Prev Active"]>0,
        df["Prev Real MRR (€)"]/df["Prev Active"],
        df["Price MRR (€)"]
    )

    # New/Churned MRR using catalog price and prev ARPU
    df["New MRR (€)"] = df["New Customers"] * df["Price MRR (€)"]
    df["Churned MRR (€)"] = df["Lost Customers"] * df["Prev ARPU (€)"]

    # Residual method for inferred expansion/downgrade
    df["ΔMRR Real (€)"] = df["Real MRR used (€)"] - df["Prev Real MRR (€)"]
    df["Residual (€)"] = df["ΔMRR Real (€)"] - (df["New MRR (€)"] - df["Churned MRR (€)"])
    df["Expansion MRR (inferred €)"] = df["Residual (€)"].clip(lower=0.0)
    df["Downgraded MRR (inferred €)"] = (-df["Residual (€)"]).clip(lower=0.0)

    return df

def build_monthly_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    g = enriched.groupby("Date", as_index=False).agg({
        "New Customers":"sum",
        "Lost Customers":"sum",
        "Active Used":"sum",
        "New MRR (€)":"sum",
        "Expansion MRR (inferred €)":"sum",
        "Churned MRR (€)":"sum",
        "Downgraded MRR (inferred €)":"sum",
        "Real MRR used (€)":"sum"
    }).sort_values("Date")
    g = g.rename(columns={
        "Active Used":"Active Customers",
        "Real MRR used (€)":"Total MRR (€)"
    })
    g["Net New MRR (€)"] = g["New MRR (€)"] + g["Expansion MRR (inferred €)"] - g["Churned MRR (€)"] - g["Downgraded MRR (inferred €)"]
    g["Total ARR (€)"] = g["Total MRR (€)"] * 12.0
    g["Start MRR (€)"] = g["Total MRR (€)"].shift(1).fillna(0.0)

    # Ratios
    denom = g["Churned MRR (€)"] + g["Downgraded MRR (inferred €)"]
    numer = g["New MRR (€)"] + g["Expansion MRR (inferred €)"]
    g["Quick Ratio"] = np.where(denom>0, numer/denom, np.nan)
    g["GRR %"] = np.where(g["Start MRR (€)"]>0,
                          1 - (g["Churned MRR (€)"] + g["Downgraded MRR (inferred €)"]) / g["Start MRR (€)"],
                          np.nan)
    g["NRR %"] = np.where(g["Start MRR (€)"]>0,
                          1 + (g["Expansion MRR (inferred €)"] - (g["Churned MRR (€)"] + g["Downgraded MRR (inferred €)"])) / g["Start MRR (€)"],
                          np.nan)
    g["MoM Growth %"] = g["Total MRR (€)"].pct_change()
    g["Churn % (customers)"] = g["Lost Customers"] / g["Active Customers"].shift(1).replace(0, np.nan)
    g["ARPU (€)"] = g["Total MRR (€)"] / g["Active Customers"].replace(0, np.nan)
    return g

def ytd_metrics(summary: pd.DataFrame) -> dict:
    if summary.empty:
        return {"Growth YTD":0,"Quick YTD":np.nan,"GRR YTD":np.nan,"NRR YTD":np.nan}
    s = summary.copy().sort_values("Date")
    first = s.iloc[0]
    last = s.iloc[-1]
    growth_ytd = (last["Total MRR (€)"] / first["Total MRR (€)"] - 1.0) if first["Total MRR (€)"]>0 else np.nan
    quick_ytd = (s["New MRR (€)"].sum() + s["Expansion MRR (inferred €)"].sum()) / \
                max((s["Churned MRR (€)"].sum() + s["Downgraded MRR (inferred €)"].sum()), 1e-9)
    ratios_grr = s["GRR %"].dropna().values
    ratios_nrr = s["NRR %"].dropna().values
    grr_ytd = np.prod(ratios_grr) if len(ratios_grr)>0 else np.nan
    nrr_ytd = np.prod(ratios_nrr) if len(ratios_nrr)>0 else np.nan
    return {"Growth YTD": growth_ytd, "Quick YTD": quick_ytd, "GRR YTD": grr_ytd, "NRR YTD": nrr_ytd}

def cohort_fifo_matrix(new_series: pd.Series, lost_series: pd.Series, max_age: int = 12):
    new_series = new_series.astype(float).fillna(0.0)
    lost_series = lost_series.astype(float).fillna(0.0)
    dates = pd.to_datetime(new_series.index).sort_values()
    cohorts = []
    loss_map = {}
    for d in dates:
        new_n = new_series.loc[d]
        if new_n > 0:
            cohorts.append([d, new_n])
        lost_n = lost_series.loc[d]
        while lost_n > 1e-9 and len(cohorts) > 0:
            c_date, c_rem = cohorts[0]
            take = min(c_rem, lost_n)
            age = (d.to_period("M") - pd.Timestamp(c_date).to_period("M")).n
            loss_map.setdefault(c_date, {}).setdefault(age, 0.0)
            loss_map[c_date][age] += take
            c_rem -= take
            lost_n -= take
            if c_rem <= 1e-9:
                cohorts.pop(0)
            else:
                cohorts[0][1] = c_rem

    cohort_months = sorted(set(list(new_series[new_series>0].index)))
    if not cohort_months:
        return pd.DataFrame()

    overall_last = dates.max()
    max_age_obs = max((overall_last.to_period("M") - pd.Timestamp(cm).to_period("M")).n for cm in cohort_months)
    horizon = min(max_age_obs, max_age)

    rows = []
    for cm in cohort_months:
        init = new_series.loc[cm]
        survivors_by_age = []
        cumulative_lost = 0.0
        for age in range(horizon+1):
            lost_at_age = loss_map.get(cm, {}).get(age, 0.0)
            cumulative_lost += lost_at_age
            survivors = max(init - cumulative_lost, 0.0)
            survivors_by_age.append(survivors)
        rows.append({
            "Cohort Month": cm,
            **{f"m{age}": survivors_by_age[age] for age in range(horizon+1)},
            "Initial": init
        })
    mat = pd.DataFrame(rows).sort_values("Cohort Month")
    mat["Cohort Year"] = pd.to_datetime(mat["Cohort Month"]).dt.year
    agg = mat.groupby("Cohort Year").sum(numeric_only=True)
    for col in [c for c in agg.columns if c.startswith("m")]:
        agg[col] = np.where(agg["Initial"]>0, agg[col] / agg["Initial"], np.nan)
    ordered = ["Initial"] + [c for c in agg.columns if c.startswith("m")]
    return agg[ordered]

def valuation_by_plan(df_enriched: pd.DataFrame, multiples_override: dict | None = None):
    last_date = df_enriched["Date"].max()
    snap = df_enriched[df_enriched["Date"] == last_date].groupby("Plan", as_index=False).agg({
        "Real MRR used (€)":"sum",
        "Multiple (x ARR)":"first"
    })
    snap["ARR (€)"] = snap["Real MRR used (€)"] * 12.0
    if multiples_override:
        snap["Multiple (x ARR)"] = snap["Plan"].map(lambda p: multiples_override.get(p, snap.loc[snap["Plan"]==p, "Multiple (x ARR)"].values[0]))
    snap["Valuation (€)"] = snap["ARR (€)"] * snap["Multiple (x ARR)"]
    return snap.sort_values("Valuation (€)", ascending=False)

# -------------------- UI --------------------
st.title("📈 SaaS Investor Dashboard")
st.caption("Carga tu Excel con hojas **Data** y **Prices** (plantilla 'SaaS_Final_Template'). El dashboard calcula KPIs, NRR/GRR, cohortes, y valoración por plan.")

with st.sidebar:
    uploaded = st.file_uploader("Sube tu Excel", type=["xlsx","xls","csv"])
    st.markdown("---")
    st.markdown("### Filtros")

if not uploaded:
    st.info("Sube un archivo para empezar. Puedes usar el `SaaS_Final_Template_PRECOMPUTED.xlsx`.")
    st.stop()

book = read_book(uploaded)
df_data = book.get("Data")
df_prices = book.get("Prices")

ensure_cols(df_data, ["Date","Plan","New Customers","Lost Customers"])

enriched = enrich_data(df_data, df_prices)
summary = build_monthly_summary(enriched)

# Sidebar filters (years & YTD)
years = sorted(pd.to_datetime(enriched["Date"]).dt.year.unique().tolist())
default_years = years
with st.sidebar:
    ytd_toggle = st.toggle("Solo Año actual (YTD)", value=False)
    selected_years = st.multiselect("Años", options=years, default=(years[-1] if ytd_toggle and years else default_years))
    plan_filter = st.selectbox("Plan", options=["(Todos)"] + sorted(enriched["Plan"].unique().tolist()))

def is_selected_year(date):
    return pd.to_datetime(date).year in selected_years

enriched_f = enriched[enriched["Date"].apply(is_selected_year)].copy()
summary_f = summary[summary["Date"].apply(is_selected_year)].copy()

if ytd_toggle and len(selected_years)==1:
    target_year = selected_years[0]
    today = pd.Timestamp.today()
    end_month = pd.Timestamp(year=target_year, month=(today.month if today.year==target_year else 12), day=1)
    enriched_f = enriched_f[enriched_f["Date"] <= end_month]
    summary_f = summary_f[summary_f["Date"] <= end_month]

if plan_filter != "(Todos)":
    enriched_plan = enriched_f[enriched_f["Plan"] == plan_filter].copy()
else:
    enriched_plan = enriched_f.copy()

# -------------------- KPI Header --------------------
if not summary_f.empty:
    last_row = summary_f.iloc[-1]
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Clientes activos", f"{int(last_row['Active Customers']):,}".replace(",", "."))
    k2.metric("MRR total", fmt_eur(last_row["Total MRR (€)"]))
    k3.metric("ARR total", fmt_eur(last_row["Total ARR (€)"]))
    k4.metric("Net New MRR (últ. mes)", fmt_eur(last_row["Net New MRR (€)"]))
    ytd = ytd_metrics(summary_f)
    k5.metric("Growth YTD", f"{(ytd['Growth YTD']*100):.1f}%")
    k6.metric("GRR YTD", f"{(ytd['GRR YTD']*100):.1f}%" if pd.notna(ytd["GRR YTD"]) else "—")
    k7.metric("NRR YTD", f"{(ytd['NRR YTD']*100):.1f}%" if pd.notna(ytd["NRR YTD"]) else "—")

with st.expander("Ver tablas (Data enriquecida y Monthly Summary)"):
    st.dataframe(enriched_f.head(100), use_container_width=True)
    st.dataframe(summary_f.tail(24), use_container_width=True)

# -------------------- Charts --------------------
st.markdown("### Evolución de MRR y desglose del Net New")
c1, c2 = st.columns(2)
with c1:
    st.line_chart(summary_f.set_index("Date")[["Total MRR (€)"]], use_container_width=True)
with c2:
    area_df = summary_f.set_index("Date")[["New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]]
    st.area_chart(area_df, use_container_width=True)

# -------------------- Per-plan snapshot --------------------
st.markdown("### Desglose por plan (último mes del periodo filtrado)")
latest_date = enriched_f["Date"].max() if not enriched_f.empty else None
if latest_date is not None:
    snap = enriched_f[enriched_f["Date"] == latest_date].groupby("Plan", as_index=False).agg({
        "Active Used":"sum",
        "Real MRR used (€)":"sum",
        "New MRR (€)":"sum",
        "Expansion MRR (inferred €)":"sum",
        "Churned MRR (€)":"sum",
        "Downgraded MRR (inferred €)":"sum"
    }).rename(columns={"Active Used":"Active","Real MRR used (€)":"MRR"})
    snap["ARR"] = snap["MRR"] * 12.0
    snap["Mix %"] = snap["MRR"] / snap["MRR"].sum() * 100.0
    st.dataframe(
        snap.assign(
            MRR_fmt=snap["MRR"].map(fmt_eur),
            ARR_fmt=snap["ARR"].map(fmt_eur),
            Mix_fmt=snap["Mix %"].map(lambda x: f"{x:.1f}%")
        )[["Plan","Active","MRR_fmt","ARR_fmt","Mix_fmt","New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]],
        hide_index=True, use_container_width=True
    )

# -------------------- Cohorts (FIFO approximation) --------------------
st.markdown("---")
st.markdown("## Cohorts por año de alta (aprox. FIFO)")
cohort_plan = st.selectbox("Plan para cohorts", options=["(Todos)"] + sorted(enriched["Plan"].unique().tolist()), index=0)
max_age = st.slider("Horizonte (meses)", 6, 24, 12, step=1)

if cohort_plan == "(Todos)":
    new_series = enriched.groupby("Date")["New Customers"].sum()
    lost_series = enriched.groupby("Date")["Lost Customers"].sum()
else:
    dfp = enriched[enriched["Plan"] == cohort_plan]
    new_series = dfp.groupby("Date")["New Customers"].sum()
    lost_series = dfp.groupby("Date")["Lost Customers"].sum()

coh = cohort_fifo_matrix(new_series, lost_series, max_age=max_age)
if not coh.empty:
    st.caption("Cada fila es el **año de alta**, columnas m0..mN son la **supervivencia (%)** del cohort a esa edad.")
    show = coh.copy()
    for c in [c for c in coh.columns if c.startswith("m")]:
        show[c] = (show[c] * 100).round(1)
    st.dataframe(show, use_container_width=True)
else:
    st.info("No hay suficientes datos de 'New'/'Lost' para construir cohorts.")

# -------------------- Valuation --------------------
st.markdown("---")
st.markdown("## Valoración por plan y total")
plans = sorted(enriched["Plan"].unique().tolist())
mult_df = pd.DataFrame({
    "Plan": plans,
    "Multiple (x ARR)": [enriched.loc[enriched["Plan"]==p, "Multiple (x ARR)"].iloc[-1] if (enriched["Plan"]==p).any() else DEFAULT_MULTIPLES.get(p,8.0) for p in plans]
})
edited_mult = st.data_editor(mult_df, use_container_width=True, key="mults_editor")
override_map = dict(zip(edited_mult["Plan"], edited_mult["Multiple (x ARR)"]))

val = valuation_by_plan(enriched, multiples_override=override_map)
val_total = val["Valuation (€)"].sum()

colL, colR = st.columns([2,1])
with colL:
    st.dataframe(val.assign(
        ARR_fmt=val["ARR (€)"].map(fmt_eur),
        Val_fmt=val["Valuation (€)"].map(fmt_eur)
    )[["Plan","ARR (€)","ARR_fmt","Multiple (x ARR)","Valuation (€)","Val_fmt"]], hide_index=True, use_container_width=True)
with colR:
    st.metric("Valoración total", fmt_eur(val_total))

# Export snapshot
st.markdown("### Exportar snapshot (Resumen filtrado)")
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    summary_f.to_excel(writer, sheet_name="Monthly_Summary_Filtered", index=False)
    val.to_excel(writer, sheet_name="Valuation_By_Plan", index=False)
    enriched_f.to_excel(writer, sheet_name="Data_Enriched", index=False)
st.download_button("⬇️ Descargar Excel (snapshot)", data=buf.getvalue(), file_name="investor_snapshot.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Notas: GRR/NRR mensuales usan el MRR de inicio (mes anterior). YTD compone multiplicando ratios mensuales. Cohorts se aproximan con FIFO sobre New/Lost agregados.")
