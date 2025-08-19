
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="SaaS Investor Dashboard", page_icon="📈", layout="wide")

# -------------------- Defaults --------------------
DEFAULT_MULTIPLES = {"Academy/Starter": 8.0, "Basic": 8.0, "Advance": 9.0, "Pro": 10.0}

def fmt_eur(x: float) -> str:
    try:
        return f"{x:,.0f} €".replace(",", ".")
    except Exception:
        return str(x)

# -------------------- IO --------------------
@st.cache_data
def read_book(file):
    """Read Excel/CSV robustly. Tries openpyxl, then pandas auto-detect, with clear error messages."""
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
        return {"Data": df, "Prices": None}
    try:
        return pd.read_excel(file, sheet_name=None, engine="openpyxl")
    except Exception:
        try:
            return pd.read_excel(file, sheet_name=None)
        except Exception as e2:
            st.error("No se pudo leer el Excel. Asegúrate de que **openpyxl** está instalado (requirements.txt).")
            raise e2

def ensure_cols(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en 'Data': {missing}")
        st.stop()

# -------------------- Core calc --------------------
def enrich_data(df_data: pd.DataFrame, df_prices: pd.DataFrame | None):
    df = df_data.copy()
    df = df.rename(columns={
        "Real MRR (optional €)": "Real MRR (optional €)",
        "Active Customers (optional)": "Active Customers (optional)",
        "New Customers": "New Customers",
        "Lost Customers": "Lost Customers",
        "Plan": "Plan",
        "Date": "Date",
    })
    ensure_cols(df, ["Date","Plan","New Customers","Lost Customers"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Plan","Date"]).reset_index(drop=True)

    # ---- SAFE prices/multiples (no DataFrame truthiness) ----
    if df_prices is not None and not df_prices.empty:
        prices = df_prices.set_index("Plan")
    else:
        prices = pd.DataFrame(
            columns=["Price MRR (€)", "Price ARR (€)", "Multiple (x ARR)"]
        )
        prices.index.name = "Plan"

    price_map = prices["Price MRR (€)"].to_dict() if "Price MRR (€)" in prices else {}
    multiple_map = prices["Multiple (x ARR)"].to_dict() if "Multiple (x ARR)" in prices else {}
    for p in df["Plan"].unique():
        price_map.setdefault(p, 0.0)
        multiple_map.setdefault(p, DEFAULT_MULTIPLES.get(p, 8.0))

    df["Price MRR (€)"] = df["Plan"].map(price_map)
    df["Multiple (x ARR)"] = df["Plan"].map(multiple_map)

    # ---- Active Used rollforward (index preserving; no groupby.apply) ----
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
            out.append(val); prev = val
        df.loc[g.index, "Active Used"] = out
    df["Active Used"] = df["Active Used"].astype(float)

    # ---- MRRs ----
    df["MRR Calculated (€)"] = df["Active Used"] * df["Price MRR (€)"]
    if "Real MRR (optional €)" in df.columns:
        df["Real MRR used (€)"] = df["Real MRR (optional €)"].where(
            pd.notna(df["Real MRR (optional €)"]), df["MRR Calculated (€)"]
        )
    else:
        df["Real MRR used (€)"] = df["MRR Calculated (€)"]

    df["Prev Real MRR (€)"] = df.groupby("Plan")["Real MRR used (€)"].shift(1).fillna(0.0)
    df["Prev Active"] = df.groupby("Plan")["Active Used"].shift(1).fillna(0.0)
    df["Prev ARPU (€)"] = np.where(
        df["Prev Active"]>0, df["Prev Real MRR (€)"]/df["Prev Active"], df["Price MRR (€)"]
    )

    df["New MRR (€)"] = df["New Customers"] * df["Price MRR (€)"]
    df["Churned MRR (€)"] = df["Lost Customers"] * df["Prev ARPU (€)"]

    df["ΔMRR Real (€)"] = df["Real MRR used (€)"] - df["Prev Real MRR (€)"]
    df["Residual (€)"] = df["ΔMRR Real (€)"] - (df["New MRR (€)"] - df["Churned MRR (€)"])
    df["Expansion MRR (inferred €)"] = df["Residual (€)"].clip(lower=0.0)
    df["Downgraded MRR (inferred €)"] = (-df["Residual (€)"]).clip(lower=0.0)

    return df

def monthly_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    g = (enriched.groupby("Date", as_index=False).agg({
        "New Customers":"sum",
        "Lost Customers":"sum",
        "Active Used":"sum",
        "New MRR (€)":"sum",
        "Expansion MRR (inferred €)":"sum",
        "Churned MRR (€)":"sum",
        "Downgraded MRR (inferred €)":"sum",
        "Real MRR used (€)":"sum"
    }).sort_values("Date"))
    g = g.rename(columns={
        "Active Used":"Active Customers",
        "Real MRR used (€)":"Total MRR (€)"
    })
    g["Net New MRR (€)"] = g["New MRR (€)"] + g["Expansion MRR (inferred €)"] - g["Churned MRR (€)"] - g["Downgraded MRR (inferred €)"]
    g["Total ARR (€)"] = g["Total MRR (€)"] * 12.0
    g["Start MRR (€)"] = g["Total MRR (€)"].shift(1).fillna(0.0)

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
    first, last = s.iloc[0], s.iloc[-1]
    growth_ytd = (last["Total MRR (€)"] / first["Total MRR (€)"] - 1.0) if first["Total MRR (€)"]>0 else np.nan
    quick_ytd = (s["New MRR (€)"].sum() + s["Expansion MRR (inferred €)"].sum()) / max((s["Churned MRR (€)"].sum() + s["Downgraded MRR (inferred €)"].sum()), 1e-9)
    ratios_grr = s["GRR %"].dropna().values
    ratios_nrr = s["NRR %"].dropna().values
    grr_ytd = np.prod(ratios_grr) if len(ratios_grr)>0 else np.nan
    nrr_ytd = np.prod(ratios_nrr) if len(ratios_nrr)>0 else np.nan
    return {"Growth YTD": growth_ytd, "Quick YTD": quick_ytd, "GRR YTD": grr_ytd, "NRR YTD": nrr_ytd}

# -------------------- UI --------------------
st.title("📈 SaaS Investor Dashboard")
st.caption("Carga tu Excel con hojas **Data** y **Prices** (plantilla 'SaaS_Final_Template'). El dashboard calcula KPIs, NRR/GRR y más.")

with st.sidebar:
    uploaded = st.file_uploader("Sube tu Excel (XLSX/XLS/CSV)", type=["xlsx","xls","csv"])

if not uploaded:
    st.info("Sube un archivo para empezar. Recomendado: `SaaS_Final_Template_PRECOMPUTED.xlsx`.")
    st.stop()

book = read_book(uploaded)
df_data = book.get("Data"); df_prices = book.get("Prices")
ensure_cols(df_data, ["Date","Plan","New Customers","Lost Customers"])

# Enrich & summarize
enr = enrich_data(df_data, df_prices)
summ = monthly_summary(enr)

# -------- Sidebar filters (years, months, plan) --------
month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
month_num_by_name = {name:i+1 for i,name in enumerate(month_names)}

with st.sidebar:
    apply_to_kpis = st.toggle("Aplicar filtros a KPIs", value=True)
    all_years = sorted(pd.to_datetime(enr["Date"]).dt.year.unique().tolist())
    sel_years = st.multiselect("Años", options=all_years, default=all_years)

    sel_month_names = st.multiselect("Meses", options=month_names, default=month_names)
    sel_months = [month_num_by_name[m] for m in sel_month_names]

    plan_filter = st.selectbox("Plan", options=["(Todos)"] + sorted(enr["Plan"].unique().tolist()))

    components = st.multiselect("Componentes Net New",
        options=["New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"],
        default=["New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]
    )

    graph_metric = st.radio("Métrica para gráficos", ["MRR", "ARR"], index=0, horizontal=True)

def apply_filters(df):
    mask_year = df["Date"].dt.year.isin(sel_years) if sel_years else True
    mask_month = df["Date"].dt.month.isin(sel_months) if sel_months else True
    mask_plan = (df["Plan"] == plan_filter) if plan_filter != "(Todos)" else True
    return df[mask_year & mask_month & mask_plan]

enr_f = apply_filters(enr)
summ_f = apply_filters(summ)

# -------------------- KPIs Header (optionally filtered) --------------------
kpi_src = summ_f if apply_to_kpis else summ
if not kpi_src.empty:
    last_row = kpi_src.iloc[-1]
    ytd = ytd_metrics(kpi_src)
    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("Clientes activos", f"{int(last_row['Active Customers']):,}".replace(",", "."))
    k2.metric("MRR total", fmt_eur(last_row["Total MRR (€)"]))
    k3.metric("ARR total", fmt_eur(last_row["Total ARR (€)"]))
    k4.metric("Net New MRR (últ. mes)", fmt_eur(last_row["Net New MRR (€)"]))
    k5.metric("Growth YTD", f"{(ytd['Growth YTD']*100):.1f}%")
    k6.metric("GRR YTD", f"{(ytd['GRR YTD']*100):.1f}%" if pd.notna(ytd["GRR YTD"]) else "—")
    k7.metric("NRR YTD", f"{(ytd['NRR YTD']*100):.1f}%" if pd.notna(ytd["NRR YTD"]) else "—")

with st.expander("Ver tablas (Data enriquecida y Monthly Summary)"):
    st.dataframe(enr_f.head(200), use_container_width=True)
    st.dataframe(summ_f.tail(24), use_container_width=True)

# -------------------- Charts --------------------
st.markdown("### Evolución de MRR/ARR y desglose del Net New")
metric_col = "Total MRR (€)" if graph_metric == "MRR" else "Total ARR (€)"
if not summ_f.empty:
    st.line_chart(summ_f.set_index("Date")[[metric_col]], use_container_width=True)

if not summ_f.empty:
    area_df = summ_f.set_index("Date")[components]
    st.area_chart(area_df, use_container_width=True)

# Per plan snapshot (último mes filtrado)
st.markdown("### Desglose por plan (último mes del periodo filtrado)")
if not enr_f.empty:
    latest_date = enr_f["Date"].max()
    snap = enr_f[enr_f["Date"] == latest_date].groupby("Plan", as_index=False).agg({
        "Active Used":"sum",
        "Real MRR used (€)":"sum",
        "New MRR (€)":"sum",
        "Expansion MRR (inferred €)":"sum",
        "Churned MRR (€)":"sum",
        "Downgraded MRR (inferred €)":"sum"
    }).rename(columns={"Active Used":"Active","Real MRR used (€)":"MRR"})
    snap["ARR"] = snap["MRR"] * 12.0
    snap["Mix %"] = snap["MRR"] / snap["MRR"].sum() * 100.0

    show = snap.assign(
        MRR_fmt=snap["MRR"].map(fmt_eur),
        ARR_fmt=snap["ARR"].map(fmt_eur),
        Mix_fmt=snap["Mix %"].map(lambda x: f"{x:.1f}%")
    )[["Plan","Active","MRR_fmt","ARR_fmt","Mix_fmt","New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]]
    st.dataframe(show, hide_index=True, use_container_width=True)
else:
    st.info("No hay datos tras aplicar filtros.")

# Year-end close chart (último mes por año)
st.markdown("### Cierre de año por año (MRR/ARR del último mes disponible)")
if not summ_f.empty:
    cl = (summ_f.assign(Year=summ_f["Date"].dt.year)
                 .sort_values("Date")
                 .groupby("Year")
                 .tail(1))
    cl_metric = "Total MRR (€)" if graph_metric == "MRR" else "Total ARR (€)"
    cl_plot = cl[["Year", cl_metric]].set_index("Year")
    st.bar_chart(cl_plot, use_container_width=True)

# -------------------- Export snapshot --------------------
st.markdown("### Exportar snapshot (Resumen filtrado)")
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    summ_f.to_excel(writer, sheet_name="Monthly_Summary_Filtered", index=False)
    enr_f.to_excel(writer, sheet_name="Data_Enriched_Filtered", index=False)
st.download_button("⬇️ Descargar Excel (snapshot filtrado)",
                   data=buf.getvalue(),
                   file_name="investor_snapshot_filtered.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
