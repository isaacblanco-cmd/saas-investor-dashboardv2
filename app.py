import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="SaaS Investor Dashboard",
    page_icon="💼",
    layout="wide"
)

# ------------------------- Utils -------------------------

@st.cache_data(show_spinner=False)
def read_book(file) -> dict:
    # Lee todas las hojas. Si subes CSV, cae a leerlo como Data.
    try:
        book = pd.read_excel(file, sheet_name=None)
        return book
    except Exception:
        try:
            df = pd.read_csv(file)
            return {"Data": df}
        except Exception as e:
            raise e


def ensure_cols(df: pd.DataFrame, cols: list[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas obligatorias en hoja Data: {miss}")


# ------------------------- Enriquecido -------------------------

def enrich_data(df_data: pd.DataFrame, df_prices: pd.DataFrame | None) -> pd.DataFrame:
    df = df_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Plan", "Date"]).reset_index(drop=True)

    # hoja Prices opcional, evitar truth-value ambiguity con pandas
    if df_prices is None or (isinstance(df_prices, pd.DataFrame) and df_prices.empty):
        prices = pd.DataFrame(index=pd.Index([], name="Plan"))
    else:
        prices = df_prices.copy()
        if "Plan" not in prices.columns:
            prices = prices.rename(columns={prices.columns[0]: "Plan"})
        prices = prices.set_index("Plan")

    price_map = prices["Price MRR (€)"].to_dict() if "Price MRR (€)" in prices.columns else {}
    multiple_map = prices["Multiple (x ARR)"].to_dict() if "Multiple (x ARR)" in prices.columns else {}

    # MRR calculado por precio de tarifa
    df["MRR Calculated (€)"] = df["Active Customers"].fillna(0).astype(float) * df["Plan"].map(price_map).fillna(0)

    # MRR real opcional y el utilizado
    if "Real MRR used (€)" not in df.columns and "Real MRR (optional €)" in df.columns:
        df["Real MRR used (€)"] = df["Real MRR (optional €)"]

    df["MRR Used (€)"] = df["Real MRR used (€)"].fillna(df["MRR Calculated (€)"])

    # ARR y múltiplos
    df["ARR (€)"] = df["MRR Used (€)"] * 12
    df["Multiple (x ARR)"] = df["Plan"].map(multiple_map).fillna(0.0)

    # Net-new inferido si no viene
    if "New MRR (calc €)" not in df.columns:
        df["New MRR (calc €)"] = (
            df["New Customers"].fillna(0).astype(float) * df["Plan"].map(price_map).fillna(0)
        )

    if "Churned MRR (€)" not in df.columns:
        df["Churned MRR (€)"] = (
            df["Lost Customers"].fillna(0).astype(float) * df["Plan"].map(price_map).fillna(0)
        ) * (-1)

    df["Downgraded MRR (inferred €)"] = df.get("Downgraded MRR (calc €)", 0)
    df["Expansion MRR (inferred €)"] = df.get("Expansion MRR (calc €)", 0)

    # Activos “usados” por rollforward plan a plan
    def _roll(g: pd.DataFrame) -> pd.Series:
        active = []
        curr = 0.0
        for _, r in g.iterrows():
            curr += float(r.get("New Customers", 0)) - float(r.get("Lost Customers", 0))
            active.append(max(curr, 0))
        return pd.Series(active, index=g.index)

    df["Active Used"] = (
        df.groupby("Plan", group_keys=False)
          .apply(lambda g: _roll(g.reset_index(drop=True)))
          .reset_index(level=0, drop=True)
          .astype(int)
    )

    # helpers de fecha
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.month_name()

    return df


def monthly_summary(enr: pd.DataFrame) -> pd.DataFrame:
    grp = (enr.groupby(["Date"], as_index=False)
              .agg({
                  "MRR Used (€)": "sum",
                  "ARR (€)": "sum",
                  "New MRR (calc €)": "sum",
                  "Churned MRR (€)": "sum",
                  "Downgraded MRR (inferred €)": "sum",
                  "Expansion MRR (inferred €)": "sum",
                  "Active Used": "sum"
              }))
    grp["Year"] = grp["Date"].dt.year
    grp["Month"] = grp["Date"].dt.month
    grp["MonthName"] = grp["Date"].dt.month_name()
    return grp


def apply_filters(df: pd.DataFrame,
                  sel_years: list[int] | None,
                  sel_months: list[int] | None,
                  plan_filter: str | None) -> pd.DataFrame:
    # Años
    mask_year = True
    if sel_years:
        if "Year" not in df.columns:
            df = df.assign(Year=df["Date"].dt.year)
        mask_year = df["Year"].isin(sel_years)

    # Meses
    mask_month = True
    if sel_months:
        if "Month" not in df.columns:
            df = df.assign(Month=df["Date"].dt.month)
        mask_month = df["Month"].isin(sel_months)

    # Plan (solo si existe)
    mask_plan = True
    if plan_filter and plan_filter != "(Todos)" and "Plan" in df.columns:
        mask_plan = (df["Plan"] == plan_filter)

    return df[mask_year & mask_month & mask_plan]


# ------------------------- Cohortes (FIFO) -------------------------

def build_cohorts_fifo(enr: pd.DataFrame) -> pd.DataFrame:
    use_cols = ["Date", "Plan", "New Customers", "Lost Customers"]
    miss = [c for c in use_cols if c not in enr.columns]
    if miss:
        raise ValueError(f"Faltan columnas para cohorts: {miss}")

    df = enr.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Plan", "Date"])

    all_mats = []
    for plan, g in df.groupby("Plan"):
        g = g[["Date", "New Customers", "Lost Customers"]].reset_index(drop=True)
        months = pd.to_datetime(g["Date"].dt.to_period("M").astype(str)).tolist()

        size = len(g)
        mat = np.zeros((size, size), dtype=float)
        cohort_sizes = g["New Customers"].fillna(0).astype(float).values
        alive = cohort_sizes.copy()
        mat[np.arange(size), 0] = cohort_sizes

        for t in range(1, size):
            churn = float(g.loc[t, "Lost Customers"] or 0.0)
            for c in range(0, t + 1):
                if churn <= 0:
                    break
                take = min(alive[c], churn)
                alive[c] -= take
                churn -= take
            for c in range(0, t + 1):
                mat[c, t] = max(alive[c], 0.0)

        mat_df = pd.DataFrame(mat, index=months, columns=months)
        mat_df.index.name = "Cohort"
        mat_df.columns.name = "Month"
        mat_df = mat_df.stack().reset_index(name="Alive")
        mat_df["Plan"] = plan
        mat_df = mat_df[mat_df["Month"] >= mat_df["Cohort"]]
        all_mats.append(mat_df)

    out = pd.concat(all_mats, ignore_index=True)
    out = (out.groupby(["Cohort", "Month"], as_index=False)["Alive"].sum())

    cohort_size = (enr.groupby(["Date"], as_index=False)["New Customers"].sum()
                     .rename(columns={"Date": "Cohort", "New Customers": "CohortSize"}))
    out = out.merge(cohort_size, on="Cohort", how="left")
    out["Retention"] = np.where(out["CohortSize"] > 0, out["Alive"] / out["CohortSize"], np.nan)
    out["Age"] = ((out["Month"].dt.to_period("M") - out["Cohort"].dt.to_period("M")).apply(int))
    return out


def cohorts_pivot_heatmap(coh: pd.DataFrame) -> pd.DataFrame:
    tmp = coh.copy()
    tmp["CohortLabel"] = tmp["Cohort"].dt.strftime("%Y-%m")
    piv = tmp.pivot_table(index="CohortLabel", columns="Age", values="Retention", aggfunc="mean")
    piv = piv.sort_index()
    return piv


# ------------------------- UI -------------------------

st.title("SaaS Investor Dashboard")

with st.sidebar:
    st.subheader("Sube tu Excel")
    st.caption("Límite 200 MB. Recomendado: plantilla **SaaS_Final_Template.xlsx** con hojas **Data** y **Prices**.")
    up = st.file_uploader("Arrastra tu archivo aquí", type=["xlsx", "xls", "csv"])

    st.markdown("---")
    apply_to_kpis = st.toggle("Aplicar filtros a KPIs", value=False)
    st.caption("Si está activado, los KPIs de cabecera se calculan con los mismos filtros.")

if not up:
    st.info("Sube un Excel/CSV para empezar.")
    st.stop()

book = read_book(up)
df_data = book.get("Data")
df_prices = book.get("Prices")

ensure_cols(df_data, ["Date", "Plan", "New Customers", "Lost Customers", "Active Customers"])

# Enriquecer y resumir
enr = enrich_data(df_data, df_prices)
summ = monthly_summary(enr)

# ----------- Filtros (años, meses, plan) -----------
years = sorted(enr["Year"].unique().tolist())
months_map = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
plans = ["(Todos)"] + sorted(enr["Plan"].unique().tolist())

with st.sidebar:
    st.subheader("Filtros")
    sel_years = st.multiselect("Años", years, default=years)
    # selector meses por nombre -> convertir a número
    sel_month_names = st.multiselect(
        "Meses", list(months_map.values()),
        default=list(months_map.values())
    )
    rev_map = {v:k for k,v in months_map.items()}
    sel_months = [rev_map[m] for m in sel_month_names] if sel_month_names else []

    plan_filter = st.selectbox("Plan", plans, index=0)

# DataFrames filtrados
enr_f = apply_filters(enr, sel_years, sel_months, plan_filter)
summ_f = apply_filters(summ, sel_years, sel_months, plan_filter)

# Fuente de KPIs (filtrada u original)
kpi_src = summ_f if apply_to_kpis else summ

# ---------------- KPIs ----------------
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    active = int(enr_f.loc[enr_f["Date"] == enr_f["Date"].max(), "Active Used"].sum()) if not enr_f.empty else 0
    st.metric("Clientes activos", f"{active}")
with col2:
    mrr_total = float(kpi_src["MRR Used (€)"].iloc[-1]) if not kpi_src.empty else 0.0
    st.metric("MRR total", f"{mrr_total:,.0f} €".replace(",", "."))
with col3:
    arr_total = float(kpi_src["ARR (€)"].iloc[-1]) if not kpi_src.empty else 0.0
    st.metric("ARR total", f"{arr_total:,.0f} €".replace(",", "."))
with col4:
    last_net = float(kpi_src["New MRR (calc €)"].iloc[-1] + kpi_src["Expansion MRR (inferred €)"].iloc[-1] + kpi_src["Downgraded MRR (inferred €)"].iloc[-1] + kpi_src["Churned MRR (€)"].iloc[-1]) if not kpi_src.empty else 0.0
    st.metric("Net New MRR (últ. mes)", f"{last_net:,.0f} €".replace(",", "."))
with col5:
    # Growth YTD (MRR último mes filtrado vs primer mes del año filtrado)
    try:
        y = max(sel_years) if sel_years else enr["Year"].max()
        base = kpi_src[kpi_src["Year"] == y]
        if not base.empty:
            first = float(base.iloc[0]["MRR Used (€)"])
            last = float(base.iloc[-1]["MRR Used (€)"])
            growth = (last - first) / first * 100 if first > 0 else 0.0
        else:
            growth = 0.0
    except Exception:
        growth = 0.0
    st.metric("Growth YTD", f"{growth:.1f}%")
with col6:
    # GRR YTD (sin expansiones ni nuevas altas: 1 + churn_downg / base)
    try:
        y = max(sel_years) if sel_years else enr["Year"].max()
        base = kpi_src[kpi_src["Year"] == y]
        if not base.empty:
            first = float(base.iloc[0]["MRR Used (€)"])
            churn_down = float(base["Churned MRR (€)"].sum() + base["Downgraded MRR (inferred €)"].sum())
            grr = (first + churn_down) / first * 100 if first > 0 else 100.0
            grr = max(min(grr, 100.0), 0.0)
        else:
            grr = 100.0
    except Exception:
        grr = 100.0
    st.metric("GRR YTD", f"{grr:.1f}%")

# NRR YTD
try:
    y = max(sel_years) if sel_years else enr["Year"].max()
    base = kpi_src[kpi_src["Year"] == y]
    if not base.empty:
        first = float(base.iloc[0]["MRR Used (€)"])
        ups = float(base["Expansion MRR (inferred €)"].sum())
        churn_down = float(base["Churned MRR (€)"].sum() + base["Downgraded MRR (inferred €)"].sum())
        nrr = (first + ups + churn_down) / first * 100 if first > 0 else 100.0
        nrr = max(nrr, 0.0)
    else:
        nrr = 100.0
except Exception:
    nrr = 100.0
st.metric("NRR YTD", f"{nrr:.1f}%")

st.markdown("—")

# ---------------- Gráficos principales ----------------

with st.container():
    st.subheader("Evolución de MRR y desglose del Net New")

    # Serie MRR
    mrr_line = alt.Chart(summ_f).mark_line(point=False).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("MRR Used (€):Q", title="MRR (€)")
    ).properties(height=320)

    # Componentes NetNew seleccionables
    components = {
        "New MRR (€)": "New MRR (calc €)",
        "Churned MRR (€)": "Churned MRR (€)",
        "Downgraded MRR (inferred €)": "Downgraded MRR (inferred €)",
        "Expansion MRR (inferred €)": "Expansion MRR (inferred €)"
    }
    with st.expander("Componentes NetNew a mostrar", expanded=False):
        default_sel = list(components.keys())
        pick = st.multiselect("Series", list(components.keys()), default=default_sel)
    pick_cols = [components[k] for k in pick]

    net = summ_f[["Date"] + pick_cols].copy()
    net_long = net.melt("Date", var_name="Component", value_name="Amount")

    area = alt.Chart(net_long).mark_area(opacity=0.6).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Amount:Q", title="€"),
        color=alt.Color("Component:N", title="Componente")
    ).properties(height=320)

    st.altair_chart((mrr_line | area).resolve_scale(y='independent'), use_container_width=True)

# ---------------- Desglose por plan (último mes filtrado) ----------------
st.subheader("Desglose por plan (último mes del periodo filtrado)")
if enr_f.empty:
    st.info("Sin datos en el rango seleccionado.")
else:
    last_month = enr_f["Date"].max()
    snap = enr_f[enr_f["Date"] == last_month].groupby("Plan", as_index=False).agg(
        Active=("Active Used", "sum"),
        MRR_fmt=("MRR Used (€)", "sum"),
        ARR_fmt=("ARR (€)", "sum"),
        New_MRR=("New MRR (calc €)", "sum"),
        Expansion=("Expansion MRR (inferred €)", "sum"),
        Churned=("Churned MRR (€)", "sum"),
        Downgraded=("Downgraded MRR (inferred €)", "sum")
    )
    st.dataframe(snap, use_container_width=True)

# ---------------- Cohortes ----------------
st.subheader("Cohortes (retención por mes de alta)")
with st.spinner("Calculando cohorts..."):
    try:
        coh = build_cohorts_fifo(enr_f)
        piv = cohorts_pivot_heatmap(coh)
        st.caption("Retención: clientes vivos / clientes de la cohorte. Valores en %.")
        st.dataframe((piv * 100).round(1), use_container_width=True)

        m = piv.reset_index().melt(id_vars="CohortLabel", var_name="Age", value_name="Retention")
        m["RetentionPct"] = (m["Retention"] * 100).round(1)
        heat = alt.Chart(m).mark_rect().encode(
            x=alt.X('Age:O', title='Mes desde alta (Age)'),
            y=alt.Y('CohortLabel:O', title='Cohorte (YYYY-MM)'),
            color=alt.Color('Retention:Q', scale=alt.Scale(scheme='blues'), title='Retención'),
            tooltip=['CohortLabel','Age','RetentionPct']
        ).properties(height=320)
        st.altair_chart(heat, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo construir la tabla de cohortes: {e}")

# ---------------- Datos fuente (opcional) ----------------
with st.expander("Ver tablas (Data enriquecida y Monthly Summary)"):
    st.write("**Data enriquecida**")
    st.dataframe(enr_f, use_container_width=True, height=300)
    st.write("**Monthly Summary**")
    st.dataframe(summ_f, use_container_width=True, height=300)
