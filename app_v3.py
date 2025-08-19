
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="SaaS Investor Dashboard", page_icon="📈", layout="wide")

DEFAULT_MULTIPLES = {"Academy/Starter":8.0,"Basic":8.0,"Advance":9.0,"Pro":10.0}
def fmt_eur(x): 
    try: return f"{x:,.0f} €".replace(",",".")
    except: return str(x)

@st.cache_data
def read_book(file):
    name = file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(file)
        return {"Data": df, "Prices": None}
    try:
        return pd.read_excel(file, sheet_name=None, engine="openpyxl")
    except Exception:
        return pd.read_excel(file, sheet_name=None)

def ensure_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        st.error(f"Faltan columnas en 'Data': {miss}")
        st.stop()

def enrich_data(df_data, df_prices):
    df = df_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Plan","Date"]).reset_index(drop=True)
    prices = (df_prices or pd.DataFrame()).set_index("Plan") if df_prices is not None else pd.DataFrame().set_index(pd.Index([]))
    price_map = prices["Price MRR (€)"].to_dict() if "Price MRR (€)" in prices else {}
    mult_map = prices["Multiple (x ARR)"].to_dict() if "Multiple (x ARR)" in prices else {}
    for p in df["Plan"].unique():
        price_map.setdefault(p,0.0); mult_map.setdefault(p, DEFAULT_MULTIPLES.get(p,8.0))
    df["Price MRR (€)"] = df["Plan"].map(price_map); df["Multiple (x ARR)"]=df["Plan"].map(mult_map)
    # Active Used rollforward robust
    df["Active Used"]=np.nan
    for plan,g in df.groupby("Plan"):
        g=g.sort_values("Date")
        prev=0.0; out=[]
        for _,row in g.iterrows():
            ov=row.get("Active Customers (optional)")
            if pd.notna(ov): val=float(ov)
            else: val=max(prev+float(row["New Customers"])-float(row["Lost Customers"]),0.0)
            out.append(val); prev=val
        df.loc[g.index,"Active Used"]=out
    df["Active Used"]=df["Active Used"].astype(float)
    df["MRR Calculated (€)"]=df["Active Used"]*df["Price MRR (€)"]
    if "Real MRR (optional €)" in df.columns:
        df["Real MRR used (€)"]=df["Real MRR (optional €)"].where(pd.notna(df["Real MRR (optional €)"]), df["MRR Calculated (€)"])
    else:
        df["Real MRR used (€)"]=df["MRR Calculated (€)"]
    df["Prev Real MRR (€)"]=df.groupby("Plan")["Real MRR used (€)"].shift(1).fillna(0.0)
    df["Prev Active"]=df.groupby("Plan")["Active Used"].shift(1).fillna(0.0)
    df["Prev ARPU (€)"]=np.where(df["Prev Active"]>0, df["Prev Real MRR (€)"]/df["Prev Active"], df["Price MRR (€)"])
    df["New MRR (€)"]=df["New Customers"]*df["Price MRR (€)"]
    df["Churned MRR (€)"]=df["Lost Customers"]*df["Prev ARPU (€)"]
    df["ΔMRR Real (€)"]=df["Real MRR used (€)"]-df["Prev Real MRR (€)"]
    df["Residual (€)"]=df["ΔMRR Real (€)"]-(df["New MRR (€)"]-df["Churned MRR (€)"])
    df["Expansion MRR (inferred €)"]=df["Residual (€)"].clip(lower=0.0)
    df["Downgraded MRR (inferred €)"]=(-df["Residual (€)"]).clip(lower=0.0)
    return df

def monthly_summary(enriched):
    g=(enriched.groupby("Date",as_index=False).agg({
        "New Customers":"sum","Lost Customers":"sum","Active Used":"sum",
        "New MRR (€)":"sum","Expansion MRR (inferred €)":"sum","Churned MRR (€)":"sum",
        "Downgraded MRR (inferred €)":"sum","Real MRR used (€)":"sum"
    }).sort_values("Date"))
    g=g.rename(columns={"Active Used":"Active Customers","Real MRR used (€)":"Total MRR (€)"})
    g["Net New MRR (€)"]=g["New MRR (€)"]+g["Expansion MRR (inferred €)"]-g["Churned MRR (€)"]-g["Downgraded MRR (inferred €)"]
    g["Total ARR (€)"]=g["Total MRR (€)"]*12.0
    g["Start MRR (€)"]=g["Total MRR (€)"].shift(1).fillna(0.0)
    denom=g["Churned MRR (€)"]+g["Downgraded MRR (inferred €)"]
    numer=g["New MRR (€)"]+g["Expansion MRR (inferred €)"]
    g["Quick Ratio"]=np.where(denom>0, numer/denom, np.nan)
    g["GRR %"]=np.where(g["Start MRR (€)"]>0,1-(g["Churned MRR (€)"]+g["Downgraded MRR (inferred €)"])/g["Start MRR (€)"],np.nan)
    g["NRR %"]=np.where(g["Start MRR (€)"]>0,1+(g["Expansion MRR (inferred €)"]-(g["Churned MRR (€)"]+g["Downgraded MRR (inferred €)"]))/g["Start MRR (€)"],np.nan)
    g["MoM Growth %"]=g["Total MRR (€)"].pct_change()
    g["Churn % (customers)"]=g["Lost Customers"]/g["Active Customers"].shift(1).replace(0,np.nan)
    g["ARPU (€)"]=g["Total MRR (€)"]/g["Active Customers"].replace(0,np.nan)
    return g

st.title("📈 SaaS Investor Dashboard")

with st.sidebar:
    up = st.file_uploader("Sube tu Excel", type=["xlsx","xls","csv"])
    st.markdown("---")
    apply_to_kpis = st.toggle("Aplicar filtros a KPIs", value=True)
    months_map = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    month_choices = st.multiselect("Meses", list(months_map.values()), default=list(months_map.values()))
    components = st.multiselect("Componentes Net New", ["New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"],
                                default=["New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"])
    metric_type = st.selectbox("Métrica para gráficos", ["MRR","ARR"], index=0)

if not up:
    st.info("Sube tu fichero (usa SaaS_Final_Template_PRECOMPUTED.xlsx para probar).")
    st.stop()

book = read_book(up)
df_data = book.get("Data"); df_prices = book.get("Prices")
ensure_cols(df_data, ["Date","Plan","New Customers","Lost Customers"])

enr = enrich_data(df_data, df_prices)
summ = monthly_summary(enr)

# Filters years, months, plan
years = sorted(pd.to_datetime(enr["Date"]).dt.year.unique().tolist())
with st.sidebar:
    years_sel = st.multiselect("Años", years, default=years)
    plan_sel = st.selectbox("Plan", ["(Todos)"] + sorted(enr["Plan"].unique().tolist()))

def keep_year_month(d):
    d = pd.to_datetime(d)
    return (d.year in years_sel) and (d.strftime("%B") in month_choices)

enr_f = enr[enr["Date"].apply(keep_year_month)].copy()
summ_f = summ[summ["Date"].apply(keep_year_month)].copy()
if plan_sel != "(Todos)":
    enr_f = enr_f[enr_f["Plan"]==plan_sel].copy()

# KPIs
def header_from(summary_df):
    if summary_df.empty:
        return 0,0,0,0,np.nan,np.nan
    last = summary_df.iloc[-1]
    growth = (last["Total MRR (€)"]/summary_df.iloc[0]["Total MRR (€)"]-1) if summary_df.iloc[0]["Total MRR (€)"]>0 else np.nan
    # Approx YTD composite on filtered
    grr = np.prod(summary_df["GRR %"].dropna().values) if summary_df["GRR %"].notna().any() else np.nan
    nrr = np.prod(summary_df["NRR %"].dropna().values) if summary_df["NRR %"].notna().any() else np.nan
    return int(last["Active Customers"]), last["Total MRR (€)"], last["Total ARR (€)"], last["Net New MRR (€)"], grr, nrr

if apply_to_kpis:
    ac, mrr, arr, netnew, grr_ytd, nrr_ytd = header_from(summ_f)
else:
    ac, mrr, arr, netnew, grr_ytd, nrr_ytd = header_from(summ)

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Clientes activos", f"{ac:,}".replace(",","."))
k2.metric("MRR total", fmt_eur(mrr))
k3.metric("ARR total", fmt_eur(arr))
k4.metric("Net New MRR (últ. mes)", fmt_eur(netnew))
k5.metric("GRR YTD", f"{(grr_ytd*100):.1f}%" if pd.notna(grr_ytd) else "—")
k6.metric("NRR YTD", f"{(nrr_ytd*100):.1f}%" if pd.notna(nrr_ytd) else "—")

# Charts
st.markdown("### Evolución principal")
series_name = "Total ARR (€)" if metric_type=="ARR" else "Total MRR (€)"
st.line_chart(summ_f.set_index("Date")[[series_name]], use_container_width=True)

st.markdown("### Desglose del Net New (filtrable)")
st.area_chart(summ_f.set_index("Date")[components], use_container_width=True)

# Year-end close chart
st.markdown("### Cierre de año")
if not summ_f.empty:
    temp = summ_f.copy()
    temp["Year"]=pd.to_datetime(temp["Date"]).dt.year
    last_by_year = temp.sort_values("Date").groupby("Year").tail(1)
    metric_col = "Total ARR (€)" if metric_type=="ARR" else "Total MRR (€)"
    data = last_by_year[["Year", metric_col]].set_index("Year")
    st.bar_chart(data, use_container_width=True)
    st.caption("Barra = MRR/ARR del **último mes** disponible de cada año filtrado.")
else:
    st.info("No hay datos con el filtro actual.")

# Per plan snapshot for last month filtered
st.markdown("### Desglose por plan (último mes de periodo filtrado)")
if not enr_f.empty:
    last_date = enr_f["Date"].max()
    snap = enr_f[enr_f["Date"]==last_date].groupby("Plan", as_index=False).agg({
        "Active Used":"sum","Real MRR used (€)":"sum","New MRR (€)":"sum",
        "Expansion MRR (inferred €)":"sum","Churned MRR (€)":"sum","Downgraded MRR (inferred €)":"sum"
    }).rename(columns={"Active Used":"Active","Real MRR used (€)":"MRR"})
    snap["ARR"]=snap["MRR"]*12.0; snap["Mix %"]=snap["MRR"]/snap["MRR"].sum()*100.0
    st.dataframe(snap.assign(MRR_fmt=snap["MRR"].map(fmt_eur), ARR_fmt=snap["ARR"].map(fmt_eur),
                             Mix_fmt=snap["Mix %"].map(lambda x:f"{x:.1f}%"))[
        ["Plan","Active","MRR_fmt","ARR_fmt","Mix_fmt","New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]
    ], hide_index=True, use_container_width=True)
else:
    st.info("No hay datos para mostrar en el periodo filtrado.")

# Export snapshot
st.markdown("### Exportar snapshot (Resumen filtrado)")
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    summ_f.to_excel(writer, sheet_name="Monthly_Summary_Filtered", index=False)
    enr_f.to_excel(writer, sheet_name="Data_Enriched_Filtered", index=False)
st.download_button("⬇️ Descargar Excel (snapshot)", data=buf.getvalue(), file_name="investor_snapshot_filtered.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
