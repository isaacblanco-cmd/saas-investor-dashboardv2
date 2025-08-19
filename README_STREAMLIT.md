# 📊 SaaS Investor Dashboard

Sube un Excel con las hojas **Data** y **Prices** (plantilla `SaaS_Final_Template`). La app calcula KPIs SaaS (MRR/ARR), Net New (New/Expansion/Churn/Downgrade), **NRR/GRR**, **cohorts** y **valoración** por plan.

## Archivos del repo
- `app.py` – app Streamlit con lectura robusta de Excel y cálculo estable de *Active Used* (sin `groupby.apply`).
- `requirements.txt` – dependencias (incluye `openpyxl`).
- `runtime.txt` – fija **Python 3.12** para máxima compatibilidad.
- `.gitignore` – evita subir Excels y archivos locales.
- `README.md` – este archivo.

## Excel esperado
- Hoja **Data**: `Date, Plan, New Customers, Lost Customers, Active Customers (optional), Real MRR (optional €)`
- Hoja **Prices**: `Plan, Price MRR (€), Price ARR (€), Multiple (x ARR)`

> Si `Active Customers (optional)` está vacío en un mes, la app calcula: `prev + new − lost`.

## Despliegue rápido (Streamlit Cloud)
1. Crea repo en GitHub y sube estos archivos a la **raíz**.
2. Crea la app y establece **Main file path** = `app.py`.
3. En *Advanced settings*, deja **Python version: 3.12** (coincide con `runtime.txt`).
4. Abre la app y sube tu Excel (recomendado: `SaaS_Final_Template_PRECOMPUTED.xlsx`).

