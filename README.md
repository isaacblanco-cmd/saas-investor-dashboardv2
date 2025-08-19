# SaaS Investor Dashboard

Dashboard en Streamlit para analizar KPIs SaaS: MRR/ARR, Net New desglosado, GRR/NRR, **Cohortes** por mes de alta, y desglose por plan.  
Carga tu Excel con la plantilla `SaaS_Final_Template.xlsx` (hojas **Data** y **Prices**).

## Estructura esperada del Excel

- **Hoja `Data`** (mínimo):
  - `Date` (YYYY-MM-DD o similar)
  - `Plan`
  - `New Customers`
  - `Lost Customers`
  - `Active Customers`
  - (Opcional) `Real MRR (optional €)`, `Downgraded MRR (calc €)`, `Expansion MRR (calc €)`, etc.

- **Hoja `Prices`** (opcional):
  - `Plan`
  - `Price MRR (€)`
  - `Multiple (x ARR)`

> Si `Prices` no está, el dashboard sigue funcionando; simplemente infiere menos métricas basadas en precio.

## Deploy en Streamlit Cloud

1. Crea un repo con estos archivos:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml` (opcional)
2. En Streamlit Cloud, selecciona `app.py` como main module.
3. (Opcional) En **Advanced settings**, Python 3.12.
4. Sube tu Excel en el panel lateral de la app.

## Filtros

- **Años** (multi)
- **Meses** (multi)
- **Plan** (incluye `(Todos)`)
- Opción **Aplicar filtros a KPIs** para que los totales de la cabecera respeten los filtros.

## Cohortes

Cálculo FIFO desde `New Customers` y `Lost Customers` por mes y plan.  
Si en el futuro aportas IDs por cliente, puede migrarse a cohorts de cliente real (más preciso).

## Requisitos

Consulta `requirements.txt`. Se usan `openpyxl` y `pyarrow` para lectura de Excel/CSV, `pandas`/`numpy` para cálculos, `altair` para gráficos y `streamlit` para la UI.
