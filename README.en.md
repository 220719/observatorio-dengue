# Observatório Dengue × Clima — Maringá

> Analytical pipeline integrating epidemiological data from InfoDengue, climate data from Open-Meteo, and (Wave 2) remote sensing via Google Earth Engine — to investigate dengue dynamics in the metropolitan region of Maringá, PR (Brazil).

[🇧🇷 Português](./README.md) · 🇬🇧 **English** (you are here)

---

> ⚠️ **English version coming soon.** For now, please refer to the [Portuguese README](./README.md) — most code identifiers are in Portuguese to match the local terminology (e.g., `ano_epi`, `semana_epi`, `casos`), but module structure and docstrings are bilingual-friendly.

## Quick summary

This project builds a reproducible data pipeline to test the hypothesis that climate variables with a 3–5 week lag (precipitation, temperature, humidity) are predictors of dengue incidence in Maringá, Paraná. It uses exclusively open data sources (InfoDengue, Open-Meteo, planned: Google Earth Engine).

**Tech stack:** Python 3.12, uv, DuckDB, pandas, epiweeks, pytest. 44 tests passing.

**Status:** Wave 1 in progress (5/8 stages done).

## Author

**Anuar Mincache** · PhD in Condensed Matter Physics · Data Scientist
- [LinkedIn](https://www.linkedin.com/in/anuar-mincache/)
- [GitHub](https://github.com/220719)
- [ORCID](https://orcid.org/0000-0001-8528-8020)

## License

[MIT](./LICENSE)