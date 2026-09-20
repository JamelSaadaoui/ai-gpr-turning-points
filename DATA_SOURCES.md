# Data sources, transformations, and retrieval vintage

The package is a frozen-data reproduction built from complete current-source
histories. `code/build_data.py` reads the provider files in `data/source_csv/`
and recreates both `data/current_source_merge.csv` and
`data/monthly_data.csv`. It performs no vintage splice, level rebasing, or
extrapolation. The only fill is the documented October 2025 CPI interpolation.

## Provider files

| Variables | Provider and series | Included endpoint |
|---|---|---|
| Aggregate and eight event-type indices | Iacoviello and Tong, [AI-GPR monthly event-type data](https://www.matteoiacoviello.com/ai_gpr.html), `ai_gpr_eventtype_monthly.csv` | August 2026 |
| WTI price | U.S. EIA, Cushing, Oklahoma WTI Spot Price FOB, monthly | August 2026 |
| Brent price | U.S. EIA, Europe Brent Spot Price FOB, monthly | August 2026 |
| Consumer prices | U.S. BLS bulk series `CUSR0000SA0`, all items, seasonally adjusted, 1982–84=100 | August 2026 |
| World industrial production | Baumeister–Hamilton OECD plus six non-member economies index | June 2026 |
| World crude-oil production | EIA international series `57-1-WORL-TBPD.M`, distributed by DBnomics | February 2026 |

The source histories were downloaded on September 19, 2026. The EIA price
pages report a September 16, 2026 release. The Baumeister–Hamilton workbook was
modified August 31, 2026. The AI-GPR page reports an August 31, 2026 update.
SHA-256 hashes in `data/input_checksums.json` identify every included file.

## Construction

The calendar is January 1990–August 2026. Nominal WTI and Brent are divided by
`CUSR0000SA0/100` and logged. World industrial production and world crude-oil
production are logged directly. Oil-production growth is the first difference
of logged production. The BLS source contains no value for October 2025.
Accordingly, that CPI level is log-linearly interpolated between September and
November 2025 (324.245 and 325.063), giving approximately 324.654. The nominal
WTI and Brent observations remain the observed EIA values; only their common
deflator is filled. No other value is interpolated or carried forward.

For each raw event index `G`, the treatment is `log(1+G)` and the excluded
instrument is `G[t] - 2*G[t-1] + G[t-2]`. AI-GPR measures score-weighted news
coverage, not independent event counts. Category values retain the provider's
normalization.

All lags and leads are constructed on the complete calendar before regression-
specific complete-case selection. Every output reports its actual number and
range of regression origins; the data endpoint is not automatically the final
origin at every horizon.
