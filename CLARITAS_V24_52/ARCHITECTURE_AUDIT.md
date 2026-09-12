# V24.52 Architecture Audit

V24.52 keeps a single authoritative production transport path in `claritas_tardiis_core_v24_52.py`. `v24_52_model.py` constructs that model from `claritas_v24_52_config.json`; the host-only `v24_52_complete_geometry.py` is an independent oracle/manifest generator and is not a production fallback.

The active production geometry distinguishes water, air headspace, connected acrylic sidewall+bottom, and external air. `Zmean` controls the free surface; `Zwall_top` controls the physical tube extent. A dedicated headspace path handles water/air recrossing, headspace/acrylic interaction and open-top escape. Production failure diagnostics include a distinct headspace no-boundary class.

`v24_52_legacy_core.py` is retained only for explicit historical/reference use. The canonical model does not select it through an environment variable or automatic fallback. Old validator/global-orbit evidence is stored under `references/` and cannot authorize production.

Qualification architecture is fail-closed but sediment diagnostics are separately observable: complete geometry -> exact 1,000,000-ray production integrity -> five-case integrity matrix for authorization; after complete geometry passes, the six sediment diagnostic cases are executed even if a later authorization gate fails. `sediment_diagnostic_campaign_run` and `sediment_campaign_authorized` are independent statuses.
