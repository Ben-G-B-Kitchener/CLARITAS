# V24.52 Process Implementation Audit

The canonical `run_v24_52_campaign.py` owns the runtime sequence and refreshes machine-readable status after every major stage.

1. retained boundary/microsurface GPU validation;
2. comprehensive finite-apparatus GPU geometry gate;
3. exact 1,000,000-ray production integrity qualification;
4. complete five-case cross-case integrity matrix when the million-ray gate passes;
5. six-case sediment diagnostic campaign after the geometry gate, regardless of later authorization failure;
6. final qualification/authorization decision and receipt.

If the complete geometry gate fails, sediment execution is suppressed because its geometry would be scientifically meaningless. If geometry passes but the million-ray or integrity matrix fails, sediment diagnostics still run and are labelled **DIAGNOSTIC / NOT PRODUCTION-AUTHORIZED**. This exposes actual CLARITAS sediment outputs without weakening production authorization.

`production_transport_validated=true` and `sediment_campaign_authorized=true` require all production prerequisites. A diagnostic sediment run never changes those gates. The receipt is written only after full authorization and includes hashes of the active core, validators, config and decision file.

The five-case matrix always executes all five cases when reached; it no longer stops at the first failure, so the release reports the whole cross-case picture in one run.
