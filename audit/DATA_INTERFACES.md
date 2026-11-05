# Public test data and excluded empirical inputs

All supplied executable test samples are synthetic. A stock-code-shaped string in a unit test is an identifier, not evidence of that security's actual market history. The standalone example uses `SYNTHETIC_AUDIT` and two invented bars.

## Bars

Canonical research bars use `date`, `open`, `high`, `low`, `close`, and `volume`; some interfaces also require `exchange`, `symbol`, `amount`, or identity/status fields. Preserve missingness and the documented units. Consult `src/data/schema.py` and the consuming function instead of inferring undocumented conversions.

## Corporate-action inputs

The adjustment engine needs effective dates and explicit reference-basis parameters. Actual shareholder cash entitlement and the virtual cash used in the exchange reference-price calculation are distinct concepts. Unknown basis is not zero. A source label in a synthetic test is only a test stub, not a claim of official verification.

## Research and entry proxies

Feature computation and selection must not consume future outcomes or later-published corporate-action evidence. A future high or a later-observed one-price day can support an ex-post diagnostic, not a causal assertion that an order would have filled at the open.

## Excluded datasets

No original `data/`, `outputs/`, model, vendor binary, PDF extract, market response, gbbq cache, or empirical fixture is bundled. Real-data PIT tests and announcement-evidence tests are excluded rather than rewritten to assert synthetic data is historical truth. An auditor may supply independently licensed inputs later, under their own rights and environment.

For unavailable inputs, review interface logic and synthetic edge cases. Record empirical conclusions as unverified; do not recreate historical performance figures from invented data.
