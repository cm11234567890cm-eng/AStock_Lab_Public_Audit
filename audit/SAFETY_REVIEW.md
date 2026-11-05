# Publication review

Decision: **PASS for this bounded code snapshot**, reviewed 2026-11-05. This is a publication-scope review, not a guarantee that every possible secret format or intellectual-property claim can be detected automatically.

The repository owner explicitly authorized publication of a separate code-review snapshot. Original algorithm and research configuration files are included under that authorization. No new general-purpose open-source license has been invented. See RIGHTS_AND_EXCLUSIONS.md.

## Checks and exclusions

- The source is one fixed `main` revision, recorded in FILE_MANIFEST.json. All 56 included original files match their original SHA-256 hashes. The other 383 source-tree files are excluded with reasons.
- Real market data, corporate-action payloads, original empirical fixtures, PDFs, historical reports, trained artifacts, private operational configuration, authentication integration modules and credential files were excluded. Mixed real/synthetic test files were excluded whole.
- Final text scanning checked credential patterns, private-key markers, credential-bearing URLs, email addresses, personal user-directory paths, prohibited data/binary paths, and exact/encoded matches against existing locally configured data credentials. The comparison values were held only in local memory and never written into the snapshot or reports. No matches were found.
- Source and test review found no bundled third-party package, native library, copied vendor dataset or third-party implementation that required redistribution. Dependency names remain in requirements files; packages retain their own licenses and are installed separately.
- Public exchange URLs, security-code classification rules, fixed calendar dates, schema definitions, original strategy thresholds and code comments are retained as implementation context. They are not a redistributed vendor price/event database. A calendar or code comment does not establish current market-data accuracy.
- Original Git history and author metadata are not copied. The public repository begins with a new root commit and a GitHub noreply identity.
- The offline suite ran 90 tests with zero failures and zero skips. It made no live data calls. The synthetic adjustment example also passed. See VALIDATION.json for runtime versions and warnings.
- Before publication, all 439 tracked original working-tree files were rehashed and unchanged. Original `main` remained at the recorded SHA and repository visibility remained PRIVATE. The task does not invoke production workflows or trading.

The separate snapshot is suitable for static algorithm review and the provided synthetic regression checks. It cannot reproduce private historical results, verify vendor entitlements, validate unshared Shadow branches, or demonstrate production execution.
