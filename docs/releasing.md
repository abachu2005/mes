# Releases and Zenodo archiving

MES is distributed from **GitHub** (source + tags) and the **Hugging Face Space** (demo). We do **not** publish to PyPI or Bioconda.

## One-time: connect Zenodo to GitHub

1. Sign in at [https://zenodo.org](https://zenodo.org) with your **GitHub** account.
2. Go to **Account → GitHub** and enable access for repository `abachu2005/mes`.
3. Open [https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/) and **toggle ON** sync for `abachu2005/mes`.
4. Zenodo reads [`.zenodo.json`](../.zenodo.json) at the repo root for default metadata (title, license MIT, keywords).

## Cut a release

```bash
# From main, with CHANGELOG updated for the version
git tag -a v0.2.1 -m "v0.2.1 — MIT license, Zenodo metadata, stroke pipeline"
git push origin v0.2.1
```

On GitHub: **Releases → Draft a new release** → choose tag `v0.2.1` → paste notes from `CHANGELOG.md`.

Zenodo will build an archive automatically (may take a few minutes). Copy the new **DOI** from the Zenodo record.

## After Zenodo assigns a DOI

1. Edit `CITATION.cff` and add under `identifiers`:

   ```yaml
   - type: doi
     value: 10.5281/zenodo.XXXXXXX
   ```

2. Optionally add to `README.md` citation block.
3. Commit: `docs: add Zenodo DOI for v0.2.1`.

## Version bumps

Keep these in sync when releasing:

| File | Field |
|------|--------|
| `mes_core/__init__.py` | `__version__` |
| `pyproject.toml` | `version`, `name` (`motor-engagement-signal`) |
| `CITATION.cff` | `version` |
| `.zenodo.json` | `version` |
| `CHANGELOG.md` | dated section |

## License

All releases are **MIT** — see [`LICENSE`](../LICENSE).
