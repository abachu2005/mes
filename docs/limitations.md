# Limitations & known issues

## Scientific limitations

- **Channel-count gap.** 16-channel mapped data underperforms 64-channel
  research caps for cross-subject motor-imagery classification. The
  validation benchmarks honestly report this gap.
- **Stroke validation depth.** Only two open stroke datasets (Liu2024,
  Liu2025) are used. Both are small and from a single site. Real clinical
  deployment would require IRB-approved prospective multi-site data.
- **MES weights are not patient-specific.** They are fit on healthy subjects.
  Per-subject baseline calibration partly corrects for this; future work
  should learn weights stratified by stroke severity.
- **ICA quality** degrades when source data has < 32 channels (e.g., a raw
  OpenBCI 16-channel recording). For native OpenBCI uploads we skip ICA;
  for upstream 64-channel datasets we apply ICA *before* downsampling to 16.

## Engineering limitations

- **Single-tenant.** No multi-user auth. Designed for one research group
  per Space.
- **No PHI handling.** Use pseudonymous research codes only.
- **HF Space free-tier sleep.** First request after 48 h of inactivity may
  take 1–2 min to cold-start.
- **In-process background tasks.** Long jobs run inside the FastAPI worker
  (FastAPI BackgroundTasks). Concurrent uploads share the same worker.
  For higher throughput, replace with Celery + Redis.
- **SQLite.** Fine for one-site research deployments. For multi-site,
  switch the `DATABASE_URL` to Postgres.

## Regulatory

- **Research use only.** Not FDA or CE cleared. No claims about diagnosis,
  treatment, or prognosis.
