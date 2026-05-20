import { test, expect } from "@playwright/test";

test("landing page renders and demo seed leads to a session report", async ({ page, request }) => {
  // Seed demo via API first so the click flow is fast & deterministic.
  const seeded = await request.post("/api/demo/seed");
  expect(seeded.ok()).toBeTruthy();
  const sessions = await seeded.json();
  expect(sessions.length).toBeGreaterThanOrEqual(2);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Motor Engagement Signal" })).toBeVisible();
  await expect(page.getByRole("button", { name: /try the live demo/i })).toBeVisible();

  await page.goto(`/sessions/${sessions[0].id}`);
  await expect(page.getByRole("heading", { name: /session report/i })).toBeVisible();
  await expect(page.getByText("MES").first()).toBeVisible();
});

test("dashboard lists participants and supports adding one", async ({ page }) => {
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: "Participants" })).toBeVisible();
  await page.getByRole("button", { name: "New participant" }).click();
  const code = `E2E-${Date.now().toString().slice(-6)}`;
  await page.getByPlaceholder("e.g. P-0042").fill(code);
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.getByText(code)).toBeVisible({ timeout: 5000 });
});

test("healthz endpoint responds ok", async ({ request }) => {
  const r = await request.get("/api/healthz");
  expect(r.ok()).toBeTruthy();
  const body = await r.json();
  expect(body.status).toBe("ok");
});
