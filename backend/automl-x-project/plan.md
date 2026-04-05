# plan.md — AutoML X (MVP)

## 1. Objectives
- Deliver a premium dark-mode, glassmorphic **AutoML X** product with 3 core surfaces: **Landing**, **Auth**, **Dashboard/Builder**.
- Use the **user-provided backend** as the functional core (upload/preview/train/experiments/leaderboard/export/predict) and focus implementation effort on:
  - **Porting + hardening** it to this repo/runtime (FastAPI service under `/app/backend`).
  - **Upgrading AI features** to use **Emergent LLM integrations** (instead of direct OpenAI SDK calls).
  - **Shipping Awwwards-grade UI** consistent with the provided design spec.
- Ensure the end-to-end workflow is production-usable:
  **Upload dataset → preview → train → live progress → metrics + leaderboard → export/download → predict**.

**Current status:** ✅ MVP delivered (Phase 2 complete) with 97% test pass rate and 0 critical bugs.

---

## 2. Implementation Steps

### Phase 1 — Core Workflow POC (Isolation) (SKIPPED)
**Status:** ✅ Skipped

**Reason:** User supplied a complete, working backend (≈3292 lines) already implementing dataset parsing, AutoML training, export/download, experiments, leaderboard, and AI chat.

**What replaced this phase:** A port/hardening phase to adapt the supplied backend into the current project structure and conventions.

---

### Phase 2 — V1 App Development (Port backend + Premium frontend) + 1 round E2E testing
**Goal:** Integrate the existing backend into the current repo with required fixes, then build the premium frontend (Landing/Auth/Dashboard) matching the futuristic Awwwards-style spec.

**Status:** ✅ Complete

#### What was delivered (this session)

**Backend (FastAPI) — Port + harden supplied backend**
- ✅ Ported backend into `/app/backend/server.py`.
- ✅ Applied consistent route scheme: **prefix all routes with `/api`**.
- ✅ Added health endpoint: `GET /api/health`.
- ✅ Standardized DB environment config to `MONGO_URL` + `DB_NAME`.
- ✅ Defined all missing constants referenced by the upstream code:
  - `MODEL_DIR`, `MODEL_PATH`, `LEADERBOARD_PATH`, `CNN_MODEL_PATH`, `TRAINING_REPORT_PATH`
  - `IMAGE_DATASET_FOLDER`
  - `GENERATED_PIPELINE_PATH`, `GENERATED_NOTEBOOK_PATH`, `GENERATED_API_PATH`, `GENERATED_REQUIREMENTS_PATH`, `GENERATED_DOCKERFILE_PATH`
  - `DOCKER_PACKAGE_PATH`, `FULL_PROJECT_ZIP_PATH`
  - `LIGHTWEIGHT_BLOCKED_MODELS`
- ✅ Installed/verified dependencies needed for training + parsing:
  - `scikit-learn`, `joblib`, `openpyxl`, `xgboost`, `lightgbm`, `optuna`.
- ✅ Implemented AI assistant via Emergent LLM:
  - `POST /api/chat` uses `emergentintegrations.llm.chat.LlmChat` when `EMERGENT_LLM_KEY` is set.
- ✅ WebSocket training progress supported:
  - `WS /api/ws/progress`.
- ✅ Confirmed working endpoints:
  - Auth: `/api/signup`, `/api/login`
  - Profile: `/api/profile/{user_id}`, `/api/update-profile`
  - Dataset: `/api/preview`, `/api/preview-columns`, `/api/analyze-dataset`, `/api/auto-ml-insights`
  - Training: `/api/train`
  - Experiments: `/api/experiments`
  - Leaderboard: `/api/leaderboard`
  - Predict: `/api/predict`
  - Exports: `/api/download-model`, `/api/download-model/{version}`, `/api/download-code/{fmt}`
  - Reports/Explainability: `/api/training-report`, `/api/model-explain`

**Frontend (React) — Premium Awwwards-style UI**
- ✅ Implemented premium dark UI system (base `#050505`) with:
  - Animated gradient orbs
  - Heavy glassmorphism (blur + border-white/10)
  - Deep rounded corners (20px+)
  - Neon accent highlights
  - Framer Motion entrance/scroll animations
  - Inter / Plus Jakarta Sans typography
- ✅ Built 10 pages + global layout:
  1) **Landing** (hero, features bento grid, how-it-works, builder preview, marquee, CTA, footer)
  2) **Login** (cinematic glass auth)
  3) **Signup** (cinematic glass auth)
  4) **Dashboard** (stat cards, quick actions, recent experiments)
  5) **Train Model** (upload, preview, target selection, websocket progress, results + leaderboard)
  6) **Experiments** (history table)
  7) **Leaderboard** (ranked models)
  8) **Predict** (upload dataset for prediction)
  9) **Download Center** (model/code/notebook export)
  10) **Profile** (edit name/phone/DOB)
- ✅ Implemented **floating AI Chat** assistant across dashboard pages.
- ✅ Auth-protected routes in the frontend.
- ✅ API integration:
  - Uses `/api/*` endpoints.
  - JWT stored in localStorage and sent via Authorization header.

#### Testing (end of Phase 2)
- ✅ Comprehensive testing complete.
  - Backend: **100% (13/13 tests passed)**
  - Frontend: **95% (18/19 tests passed; 1 interrupted due to transient browser crash)**
  - Overall: **97%**
- ✅ No critical bugs.
- ✅ Core flow verified end-to-end: signup/login → preview → train → leaderboard/experiments → downloads → predict.

---

### Phase 3 — OAuth + assistant context + experiments polish + 1 round E2E testing
**Goal:** Upgrade auth and assistant capabilities to “product-grade”, improve run metadata consistency, and complete OAuth.

**Status:** ⏭️ Next

**User stories**
1. As a user, I can sign in with Google and be routed into the dashboard.
2. As a user, my projects/experiments are scoped to my account reliably.
3. As a user, I can compare runs with consistent metrics schemas.
4. As a user, AI assistant uses my dataset preview + last run metrics as context.

**Steps**
- Implement Google OAuth (frontend + backend routes + callback handling).
- Decide canonical user identity (`user_id` vs email) and unify JWT subject usage.
- Persist assistant conversations + “run context” in MongoDB (dataset preview summary, target col, latest leaderboard/metrics).
- E2E test with 2 users (isolation + OAuth).

---

### Phase 4 — Production hardening + premium UX polish (iterative)
**Goal:** Scale and harden training + data handling, improve resiliency, and elevate micro-interactions.

**User stories**
1. As a user, I see clear validation errors and suggested fixes for dataset issues.
2. As a user, I can cancel training and see the UI update.
3. As a user, large datasets don’t crash the server; background jobs handle them.
4. As a user, the UI feels Awwwards-grade in motion/performance.

**Steps**
- Move training to background tasks/queue (Celery/RQ) + progress events.
- Add dataset validation (size limits, schema checks, leakage warnings).
- Observability: structured logs, run traces, error dashboards.
- UI polish: microinteractions, accessibility, performance budgets.

---

## 3. Next Actions
1. **Finalize OAuth scope:** confirm Google OAuth requirements (client ID, redirect URIs, allowed domains).
2. **Assistant context upgrade:** include dataset/leaderboard context in `/api/chat` requests and persist chat history.
3. **Experiment consistency:** standardize run metadata schema and add run comparison UI.
4. **Production hardening:** optionally introduce background training jobs for large datasets.

---

## 4. Success Criteria
### MVP (completed)
- ✅ Core flow works reliably: upload (CSV/XLSX/JSON) → preview → train → metrics + leaderboard → export → predict.
- ✅ `/api` namespace used consistently across backend; frontend points only to `/api/*`.
- ✅ AI assistant works using Emergent LLM and returns actionable guidance.
- ✅ Premium dark/glass UI matches spec (orbs, blur, hover glow, Framer Motion transitions).
- ✅ Auth works; protected routes enforce token presence.
- ✅ Testing complete with **0 critical bugs**.

### Next milestones
- Google OAuth implemented end-to-end.
- Assistant uses persistent, contextual memory per user/project.
- Background training enables larger datasets without blocking requests.
