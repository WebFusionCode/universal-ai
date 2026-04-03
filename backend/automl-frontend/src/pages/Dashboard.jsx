import { useEffect, useState } from "react";
import { motion as Motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";

import Chatbot from "../components/Chatbot";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

const quickActions = [
  {
    title: "Train Model",
    description: "Upload a dataset and launch an AutoML run.",
    to: "/train",
  },
  {
    title: "Predict",
    description: "Run batch predictions with the latest saved model.",
    to: "/predict",
  },
  {
    title: "Insights",
    description: "Review AI-driven recommendations and reports.",
    to: "/insights",
  },
];

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "N/A";
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    totalRuns: 0,
    bestScore: null,
    latestModel: "No runs yet",
    latestRows: null,
  });
  const [recentExperiments, setRecentExperiments] = useState([]);
  const userEmail = localStorage.getItem("user_email") || "guest@automl.local";
  const userLabel =
    userEmail === "guest@automl.local" ? "Guest User" : userEmail.split("@")[0];

  useEffect(() => {
    let isMounted = true;

    const loadDashboardData = async () => {
      try {
        const [experimentsRes, leaderboardRes] = await Promise.all([
          API.get("/experiments").catch(() => null),
          API.get("/leaderboard").catch(() => null),
        ]);

        if (!isMounted) {
          return;
        }

        const experiments = experimentsRes?.data?.experiments || [];
        const latestExperiment = experiments[0] || null;
        const leaderboard = leaderboardRes?.data || {};
        const models = leaderboard.models || [];
        const scoredModels = models.filter(
          (item) => typeof item.score === "number" && Number.isFinite(item.score),
        );

        setStats({
          totalRuns: experimentsRes?.data?.total_experiments || experiments.length,
          bestScore:
            scoredModels[0]?.score ??
            experiments
              .map((item) => item.score)
              .find(
                (item) => typeof item === "number" && Number.isFinite(item),
              ) ??
            null,
          latestModel: latestExperiment?.best_model || leaderboard.best_model || "N/A",
          latestRows: latestExperiment?.rows ?? null,
        });

        setRecentExperiments(experiments.slice(0, 3));
      } catch (err) {
        console.error(err);
      }
    };

    loadDashboardData();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="flex h-screen bg-[#0b0f19] text-white">
      <Sidebar />

      <main className="flex-1 overflow-auto p-6">
        <div className="mx-auto max-w-7xl space-y-6">
          <div className="flex flex-col gap-4 rounded-[2rem] border border-white/10 bg-[linear-gradient(135deg,rgba(34,211,238,0.14),rgba(15,23,42,0.82))] p-8 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-3">
              <p className="text-sm uppercase tracking-[0.35em] text-slate-400">
                Workspace
              </p>
              <h1 className="text-4xl font-semibold text-white">
                Welcome back, <span className="capitalize text-cyan-300">{userLabel}</span>
              </h1>
              <p className="max-w-2xl text-sm text-slate-300">
                This dashboard behaves like an ML copilot hub: train models,
                inspect performance, compare experiments, open the AI assistant,
                and move from raw data to deployable assets faster.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => navigate("/train")}
                className="rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02]"
              >
                Start Training
              </button>

              <button
                onClick={() => {
                  localStorage.removeItem("token");
                  localStorage.removeItem("user_email");
                  navigate("/login");
                }}
                className="rounded-2xl border border-red-500/60 px-5 py-3 text-red-300 transition hover:bg-red-500/15"
              >
                Logout
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.35fr,0.65fr]">
            <div className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                <Motion.div
                  whileHover={{ scale: 1.02 }}
                  className="glass rounded-3xl p-5"
                >
                  <p className="text-sm text-slate-400">Training Runs</p>
                  <p className="mt-3 text-3xl font-semibold">{stats.totalRuns}</p>
                </Motion.div>

                <Motion.div
                  whileHover={{ scale: 1.02 }}
                  className="glass rounded-3xl p-5"
                >
                  <p className="text-sm text-slate-400">Best Score</p>
                  <p className="mt-3 text-3xl font-semibold">
                    {formatScore(stats.bestScore)}
                  </p>
                </Motion.div>

                <Motion.div
                  whileHover={{ scale: 1.02 }}
                  className="glass rounded-3xl p-5"
                >
                  <p className="text-sm text-slate-400">Latest Dataset Rows</p>
                  <p className="mt-3 text-3xl font-semibold">
                    {stats.latestRows != null ? stats.latestRows.toLocaleString() : "N/A"}
                  </p>
                </Motion.div>
              </div>

              <div className="glass rounded-3xl p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                      Quick Actions
                    </p>
                    <h2 className="mt-2 text-2xl font-semibold">
                      ChatGPT-style shortcuts
                    </h2>
                  </div>

                  <Link
                    to="/compiler"
                    className="rounded-2xl border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:border-cyan-400 hover:text-cyan-300"
                  >
                    Open Compiler
                  </Link>
                </div>

                <div className="mt-5 grid gap-4 md:grid-cols-3">
                  {quickActions.map((action) => (
                    <Link
                      key={action.to}
                      to={action.to}
                      className="rounded-3xl border border-white/10 bg-black/20 p-5 transition hover:border-cyan-400/40 hover:bg-black/30"
                    >
                      <h3 className="text-lg font-semibold text-cyan-300">
                        {action.title}
                      </h3>
                      <p className="mt-3 text-sm text-slate-400">
                        {action.description}
                      </p>
                    </Link>
                  ))}
                </div>
              </div>
            </div>

            <div className="glass rounded-3xl p-5">
              <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                Recent Activity
              </p>
              <h2 className="mt-2 text-2xl font-semibold text-white">
                Latest best model
              </h2>
              <p className="mt-3 text-3xl font-semibold text-cyan-300">
                {stats.latestModel}
              </p>

              <div className="mt-6 space-y-3">
                {recentExperiments.length > 0 ? (
                  recentExperiments.map((experiment, index) => (
                    <div
                      key={`${experiment.time || "run"}-${index}`}
                      className="rounded-2xl border border-white/10 bg-black/20 p-4"
                    >
                      <p className="text-xs uppercase tracking-[0.25em] text-slate-500">
                        {experiment.time || "Recent run"}
                      </p>
                      <p className="mt-2 font-medium">{experiment.best_model}</p>
                      <p className="mt-1 text-sm text-slate-400">
                        Score: {formatScore(experiment.score)}
                      </p>
                    </div>
                  ))
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-slate-400">
                    Train a model to populate recent runs and leaderboard activity.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      <Chatbot />
    </div>
  );
}
