import { useEffect, useState } from "react";
import { motion as Motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";

import Chatbot from "../components/Chatbot";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

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
    lastRows: null,
    latestModel: "No runs yet",
  });

  useEffect(() => {
    let isMounted = true;

    const loadDashboardStats = async () => {
      try {
        const [experimentsRes, leaderboardRes] = await Promise.all([
          API.get("/experiments").catch(() => null),
          API.get("/leaderboard").catch(() => null),
        ]);

        if (!isMounted) {
          return;
        }

        const experiments = experimentsRes?.data?.experiments || [];
        const latestExperiment = experiments[0];
        const leaderboardModels = leaderboardRes?.data?.models || [];

        const experimentScores = experiments
          .map((item) => item.score)
          .filter((value) => typeof value === "number" && Number.isFinite(value));

        const leaderboardBestScore = leaderboardModels.find(
          (item) => typeof item.score === "number" && Number.isFinite(item.score),
        )?.score;

        setStats({
          totalRuns: experimentsRes?.data?.total_experiments || experiments.length,
          bestScore:
            leaderboardBestScore ??
            (experimentScores.length > 0 ? Math.max(...experimentScores) : null),
          lastRows: latestExperiment?.rows ?? null,
          latestModel:
            latestExperiment?.best_model ||
            leaderboardRes?.data?.best_model ||
            "No runs yet",
        });
      } catch {
        if (isMounted) {
          setStats((current) => current);
        }
      }
    };

    loadDashboardStats();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="flex min-h-screen bg-[radial-gradient(circle_at_top,_#0f172a,_#020617)] text-white">
      <Sidebar />

      <div className="flex-1 space-y-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-cyan-400">Dashboard</h1>
            <p className="mt-2 text-sm text-slate-400">
              Your AutoML workspace for training, comparison, prediction, and
              model delivery.
            </p>
          </div>

          <button
            onClick={() => {
              localStorage.removeItem("token");
              navigate("/login");
            }}
            className="rounded-xl border border-red-500 px-4 py-2 transition hover:bg-red-500 hover:text-white"
          >
            Logout
          </button>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <Motion.div
            whileHover={{ scale: 1.03 }}
            className="glass glow rounded-2xl p-6"
          >
            <h3 className="text-slate-400">Training Runs</h3>
            <p className="mt-2 text-3xl font-bold">{stats.totalRuns}</p>
          </Motion.div>

          <Motion.div
            whileHover={{ scale: 1.03 }}
            className="glass rounded-2xl p-6"
          >
            <h3 className="text-slate-400">Best Score</h3>
            <p className="mt-2 text-3xl font-bold">
              {formatScore(stats.bestScore)}
            </p>
          </Motion.div>

          <Motion.div
            whileHover={{ scale: 1.03 }}
            className="glass rounded-2xl p-6"
          >
            <h3 className="text-slate-400">Latest Dataset Rows</h3>
            <p className="mt-2 text-3xl font-bold">
              {stats.lastRows != null ? stats.lastRows.toLocaleString() : "N/A"}
            </p>
          </Motion.div>
        </div>

        <div className="glass rounded-2xl p-6">
          <h2 className="mb-2 text-xl text-cyan-400">Latest Best Model</h2>
          <p className="text-2xl font-semibold">{stats.latestModel}</p>
          <p className="mt-2 text-sm text-slate-400">
            This updates from your experiment history and leaderboard rather
            than static placeholder values.
          </p>
        </div>

        <div className="glass rounded-2xl p-6">
          <h2 className="mb-4 text-xl text-cyan-400">Quick Actions</h2>

          <div className="flex flex-wrap gap-4">
            <Link
              to="/train"
              className="rounded-xl bg-cyan-400 px-4 py-2 text-black transition hover:scale-105"
            >
              Train Model
            </Link>

            <Link
              to="/predict"
              className="rounded-xl bg-violet-500 px-4 py-2 transition hover:scale-105"
            >
              Predict
            </Link>

            <Link
              to="/leaderboard"
              className="rounded-xl bg-emerald-500 px-4 py-2 transition hover:scale-105"
            >
              View Leaderboard
            </Link>

            <Link
              to="/experiments"
              className="rounded-xl bg-amber-500 px-4 py-2 text-black transition hover:scale-105"
            >
              Experiment History
            </Link>

            <Link
              to="/download"
              className="rounded-xl bg-white/10 px-4 py-2 transition hover:scale-105"
            >
              Download Assets
            </Link>
          </div>
        </div>

        <div className="glass rounded-2xl p-6">
          <h2 className="mb-3 text-xl text-cyan-400">Welcome to AutoML Lab</h2>

          <p className="text-slate-400">
            Train, compare, and deploy machine learning models with a cleaner
            workflow. Explore experiments, inspect model rankings, ask the AI
            assistant for help, and export production-ready assets from one
            place.
          </p>
        </div>
      </div>

      <Chatbot />
    </div>
  );
}
