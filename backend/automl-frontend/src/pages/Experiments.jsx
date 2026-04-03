import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

function normalizeExperiments(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (Array.isArray(payload?.experiments)) {
    return payload.experiments;
  }

  return [];
}

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "Trained";
}

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    const fetchExperiments = async () => {
      try {
        const res = await API.get("/experiments");

        if (!isMounted) {
          return;
        }

        setExperiments(normalizeExperiments(res.data));
      } catch (err) {
        if (isMounted) {
          setError(
            err.response?.data?.error || err.message || "Unable to load experiments",
          );
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchExperiments();

    return () => {
      isMounted = false;
    };
  }, []);

  const bestScoredRun = experiments.reduce((best, experiment) => {
    if (typeof experiment.score !== "number") {
      return best;
    }

    if (!best || experiment.score > best.score) {
      return experiment;
    }

    return best;
  }, null);

  const chartData = experiments
    .slice()
    .reverse()
    .map((experiment, index) => ({
      run: `Run ${index + 1}`,
      score:
        typeof experiment.score === "number" && Number.isFinite(experiment.score)
          ? experiment.score
          : null,
      model: experiment.best_model || experiment.model_name || "N/A",
    }));

  const latestRun = experiments[0];
  const latestLeaderboard = latestRun?.leaderboard || [];

  return (
    <div className="flex min-h-screen bg-[radial-gradient(circle_at_top,_#0f172a,_#020617)] text-white">
      <Sidebar />

      <main className="flex-1 p-6 space-y-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">
            Experiments History
          </h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Track every training run, compare best models, and review how your
            AutoML pipeline evolves over time.
          </p>
        </div>

        {loading && (
          <div className="glass rounded-2xl p-5 text-slate-300">
            Loading experiment history...
          </div>
        )}

        {error && (
          <div className="glass rounded-2xl border border-red-500/40 p-5 text-red-300">
            {error}
          </div>
        )}

        {!loading && !error && (
          <>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="glass glow rounded-2xl p-5">
                <p className="text-sm text-slate-400">Total Runs</p>
                <p className="mt-2 text-3xl font-semibold">{experiments.length}</p>
              </div>

              <div className="glass rounded-2xl p-5">
                <p className="text-sm text-slate-400">Best Score Seen</p>
                <p className="mt-2 text-3xl font-semibold">
                  {bestScoredRun ? formatScore(bestScoredRun.score) : "N/A"}
                </p>
                {bestScoredRun && (
                  <p className="mt-2 text-sm text-cyan-300">
                    {bestScoredRun.best_model}
                  </p>
                )}
              </div>

              <div className="glass rounded-2xl p-5">
                <p className="text-sm text-slate-400">Latest Run</p>
                <p className="mt-2 text-xl font-semibold">
                  {latestRun?.best_model || "No runs yet"}
                </p>
                {latestRun?.time && (
                  <p className="mt-2 text-sm text-slate-400">{latestRun.time}</p>
                )}
              </div>
            </div>

            {chartData.some((item) => item.score != null) && (
              <div className="grid gap-6 xl:grid-cols-[1.5fr,1fr]">
                <div className="glass rounded-2xl p-5">
                  <h2 className="text-lg font-semibold text-cyan-300">
                    Performance Trend
                  </h2>
                  <div className="mt-4 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                        <XAxis dataKey="run" />
                        <YAxis />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey="score"
                          stroke="#22d3ee"
                          strokeWidth={3}
                          dot={{ r: 4 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="glass rounded-2xl p-5">
                  <h2 className="text-lg font-semibold text-cyan-300">
                    Latest Run Models
                  </h2>
                  <div className="mt-4 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={latestLeaderboard}>
                        <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                        <XAxis dataKey="model" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="score" fill="#06b6d4" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {experiments.length > 0 ? (
              <div className="space-y-4">
                {experiments.map((experiment, index) => (
                  <div
                    key={`${experiment.time || experiment.timestamp || "exp"}-${index}`}
                    className="glass rounded-2xl border border-cyan-400/20 p-5"
                  >
                    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                      <div>
                        <p className="text-sm text-slate-400">
                          {experiment.time || experiment.timestamp || "Unknown time"}
                        </p>
                        <p className="mt-2 text-xl font-semibold">
                          {experiment.best_model || experiment.model_name || "N/A"}
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2 text-xs uppercase tracking-[0.2em] text-cyan-300">
                          {experiment.dataset_type && (
                            <span>{String(experiment.dataset_type).replaceAll("_", " ")}</span>
                          )}
                          {experiment.problem_type && (
                            <span>{String(experiment.problem_type).replaceAll("_", " ")}</span>
                          )}
                        </div>
                      </div>

                      <div className="text-left md:text-right">
                        <p className="text-sm text-slate-400">Best Score</p>
                        <p className="text-2xl font-semibold text-cyan-300">
                          {formatScore(experiment.score)}
                        </p>
                        {experiment.model_version && (
                          <p className="mt-2 text-xs text-slate-500">
                            {experiment.model_version}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          Rows
                        </p>
                        <p className="mt-2 text-lg font-medium">
                          {experiment.rows ?? "N/A"}
                        </p>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          Columns
                        </p>
                        <p className="mt-2 text-lg font-medium">
                          {experiment.columns ?? "N/A"}
                        </p>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          Models Tested
                        </p>
                        <p className="mt-2 text-lg font-medium">
                          {experiment.total_models || experiment.leaderboard?.length || 0}
                        </p>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          Best Model
                        </p>
                        <p className="mt-2 text-lg font-medium">
                          {experiment.best_model || "N/A"}
                        </p>
                      </div>
                    </div>

                    {experiment.leaderboard?.length > 0 && (
                      <div className="mt-5 space-y-2">
                        {experiment.leaderboard.map((model, modelIndex) => (
                          <div
                            key={`${model.model}-${model.rank || modelIndex}`}
                            className="flex items-center justify-between rounded-xl border border-cyan-400/10 bg-black/20 px-4 py-3"
                          >
                            <div className="flex items-center gap-3">
                              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-cyan-400/10 text-cyan-300">
                                {model.rank || modelIndex + 1}
                              </span>
                              <span>{model.model}</span>
                            </div>

                            <span className="font-medium text-cyan-300">
                              {formatScore(model.score)}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="glass rounded-2xl p-6 text-slate-400">
                No experiments yet. Train your first dataset to start building a
                history of runs.
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
