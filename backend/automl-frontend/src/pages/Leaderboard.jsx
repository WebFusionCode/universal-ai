import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

function normalizeLeaderboard(payload) {
  if (Array.isArray(payload)) {
    const models = payload.map((item, index) => ({
      rank: item.rank || index + 1,
      model: item.model || "Unknown",
      score: typeof item.score === "number" ? item.score : null,
      time: typeof item.time === "number" ? item.time : null,
      metrics: item.metrics || {},
    }));

    return {
      best_model: models[0]?.model || "N/A",
      model_version: "best_model.pkl",
      total_models: models.length,
      models,
    };
  }

  if (payload && Array.isArray(payload.models)) {
    return {
      best_model: payload.best_model || payload.models[0]?.model || "N/A",
      model_version: payload.model_version || "best_model.pkl",
      total_models: payload.total_models || payload.models.length,
      dataset_type: payload.dataset_type || "",
      problem_type: payload.problem_type || "",
      models: payload.models.map((item, index) => ({
        rank: item.rank || index + 1,
        model: item.model || "Unknown",
        score: typeof item.score === "number" ? item.score : null,
        time: typeof item.time === "number" ? item.time : null,
        metrics: item.metrics || {},
      })),
    };
  }

  return {
    best_model: "N/A",
    model_version: "best_model.pkl",
    total_models: 0,
    dataset_type: "",
    problem_type: "",
    models: [],
  };
}

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "Trained";
}

function formatDatasetLabel(value) {
  if (!value) {
    return "Last trained dataset";
  }

  return String(value).replaceAll("_", " ");
}

export default function Leaderboard() {
  const [data, setData] = useState(() => normalizeLeaderboard(null));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    const fetchLeaderboard = async () => {
      try {
        const res = await API.get("/leaderboard");

        if (!isMounted) {
          return;
        }

        if (res.data?.error) {
          setError(res.data.error);
          return;
        }

        setData(normalizeLeaderboard(res.data));
      } catch (err) {
        if (isMounted) {
          setError(
            err.response?.data?.error ||
              err.message ||
              "Unable to load leaderboard",
          );
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchLeaderboard();

    return () => {
      isMounted = false;
    };
  }, []);

  const models = data.models || [];
  const scoredModels = models.filter(
    (item) => typeof item.score === "number" && Number.isFinite(item.score),
  );
  const bestScore = scoredModels[0]?.score;

  return (
    <div className="flex min-h-screen bg-[radial-gradient(circle_at_top,_#0b0f14,_#000000)] text-white">
      <Sidebar />

      <div className="flex-1 p-6 space-y-6">
        <div className="space-y-2">
          <h2 className="text-3xl font-semibold text-cyan-400">Leaderboard</h2>
          <p className="text-sm text-gray-400">
            Auto-ranked models from the latest training run.
          </p>
        </div>

        {loading && (
          <div className="glass rounded-2xl p-4 text-gray-300">
            Loading leaderboard...
          </div>
        )}

        {error && (
          <div className="glass rounded-2xl border border-red-500/40 p-4 text-red-300">
            {error}
          </div>
        )}

        {!loading && !error && (
          <>
            <div className="grid gap-4 md:grid-cols-4">
              <div className="glass rounded-2xl border border-cyan-400/40 p-5">
                <p className="text-sm text-gray-400">Best Model</p>
                <p className="mt-2 text-2xl font-semibold">{data.best_model}</p>
              </div>

              <div className="glass rounded-2xl border border-cyan-400/20 p-5">
                <p className="text-sm text-gray-400">Best Score</p>
                <p className="mt-2 text-2xl font-semibold">
                  {bestScore != null ? bestScore.toFixed(4) : "Available"}
                </p>
                <p className="mt-2 text-xs uppercase tracking-[0.2em] text-cyan-300">
                  Best score highlight
                </p>
              </div>

              <div className="glass rounded-2xl border border-cyan-400/20 p-5">
                <p className="text-sm text-gray-400">Dataset</p>
                <p className="mt-2 text-lg font-semibold capitalize">
                  {formatDatasetLabel(data.dataset_type)}
                </p>
                {data.problem_type && (
                  <p className="mt-2 text-xs uppercase tracking-[0.2em] text-cyan-300">
                    {String(data.problem_type).replaceAll("_", " ")}
                  </p>
                )}
              </div>

              <div className="glass rounded-2xl border border-cyan-400/20 p-5">
                <p className="text-sm text-gray-400">Model File</p>
                <p className="mt-2 text-lg font-semibold">
                  {data.model_version}
                </p>
                <p className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500">
                  Ready for download
                </p>
              </div>
            </div>

            {scoredModels.length > 0 && (
              <div className="glass rounded-2xl border border-cyan-400/20 p-4">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={scoredModels}>
                      <XAxis dataKey="model" />
                      <YAxis />
                      <Tooltip />
                      <Bar
                        dataKey="score"
                        fill="#22d3ee"
                        radius={[8, 8, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {models.length > 0 ? (
              <div className="space-y-3">
                {models.map((item, index) => (
                  <div
                    key={`${item.model}-${item.rank || index}`}
                    className="glass flex items-center justify-between rounded-2xl border border-cyan-400/20 p-4"
                  >
                    <div className="flex items-center gap-4">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-cyan-400/15 text-cyan-300">
                        #{item.rank || index + 1}
                      </div>

                      <div>
                        <p className="text-lg font-medium">{item.model}</p>
                        <p className="text-sm text-gray-400">
                          {item.time != null
                            ? `${item.time.toFixed(2)}s training time`
                            : `Model rank #${item.rank || index + 1}`}
                        </p>
                      </div>
                    </div>

                    <div className="text-right">
                      <p className="text-lg font-semibold text-cyan-300">
                        {formatScore(item.score)}
                      </p>
                      <p className="text-xs uppercase tracking-[0.2em] text-gray-500">
                        score
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="glass rounded-2xl p-5 text-gray-400">
                No leaderboard entries are available yet.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
