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

export default function Leaderboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    API.get("/leaderboard")
      .then((res) => {
        if (isMounted && !res.data.error) {
          setData(res.data);
        } else if (isMounted) {
          setError(res.data.error || "Unable to load leaderboard");
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || "Unable to load leaderboard");
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const models = data?.models || [];

  return (
    <div className="flex min-h-screen">
      <Sidebar />

      <div className="flex-1 p-6 space-y-6">
        <h2 className="text-2xl">Model Leaderboard</h2>

        {!data && !error && <p>Loading...</p>}

        {error && (
          <div className="glass p-4 rounded-xl text-red-400">
            {error}
          </div>
        )}

        {data && (
          <>
            <div className="bg-[#111] p-4 rounded-xl border border-cyan-400">
              <h3 className="text-lg text-cyan-400">Best Model</h3>
              <p className="text-xl font-bold">{data.best_model}</p>
              <p className="text-gray-400">Version: {data.model_version}</p>
            </div>

            {models.length > 0 ? (
              <>
                <div className="glass p-4 rounded-xl">
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={models}>
                        <XAxis dataKey="model" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="score" fill="#22d3ee" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">

                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="p-2">Rank</th>
                        <th className="p-2">Model</th>
                        <th className="p-2">Score</th>
                        <th className="p-2">Time</th>
                      </tr>
                    </thead>

                    <tbody>
                      {models.map((model, index) => (
                        <tr
                          key={`${model.model}-${model.rank || index}`}
                          className={`border-b border-gray-800 ${
                            index === 0 ? "bg-[#0f172a]" : ""
                          }`}
                        >
                          <td className="p-2">{model.rank ?? index + 1}</td>
                          <td className="p-2">{model.model}</td>
                          <td className="p-2">
                            {model.score != null ? Number(model.score).toFixed(4) : "N/A"}
                          </td>
                          <td className="p-2">
                            {model.time != null ? `${Number(model.time).toFixed(2)}s` : "N/A"}
                          </td>
                        </tr>
                      ))}
                    </tbody>

                  </table>
                </div>
              </>
            ) : (
              <p className="text-gray-400">No leaderboard entries are available yet.</p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
