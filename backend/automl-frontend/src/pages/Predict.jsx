import { useState } from "react";
import API from "../services/api";
import { motion as Motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function Predict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const normalizedProblemType = String(
    result?.problem_type || "",
  ).toLowerCase();
  const isClassification = normalizedProblemType.includes("classification");
  const isRegression = normalizedProblemType.includes("regression");
  const isTimeSeries =
    normalizedProblemType.includes("time-series") ||
    normalizedProblemType.includes("time_series");

  const handleUpload = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) return alert("Upload a file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await API.post("/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data?.error) {
        setResult(null);
        alert(res.data.error);
        return;
      }

      const forecastSeries = Array.isArray(res.data?.forecast)
        ? res.data.forecast
        : Array.isArray(res.data?.forecast?.sales)
          ? res.data.forecast.sales
          : Object.values(res.data?.forecast || {}).find(Array.isArray) || [];

      const chartData = forecastSeries.map((item) => ({
        ds: String(item.ds).slice(0, 10),
        yhat: Number(Number(item.yhat).toFixed(2)),
      }));

      setResult({ ...res.data, chartData });
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-3xl font-bold text-cyan-400">Predict with Model</h2>

      <Motion.div
        whileHover={{ scale: 1.02 }}
        className="glass p-6 rounded-xl glow flex flex-col gap-4"
      >
        <input type="file" onChange={handleUpload} className="text-gray-300" />

        <button
          onClick={handlePredict}
          disabled={loading}
          className="bg-cyan-400 text-black px-4 py-2 rounded-lg hover:scale-105 transition disabled:opacity-50"
        >
          {loading ? "Predicting..." : "Run Prediction"}
        </button>
      </Motion.div>

      {result && (
        <Motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass p-6 rounded-xl glow space-y-4"
        >
          <h3 className="text-xl text-purple-400">Prediction Results</h3>

          <p className="text-gray-400">Type: {result.problem_type}</p>

          {isClassification &&
            result.predictions?.map((item, i) => (
              <Motion.div
                key={i}
                whileHover={{ scale: 1.02 }}
                className="bg-[#1a1a1a] p-4 rounded-lg border border-cyan-400"
              >
                <p className="text-lg">
                  Prediction:{" "}
                  <span className="text-cyan-400">{item.prediction}</span>
                </p>

                <p>
                  Confidence:{" "}
                  <span className="text-green-400">
                    {item.confidence ?? "N/A"}
                  </span>
                </p>

                <p>
                  Risk Level:{" "}
                  <span className="text-yellow-400">{item.risk_level}</span>
                </p>

                {item.probabilities && (
                  <div className="mt-2 text-sm text-gray-400">
                    {Object.entries(item.probabilities).map(([k, v]) => (
                      <p key={k}>
                        Class {k}: {v}
                      </p>
                    ))}
                  </div>
                )}
              </Motion.div>
            ))}

          {isRegression &&
            result.predictions?.map((val, i) => (
              <Motion.div
                key={i}
                whileHover={{ scale: 1.02 }}
                className="bg-[#1a1a1a] p-3 rounded border border-purple-400"
              >
                <p>
                  Value: <span className="text-purple-400">{val}</span>
                </p>
              </Motion.div>
            ))}

          {isTimeSeries && (
            <div className="mt-4 space-y-4">
              <div className="mb-4 space-y-1">
                <p>
                  📊 Forecast Horizon:{" "}
                  {result.forecast_horizon ?? result.chartData?.length ?? 0}
                </p>
                <p>🧠 Model Type: {result.problem_type}</p>
              </div>

              <h3 className="text-cyan-400 mb-2">Forecast Graph</h3>

              <div className="h-[300px] w-full bg-[#111] p-4 rounded-xl">
                {result.chartData?.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={result.chartData}>
                      <XAxis dataKey="ds" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="yhat" stroke="#00F5FF" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center text-gray-400">
                    No forecast points available.
                  </div>
                )}
              </div>
            </div>
          )}
        </Motion.div>
      )}
    </div>
  );
}
