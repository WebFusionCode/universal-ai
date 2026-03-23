import { useEffect, useState } from "react";
import API from "../services/api";

export default function Insights() {
  const [insights, setInsights] = useState([]);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    Promise.all([API.get("/ai-insights"), API.get("/training-report")])
      .then(([insightsRes, reportRes]) => {
        if (!isMounted) {
          return;
        }

        if (insightsRes.data.error || reportRes.data.error) {
          setError(insightsRes.data.error || reportRes.data.error);
          return;
        }

        setInsights(insightsRes.data.insights || []);
        setReport(reportRes.data);
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || "Failed to load insights");
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="p-6 space-y-6">

      <h2 className="text-2xl">AI Insights</h2>

      {error && (
        <div className="glass p-4 rounded-xl text-red-400">
          {error}
        </div>
      )}

      {/* 🤖 Chat Style Insights */}
      <div className="glass p-4 rounded-xl glow hover:scale-105 transition space-y-3">
        {insights.map((msg, i) => (
          <div
            key={i}
            className="bg-[#1a1a1a] p-3 rounded-lg text-sm"
          >
            🤖 {msg}
          </div>
        ))}
      </div>

      {/* 📊 Dataset Quality */}
      {report && (
        <div className="glass p-4 rounded-xl glow hover:scale-105 transition">

          <h3 className="text-lg mb-3 text-cyan-400">
            Dataset Quality
          </h3>

          <p>Score: {report.dataset_quality.quality_score}</p>
          <p>Missing Ratio: {report.dataset_quality.missing_ratio}</p>
          <p>Numeric Ratio: {report.dataset_quality.numeric_feature_ratio}</p>

        </div>
      )}

      {/* 🧠 Model Strength */}
      {report && (
        <div className="glass p-4 rounded-xl glow hover:scale-105 transition">

          <h3 className="text-lg mb-3 text-purple-400">
            Model Strength
          </h3>

          <p>{report.model_strength.model_strength}</p>

          {report.model_strength.accuracy && (
            <p>Accuracy: {report.model_strength.accuracy}</p>
          )}

          {report.model_strength.r2_score && (
            <p>R2 Score: {report.model_strength.r2_score}</p>
          )}

        </div>
      )}

      {/* 📘 Explanation */}
      {report && (
        <div className="glass p-4 rounded-xl glow hover:scale-105 transition">

          <h3 className="text-lg mb-3 text-green-400">
            Explanation
          </h3>

          <p>{report.explanation}</p>

        </div>
      )}

    </div>
  );
}
