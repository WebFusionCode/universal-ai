import { useState } from "react";
import API from "../services/api";

export default function Predict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (e) => {
    setFile(e.target.files[0]);
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

      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">

      <h2 className="text-2xl mb-6">Make Predictions</h2>

      {/* 📂 Upload */}
      <div className="glass p-4 rounded-xl glow hover:scale-105 transition mb-6">

        <input type="file" onChange={handleUpload} />

        <button
          onClick={handlePredict}
          className="bg-cyan-400 text-black px-4 py-2 rounded-lg glow hover:scale-105 transition"
        >
          {loading ? "Predicting..." : "Run Prediction"}
        </button>

      </div>

      {/* 📊 RESULTS */}
      {result && (
        <div className="glass p-4 rounded-xl glow hover:scale-105 transition">

          <h3 className="text-lg mb-4 text-cyan-400">
            Results
          </h3>

          {/* Classification */}
          {result.problem_type === "Classification" && (
            <div className="space-y-3">
              {result.predictions.map((item, i) => (
                <div key={i} className="bg-[#1a1a1a] p-3 rounded">
                  <p>Prediction: {item.prediction}</p>
                  <p>Confidence: {item.confidence}</p>
                  <p>Risk: {item.risk_level}</p>
                </div>
              ))}
            </div>
          )}

          {/* Regression */}
          {result.problem_type === "Regression" && (
            <div className="space-y-2">
              {result.predictions.map((val, i) => (
                <p key={i}>Value: {val}</p>
              ))}
            </div>
          )}

          {/* Time Series */}
          {result.problem_type?.includes("Time-Series") && (
            <pre className="text-sm overflow-auto">
              {JSON.stringify(result.forecast, null, 2)}
            </pre>
          )}

        </div>
      )}

    </div>
  );
}