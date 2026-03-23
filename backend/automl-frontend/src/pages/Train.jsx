import { useState } from "react";
import API from "../services/api";
import TrainingProgress from "../components/TrainingProgress";

export default function Train() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [previewRows, setPreviewRows] = useState([]);
  const [error, setError] = useState("");
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = async (event) => {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    setColumns([]);
    setTargetColumn("");
    setPreviewRows([]);
    setResult(null);
    setError("");

    if (!nextFile) {
      return;
    }

    const formData = new FormData();
    formData.append("file", nextFile);

    try {
      setLoadingPreview(true);

      const res = await API.post("/preview", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const nextColumns = res.data.columns || [];

      setColumns(nextColumns);

// show first 5 rows manually (preview fix)
      setPreviewRows([]);

// auto select target
      setTargetColumn(res.data.suggested_target_columns?.[0] || "");
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "Unable to read file columns");
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleTrain = async () => {
    if (!file) {
      alert("Upload a dataset first");
      return;
    }

    if (!targetColumn) {
      alert("Select a target column");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", targetColumn);

    try {
      setTraining(true);
      setError("");

      const res = await API.post("/train", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data.error) {
        setError(res.data.error);
        return;
      }

      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "Training failed");
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="p-6">

      <h2 className="text-xl mb-4">Train Model</h2>

      <input type="file" onChange={handleFileChange} />

      {loadingPreview && (
        <p className="mt-3 text-gray-400">Reading dataset columns...</p>
      )}

      {columns.length > 0 && (
        <div className="mt-4 space-y-4">
          <div>
            <label className="block mb-2 text-sm text-gray-300">
              Target Column
            </label>

            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="bg-black border border-gray-600 p-2 rounded w-full max-w-sm"
            >
              <option value="">Select target column</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>

          {previewRows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">

                <thead>
                  <tr className="border-b border-gray-700">
                    {columns.map((column) => (
                      <th key={column} className="p-2">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {previewRows.map((row, index) => (
                    <tr key={index} className="border-b border-gray-800">
                      {columns.map((column) => (
                        <td key={`${index}-${column}`} className="p-2">
                          {String(row[column] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>

              </table>
            </div>
          )}
        </div>
      )}

      <button
        onClick={handleTrain}
        disabled={training || loadingPreview}
        className="bg-cyan-400 text-black px-4 py-2 rounded-lg glow hover:scale-105 transition"
      >
        {training ? "Training..." : "Start Training"}
      </button>

      {error && (
        <p className="mt-4 text-red-400">{error}</p>
      )}

      {result && (
        <div className="mt-4 rounded-xl border border-cyan-400 bg-[#111] p-4">
          <p>Best Model: {result.best_model}</p>
          <p>Target Column: {result.target_column}</p>
          {"accuracy" in result && <p>Accuracy: {result.accuracy}</p>}
          {"r2" in result && <p>R2 Score: {result.r2}</p>}
        </div>
      )}

      {/* 🔥 LIVE TRAINING UI */}
      <TrainingProgress />

    </div>
  );
}
