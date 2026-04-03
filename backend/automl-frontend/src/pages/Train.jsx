import { useState } from "react";
import API from "../services/api";
import TrainingProgress from "../components/TrainingProgress";

function resolveDatasetType(payload, fallback = "") {
  const explicitType = String(payload?.dataset_type || "").toLowerCase();

  if (explicitType) {
    return explicitType;
  }

  const problemType = String(payload?.problem_type || "").toLowerCase();

  if (problemType.includes("image")) {
    return "image";
  }

  if (
    problemType.includes("time_series") ||
    problemType.includes("time-series") ||
    payload?.detected_date_column
  ) {
    return "time_series";
  }

  if (payload?.columns || problemType) {
    return "tabular";
  }

  return fallback;
}

function formatDatasetType(value) {
  if (value === "time_series") {
    return "Time Series";
  }

  if (value === "image") {
    return "Image";
  }

  if (value === "tabular") {
    return "Tabular";
  }

  return value;
}

export default function Train() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [datasetType, setDatasetType] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [aiInsights, setAiInsights] = useState(null);
  const [previewRows, setPreviewRows] = useState([]);
  const [error, setError] = useState("");
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const activeDatasetType = result
    ? resolveDatasetType(result, datasetType)
    : datasetType;

  const handleFileChange = async (event) => {
    const nextFile = event.target.files?.[0] || null;
    const isZipFile = (nextFile?.name || "").toLowerCase().endsWith(".zip");
    setFile(nextFile);
    setDatasetType("");
    setColumns([]);
    setTargetColumn("");
    setAiInsights(null);
    setPreviewRows([]);
    setResult(null);
    setError("");

    if (!nextFile) {
      return;
    }

    if (isZipFile) {
      setDatasetType("image");
      return;
    }

    const formData = new FormData();
    formData.append("file", nextFile);
    let loadingTimeout;

    try {
      setLoadingPreview(true);
      loadingTimeout = window.setTimeout(() => {
        setLoadingPreview(false);
      }, 10000);

      const res = await API.post("/preview", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const nextColumns = res.data.columns || [];
      const nextDatasetType = resolveDatasetType(res.data, "tabular");

      setDatasetType(nextDatasetType);
      setColumns(nextColumns);
      setPreviewRows(res.data.preview || []);
      setTargetColumn(
        res.data.suggested_target_columns?.[0] || nextColumns[0] || "",
      );
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "Unable to read file columns");
    } finally {
      window.clearTimeout(loadingTimeout);
      setLoadingPreview(false);
    }
  };

  const handleTrain = async () => {
    if (!file) {
      alert("Upload a dataset first");
      return;
    }

    const isZipFile = (file.name || "").toLowerCase().endsWith(".zip");

    if (!isZipFile && !targetColumn) {
      alert("Select a target column");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    if (!isZipFile && targetColumn) {
      formData.append("target_column", targetColumn);
    }

    try {
      setTraining(true);
      setError("");
      setAiInsights(null);

      const res = await API.post("/train", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data.error) {
        setError(res.data.error);
        return;
      }

      setDatasetType(
        resolveDatasetType(
          res.data,
          datasetType || (isZipFile ? "image" : "tabular"),
        ),
      );
      setResult(res.data);

      if (!isZipFile && targetColumn) {
        const insightsFormData = new FormData();
        insightsFormData.append("file", file);
        insightsFormData.append("target_column", targetColumn);

        try {
          const insightsRes = await API.post(
            "/auto-ml-insights",
            insightsFormData,
            {
              headers: { "Content-Type": "multipart/form-data" },
            },
          );

          setAiInsights(insightsRes.data);
        } catch (insightsErr) {
          console.error(insightsErr);
        }
      }
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

      <input
        type="file"
        name="dataset"
        id="dataset"
        onChange={handleFileChange}
      />

      {datasetType === "image" && (
        <p className="text-pink-400 mt-3">
          Image dataset detected 🖼 - No target column required
        </p>
      )}

      {loadingPreview && (
        <p className="mt-3 text-gray-400">Reading dataset columns...</p>
      )}

      {datasetType && (
        <div className="mt-4">
          <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-lg text-sm">
            Dataset Type: {formatDatasetType(datasetType)}
          </span>
        </div>
      )}

      {columns.length > 0 && datasetType !== "image" && (
        <div className="mt-4 space-y-4">
          <div hidden={datasetType === "time_series"}>
            <label className="block mb-2 text-sm text-gray-300">
              Target Column
            </label>

            <select
              disabled={datasetType === "time_series"}
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="bg-black border border-gray-600 p-2 rounded w-full max-w-sm disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <option value="">Select target column</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>

          {datasetType === "time_series" && (
            <p className="text-yellow-400 text-sm">
              Time series detected. Forecast target is auto-selected for
              training.
            </p>
          )}

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

      {error && <p className="mt-4 text-red-400">{error}</p>}

      {result && (
        <div className="mt-4 rounded-xl border border-cyan-400 bg-[#111] p-4">
          <p className="text-cyan-400 font-semibold">
            Best Model: {result.best_model || "N/A"}
          </p>

          {activeDatasetType === "tabular" && (
            <>
              {"accuracy" in result && result.accuracy != null && (
                <p>Accuracy: {result.accuracy}</p>
              )}

              {"r2" in result && result.r2 != null && (
                <p>R2 Score: {result.r2}</p>
              )}
            </>
          )}

          {activeDatasetType === "time_series" && (
            <p className="text-yellow-400 mt-2">
              Time Series model trained successfully 📈
            </p>
          )}

          {activeDatasetType === "image" && (
            <p className="text-pink-400 mt-2">
              Image model trained successfully 🧠
            </p>
          )}
        </div>
      )}

      {aiInsights && (
        <div className="mt-6 rounded-xl border border-cyan-400 bg-[#111] p-4">
          <h3 className="mb-3 text-cyan-400">🤖 AI Insights</h3>

          <p>Problem Type: {aiInsights.problem_type}</p>
          <p>Missing Values: {aiInsights.missing_values}</p>
          <p>Fix: {aiInsights.fix}</p>

          {aiInsights.best_model_hint && (
            <p>Best Model Hint: {aiInsights.best_model_hint}</p>
          )}

          {aiInsights.imbalance && <p>Dataset: {aiInsights.imbalance}</p>}

          <div className="mt-3">
            <p className="text-sm text-gray-400">Recommended Models:</p>
            {aiInsights.recommended_models?.map((modelName, index) => (
              <p key={index}>• {modelName}</p>
            ))}
          </div>

          <div className="mt-3">
            <p className="text-sm text-gray-400">Feature Engineering:</p>
            {aiInsights.feature_engineering?.map((featureTip, index) => (
              <p key={index}>• {featureTip}</p>
            ))}
          </div>

          {aiInsights.issues?.length > 0 && (
            <div className="mt-3">
              <p className="text-sm text-gray-400">Detected Issues:</p>
              {aiInsights.issues.map((issue, index) => (
                <p key={index}>• {issue}</p>
              ))}
            </div>
          )}
        </div>
      )}

      <TrainingProgress />
    </div>
  );
}
