import { useState } from "react";
import Sidebar from "../components/Sidebar";
import TrainingProgress from "../components/TrainingProgress";
import API from "../services/api";

const datasetOptions = [
  { value: "tabular", label: "Tabular" },
  { value: "time_series", label: "Time Series" },
  { value: "image", label: "Image" },
  { value: "audio", label: "Audio" },
  { value: "video", label: "Video" },
];

const backendBaseUrl = API.defaults.baseURL || "http://127.0.0.1:8000";

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
  return datasetOptions.find((item) => item.value === value)?.label || value;
}

function formatScoreLabel(result, datasetType) {
  if (datasetType === "tabular") {
    if (result.accuracy != null) {
      return `Accuracy ${result.accuracy}`;
    }

    if (result.r2 != null) {
      return `R2 ${result.r2}`;
    }
  }

  return "Training complete";
}

export default function Train() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [datasetType, setDatasetType] = useState("tabular");
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

  const handleDatasetTypeChange = (event) => {
    const nextType = event.target.value;
    setDatasetType(nextType);
    setError("");

    if (nextType === "image" || nextType === "audio" || nextType === "video") {
      setTargetColumn("");
    }
  };

  const handleFileChange = async (event) => {
    const nextFile = event.target.files?.[0] || null;
    const isZipFile = (nextFile?.name || "").toLowerCase().endsWith(".zip");

    setFile(nextFile);
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
      const nextDatasetType = resolveDatasetType(
        res.data,
        datasetType || "tabular",
      );

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

    if (datasetType === "audio" || datasetType === "video") {
      setError(
        `${formatDatasetType(datasetType)} workflows are not connected to the training backend yet.`,
      );
      return;
    }

    const isZipFile = (file.name || "").toLowerCase().endsWith(".zip");

    if (datasetType === "image" && !isZipFile) {
      setError("Image training expects a ZIP dataset with class folders.");
      return;
    }

    if (datasetType === "tabular" && !targetColumn) {
      alert("Select a target column");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    if (datasetType === "tabular" && targetColumn) {
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

      setDatasetType(resolveDatasetType(res.data, datasetType));
      setResult(res.data);

      if (datasetType === "tabular" && targetColumn) {
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
    <div className="flex min-h-screen bg-[#0b0f19] text-white">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <p className="text-sm uppercase tracking-[0.35em] text-slate-500">
            Training Studio
          </p>
          <h1 className="text-3xl font-semibold text-cyan-400">
            Dynamic AutoML Trainer
          </h1>
          <p className="max-w-3xl text-sm text-slate-400">
            Choose a dataset mode, inspect the incoming schema, launch training,
            and preview generated pipeline code without leaving the workspace.
          </p>
        </div>

        <div className="grid gap-6 xl:grid-cols-[0.9fr,1.1fr]">
          <section className="space-y-6">
            <div className="glass rounded-3xl border border-white/10 p-5">
              <label className="text-sm text-slate-400">Dataset Type</label>
              <select
                value={datasetType}
                onChange={handleDatasetTypeChange}
                className="mt-2 w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
              >
                {datasetOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <div className="mt-4 rounded-2xl bg-black/30 p-4 text-sm text-slate-400">
                <p className="text-cyan-300">
                  Dataset Type: {formatDatasetType(activeDatasetType)}
                </p>
                <p className="mt-2">
                  The UI adapts automatically based on the selected or detected
                  dataset mode.
                </p>
              </div>
            </div>

            <div className="glass rounded-3xl border border-white/10 p-5">
              <label className="text-sm text-slate-400">Upload Dataset</label>
              <input
                type="file"
                name="dataset"
                id="dataset"
                onChange={handleFileChange}
                className="mt-3 w-full rounded-2xl border border-dashed border-slate-700 bg-black/40 p-4 text-sm"
              />

              {file && (
                <p className="mt-3 text-sm text-slate-400">
                  Selected file:{" "}
                  <span className="text-slate-200">{file.name}</span>
                </p>
              )}

              {activeDatasetType === "image" && (
                <p className="mt-4 text-sm text-pink-300">
                  Image dataset detected. No target column is required.
                </p>
              )}

              {activeDatasetType === "time_series" && (
                <p className="mt-4 text-sm text-amber-300">
                  Time-series mode is active. Forecast target selection is
                  handled automatically by the backend.
                </p>
              )}

              {(activeDatasetType === "audio" ||
                activeDatasetType === "video") && (
                <p className="mt-4 text-sm text-slate-400">
                  {formatDatasetType(activeDatasetType)} support is planned, but
                  training is not connected yet.
                </p>
              )}

              {loadingPreview && (
                <p className="mt-4 text-sm text-slate-400">
                  Reading dataset columns...
                </p>
              )}
            </div>

            {activeDatasetType === "tabular" && columns.length > 0 && (
              <div className="glass rounded-3xl border border-white/10 p-5">
                <label className="text-sm text-slate-400">Target Column</label>
                <select
                  value={targetColumn}
                  onChange={(event) => setTargetColumn(event.target.value)}
                  className="mt-2 w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
                >
                  <option value="">Select target column</option>
                  {columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="glass rounded-3xl border border-white/10 p-5">
              <button
                onClick={handleTrain}
                disabled={
                  training ||
                  loadingPreview ||
                  activeDatasetType === "audio" ||
                  activeDatasetType === "video"
                }
                className="w-full rounded-2xl bg-cyan-400 px-4 py-3 font-medium text-black transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {training ? "Training..." : "Start Training"}
              </button>

              {error && <p className="mt-4 text-sm text-red-300">{error}</p>}
            </div>
          </section>

          <section className="space-y-6">
            <div className="glass rounded-3xl border border-white/10 p-5">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-cyan-300">
                  Dataset Preview
                </h2>
                {columns.length > 0 && (
                  <span className="text-sm text-slate-500">
                    {columns.length} columns
                  </span>
                )}
              </div>

              {previewRows.length > 0 ? (
                <div className="mt-4 overflow-x-auto rounded-2xl border border-white/10">
                  <table className="w-full border-collapse text-left text-sm">
                    <thead className="bg-black/40 text-slate-300">
                      <tr>
                        {columns.map((column) => (
                          <th key={column} className="p-3">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>

                    <tbody>
                      {previewRows.map((row, index) => (
                        <tr key={index} className="border-t border-white/10">
                          {columns.map((column) => (
                            <td
                              key={`${index}-${column}`}
                              className="p-3 text-slate-400"
                            >
                              {String(row[column] ?? "")}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="mt-4 text-sm text-slate-400">
                  Upload a supported dataset to see a schema and data preview.
                </p>
              )}
            </div>

            {result && (
              <div className="glass rounded-3xl border border-cyan-400/20 p-5">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                      Training Result
                    </p>
                    <h2 className="mt-2 text-2xl font-semibold text-cyan-300">
                      {result.best_model || "N/A"}
                    </h2>
                    <p className="mt-2 text-sm text-slate-400">
                      {formatScoreLabel(result, activeDatasetType)}
                    </p>
                  </div>

                  <div className="text-right text-sm text-slate-400">
                    {result.model_version && <p>{result.model_version}</p>}
                    {result.experiment?.time && <p>{result.experiment.time}</p>}
                  </div>
                </div>

                {activeDatasetType === "time_series" && (
                  <p className="mt-4 text-sm text-amber-300">
                    Time-series model trained successfully. Forecasting is ready
                    inside the prediction workflow.
                  </p>
                )}

                {activeDatasetType === "image" && (
                  <p className="mt-4 text-sm text-pink-300">
                    Image model trained successfully. Use the Image AI workflow
                    to test predictions.
                  </p>
                )}

                {result.leaderboard?.length > 0 && (
                  <div className="mt-5 space-y-3">
                    <p className="text-sm text-slate-400">Top Models</p>
                    {result.leaderboard.slice(0, 3).map((item, index) => (
                      <div
                        key={`${item.model}-${index}`}
                        className="flex items-center justify-between rounded-2xl border border-white/10 bg-black/20 px-4 py-3"
                      >
                        <div className="flex items-center gap-3">
                          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-cyan-400/10 text-cyan-300">
                            {item.rank || index + 1}
                          </span>
                          <span>{item.model}</span>
                        </div>

                        <span className="text-cyan-300">
                          {item.score != null
                            ? Number(item.score).toFixed(4)
                            : "Ready"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {result?.generated_code && (
              <div className="glass rounded-3xl border border-white/10 p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-xl font-semibold text-cyan-300">
                      Model Code Preview
                    </h2>
                    <p className="mt-2 text-sm text-slate-400">
                      Review the generated pipeline code and download it when
                      you are ready.
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() =>
                        navigator.clipboard.writeText(result.generated_code)
                      }
                      className="rounded-2xl border border-white/10 px-4 py-2 text-sm transition hover:border-cyan-400"
                    >
                      Copy
                    </button>

                    <button
                      onClick={() =>
                        window.open(
                          `${backendBaseUrl}/download-code/python`,
                          "_blank",
                          "noopener,noreferrer",
                        )
                      }
                      className="rounded-2xl bg-cyan-400 px-4 py-2 text-sm font-medium text-black"
                    >
                      Download
                    </button>
                  </div>
                </div>

                <pre className="mt-4 max-h-[28rem] overflow-auto rounded-2xl bg-black/60 p-4 text-sm text-emerald-300">
                  {result.generated_code}
                </pre>
              </div>
            )}

            {aiInsights && (
              <div className="glass rounded-3xl border border-white/10 p-5">
                <h2 className="text-xl font-semibold text-cyan-300">
                  AI Insights
                </h2>

                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                    <p className="text-sm text-slate-500">Problem Type</p>
                    <p className="mt-2">{aiInsights.problem_type}</p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                    <p className="text-sm text-slate-500">Missing Values</p>
                    <p className="mt-2">{aiInsights.missing_values}</p>
                  </div>
                </div>

                <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 p-4">
                  <p className="text-sm text-slate-500">Recommended Fix</p>
                  <p className="mt-2">{aiInsights.fix}</p>
                  {aiInsights.best_model_hint && (
                    <p className="mt-2 text-sm text-cyan-300">
                      {aiInsights.best_model_hint}
                    </p>
                  )}
                  {aiInsights.imbalance && (
                    <p className="mt-2 text-sm text-amber-300">
                      {aiInsights.imbalance}
                    </p>
                  )}
                </div>

                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                    <p className="text-sm text-slate-500">Recommended Models</p>
                    <div className="mt-3 space-y-2">
                      {aiInsights.recommended_models?.map((item, index) => (
                        <p key={index}>{item}</p>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                    <p className="text-sm text-slate-500">
                      Feature Engineering
                    </p>
                    <div className="mt-3 space-y-2">
                      {aiInsights.feature_engineering?.map((item, index) => (
                        <p key={index}>{item}</p>
                      ))}
                    </div>
                  </div>
                </div>

                {aiInsights.issues?.length > 0 && (
                  <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 p-4">
                    <p className="text-sm text-slate-500">Detected Issues</p>
                    <div className="mt-3 space-y-2 text-sm text-rose-300">
                      {aiInsights.issues.map((issue, index) => (
                        <p key={index}>{issue}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>

        <TrainingProgress />
      </main>
    </div>
  );
}
