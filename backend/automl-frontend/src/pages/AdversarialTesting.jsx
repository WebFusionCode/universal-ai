import { useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

export default function AdversarialTesting() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runTest = async () => {
    if (!file) {
      setError("Upload a CSV or XLSX dataset first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      setError("");

      const res = await API.post("/adversarial-test", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Adversarial test failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">
            Adversarial Testing
          </h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Stress-test numeric features by injecting controlled noise and
            preview the transformed samples before retraining.
          </p>
        </div>

        <section className="glass rounded-3xl border border-white/10 p-6">
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={(event) => {
              setFile(event.target.files?.[0] || null);
              setResult(null);
              setError("");
            }}
            className="block w-full rounded-2xl border border-slate-700 bg-black/40 p-3"
          />

          <button
            onClick={runTest}
            disabled={loading}
            className="mt-4 rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02] disabled:opacity-60"
          >
            {loading ? "Running..." : "Run Adversarial Test"}
          </button>

          {error ? <p className="mt-4 text-sm text-red-300">{error}</p> : null}
        </section>

        {result ? (
          <section className="glass rounded-3xl border border-cyan-400/20 p-6">
            <h2 className="text-xl font-semibold text-cyan-300">Result</h2>
            <p className="mt-3 text-slate-300">{result.message}</p>
            <p className="mt-2 text-sm text-slate-400">
              Rows affected: {result.rows}
            </p>
            <p className="mt-2 text-sm text-slate-400">
              Numeric columns:{" "}
              {(result.numeric_columns || []).join(", ") || "None"}
            </p>

            {result.preview?.length ? (
              <div className="mt-5 overflow-auto rounded-2xl border border-white/10">
                <table className="min-w-full divide-y divide-white/10 text-sm">
                  <thead className="bg-black/30">
                    <tr>
                      {Object.keys(result.preview[0]).map((column) => (
                        <th
                          key={column}
                          className="px-4 py-3 text-left font-medium"
                        >
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {result.preview.map((row, index) => (
                      <tr key={index}>
                        {Object.values(row).map((value, valueIndex) => (
                          <td
                            key={valueIndex}
                            className="px-4 py-3 text-slate-300"
                          >
                            {String(value)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </section>
        ) : null}
      </main>
    </div>
  );
}
