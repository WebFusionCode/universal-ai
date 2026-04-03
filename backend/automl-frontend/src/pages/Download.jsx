import { useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

const backendBaseUrl = API.defaults.baseURL || "http://127.0.0.1:8000";

export default function DownloadCenter() {
  const [format, setFormat] = useState("python");

  const downloadCode = () => {
    window.open(
      `${backendBaseUrl}/download-code/${format}`,
      "_blank",
      "noopener,noreferrer",
    );
  };

  const downloadModel = () => {
    window.open(
      `${backendBaseUrl}/download-model`,
      "_blank",
      "noopener,noreferrer",
    );
  };

  return (
    <div className="flex min-h-screen bg-[radial-gradient(circle_at_top,_#0b0f14,_#000000)] text-white">
      <Sidebar />

      <div className="flex-1 p-6 space-y-6">
        <div className="space-y-2">
          <h2 className="text-3xl font-semibold text-cyan-400">
            Download Center
          </h2>
          <p className="text-sm text-gray-400">
            Export your best model or generate deployment-ready code assets.
          </p>
        </div>

        <div className="glass rounded-2xl border border-cyan-400/40 p-6">
          <h3 className="text-xl font-semibold text-cyan-300">
            Download Best Model
          </h3>
          <p className="mt-2 max-w-2xl text-sm text-gray-400">
            Grab the latest trained model package as <code>best_model.pkl</code>
            .
          </p>

          <button
            onClick={downloadModel}
            className="mt-5 rounded-xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02]"
          >
            Download Best Model
          </button>
        </div>

        <div className="glass rounded-2xl border border-cyan-400/20 p-6">
          <h3 className="text-xl font-semibold text-cyan-300">
            Download Code
          </h3>
          <p className="mt-2 text-sm text-gray-400">
            Generate a script, notebook, API starter, or full project package.
          </p>

          <div className="mt-5 max-w-md space-y-4">
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              className="w-full rounded-xl border border-gray-700 bg-black/70 p-3"
            >
              <option value="python">Python Script</option>
              <option value="notebook">Jupyter Notebook</option>
              <option value="api">FastAPI Backend</option>
              <option value="requirements">Requirements File</option>
              <option value="docker">Docker Package</option>
              <option value="project">Full Project ZIP</option>
            </select>

            <button
              onClick={downloadCode}
              className="rounded-xl bg-white/10 px-5 py-3 text-white transition hover:bg-white/15"
            >
              Download Code Asset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
