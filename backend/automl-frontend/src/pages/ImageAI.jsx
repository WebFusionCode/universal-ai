import { useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

export default function ImageAI() {
  const [prompt, setPrompt] = useState("");
  const [generatedImage, setGeneratedImage] = useState("");
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmap, setHeatmap] = useState("");
  const [generating, setGenerating] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const generateImage = async () => {
    if (!prompt.trim()) {
      setError("Enter a prompt to generate an image.");
      return;
    }

    try {
      setGenerating(true);
      setError("");

      const res = await API.post("/generate-image", { prompt });
      setGeneratedImage(res.data.image_url || "");
    } catch (err) {
      setError(err.response?.data?.detail || "Unable to generate image.");
    } finally {
      setGenerating(false);
    }
  };

  const analyzeImage = async () => {
    if (!file) {
      setError("Upload an image for classification.");
      return;
    }

    const predictionFormData = new FormData();
    predictionFormData.append("file", file);

    const explainFormData = new FormData();
    explainFormData.append("file", file);

    try {
      setAnalyzing(true);
      setError("");

      const predictionRes = await API.post(
        "/predict-image",
        predictionFormData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        },
      );

      setPrediction(predictionRes.data);

      const heatmapRes = await API.post("/explain-image", explainFormData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: "blob",
      });

      setHeatmap(URL.createObjectURL(heatmapRes.data));
    } catch (err) {
      setError(err.response?.data?.detail || "Unable to analyze image.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">
            Image AI Workspace
          </h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Generate new visuals from prompts or inspect uploaded images with
            the trained classifier and explanation heatmap.
          </p>
        </div>

        {error ? (
          <div className="glass rounded-2xl border border-red-500/40 p-4 text-red-300">
            {error}
          </div>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-2">
          <section className="glass rounded-3xl border border-white/10 p-6">
            <h2 className="text-xl font-semibold text-cyan-300">
              AI Image Generator
            </h2>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Describe the image you want to generate..."
              className="mt-4 h-32 w-full rounded-2xl border border-slate-700 bg-black/60 p-4 text-sm"
            />

            <button
              onClick={generateImage}
              disabled={generating}
              className="mt-4 rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02] disabled:opacity-60"
            >
              {generating ? "Generating..." : "Generate"}
            </button>

            {generatedImage ? (
              <img
                src={generatedImage}
                alt="Generated"
                className="mt-6 rounded-3xl border border-cyan-400/20"
              />
            ) : null}
          </section>

          <section className="glass rounded-3xl border border-white/10 p-6">
            <h2 className="text-xl font-semibold text-cyan-300">
              Image Analysis
            </h2>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                setFile(event.target.files?.[0] || null);
                setPrediction(null);
                setHeatmap("");
                setError("");
              }}
              className="mt-4 block w-full rounded-2xl border border-slate-700 bg-black/40 p-3"
            />

            <button
              onClick={analyzeImage}
              disabled={analyzing}
              className="mt-4 rounded-2xl border border-cyan-400/40 px-5 py-3 font-medium text-cyan-300 transition hover:bg-cyan-400/10 disabled:opacity-60"
            >
              {analyzing ? "Analyzing..." : "Analyze Image"}
            </button>

            {prediction ? (
              <div className="mt-6 rounded-2xl border border-white/10 bg-black/30 p-4">
                <p className="text-sm text-slate-400">Prediction</p>
                <p className="mt-2 text-lg font-semibold">
                  {prediction.predicted_class}
                </p>
                <p className="mt-2 text-sm text-slate-400">
                  Confidence: {prediction.confidence}
                </p>
              </div>
            ) : null}

            {heatmap ? (
              <div className="mt-6 rounded-2xl border border-white/10 bg-black/30 p-4">
                <p className="text-sm text-slate-400">Explanation Heatmap</p>
                <img src={heatmap} alt="Heatmap" className="mt-4 rounded-2xl" />
              </div>
            ) : null}
          </section>
        </div>
      </main>
    </div>
  );
}
