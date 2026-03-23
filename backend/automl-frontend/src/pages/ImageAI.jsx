import { useState } from "react";

export default function ImageAI() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (e) => {
    setFile(e.target.files[0]);
    setPrediction(null);
    setHeatmap(null);
  };

  const handlePredict = async () => {
    if (!file) return alert("Upload an image");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      // 🔹 Prediction
      const res = await fetch("http://localhost:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setPrediction(data);

      // 🔹 Heatmap (Grad-CAM)
      const heatmapRes = await fetch("http://localhost:8000/explain-image", {
        method: "POST",
        body: formData,
      });

      const blob = await heatmapRes.blob();
      const imgURL = URL.createObjectURL(blob);

      setHeatmap(imgURL);

    } catch (err) {
      console.error(err);
      alert("Error processing image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">

      <h2 className="text-2xl mb-6">Image AI</h2>

      {/* 📂 Upload */}
      <div className="glass p-4 rounded-xl glow hover:scale-105 transition mb-6">

        <input type="file" accept="image/*" onChange={handleUpload} />

        <button
          onClick={handlePredict}
          className="ml-4 bg-cyan-400 text-black px-4 py-2 rounded"
        >
          {loading ? "Processing..." : "Analyze Image"}
        </button>

      </div>

      {/* 🧠 Prediction */}
      {prediction && (
        <div className="glass p-4 rounded-xl glow hover:scale-105 transition mb-6">

          <h3 className="text-lg text-cyan-400 mb-2">
            Prediction
          </h3>

          <p>Class: {prediction.predicted_class}</p>
          <p>Confidence: {prediction.confidence}</p>

        </div>
      )}

      {/* 🔥 Heatmap */}
      {heatmap && (
        <div className="glass p-4 rounded-xl glow hover:scale-105 transition">

          <h3 className="text-lg text-purple-400 mb-2">
            Model Explanation (Heatmap)
          </h3>

          <img
            src={heatmap}
            alt="Heatmap"
            className="rounded-lg"
          />

        </div>
      )}

    </div>
  );
}