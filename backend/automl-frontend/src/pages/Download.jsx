import { useState } from "react";

export default function DownloadCenter() {
  const [format, setFormat] = useState("python");

  const downloadCode = () => {
    window.open(`http://localhost:8000/download-code/${format}`);
  };

  const downloadModel = () => {
    window.open("http://localhost:8000/download-latest-model");
  };

  return (
    <div className="p-6">

      <h2 className="text-2xl mb-6">Download Center</h2>

      {/* 📦 MODEL DOWNLOAD */}
      <div className="glass p-4 rounded-xl glow hover:scale-105 transition">
        <h3 className="text-lg mb-2 text-cyan-400">Download Trained Model</h3>

        <button
          onClick={downloadModel}
          className="bg-cyan-400 text-black px-4 py-2 rounded-lg glow hover:scale-105 transition"
        >
          Download Model (.pkl)
        </button>
      </div>

      {/* 💻 CODE DOWNLOAD */}
      <div className="glass p-4 rounded-xl glow hover:scale-105 transition">

        <h3 className="text-lg mb-4 text-cyan-400">Download Code</h3>

        <select
          value={format}
          onChange={(e) => setFormat(e.target.value)}
          className="bg-black border border-gray-600 p-2 rounded mb-4 w-full"
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
          className="bg-purple-500 px-4 py-2 rounded-lg"
        >
          Download Code
        </button>

      </div>

    </div>
  );
}