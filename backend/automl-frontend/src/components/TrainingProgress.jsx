import { useEffect, useState } from "react";

export default function TrainingProgress() {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Idle");
  const [logs, setLogs] = useState([]);
  const [eta, setEta] = useState("");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/progress");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      setProgress(data.progress || 0);
      setStatus(data.status || "Idle");
      setLogs(data.logs || []);
      setEta(data.eta || "");
    };

    return () => ws.close();
  }, []);

  return (
    <div className="bg-[#111] p-4 rounded-xl mt-6">

      <h3 className="text-lg mb-2">Training Progress</h3>

      {/* Progress Bar */}
      <div className="w-full bg-gray-800 h-4 rounded">
        <div
          className="bg-cyan-400 h-4 rounded transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Status */}
      <p className="mt-2 text-gray-400">
        {status} {eta && `• ETA: ${eta}`}
      </p>

      {/* Logs */}
      <div className="mt-3 h-32 overflow-y-auto text-sm text-gray-300">
        {logs.map((log, i) => (
          <p key={i}>• {log}</p>
        ))}
      </div>

    </div>
  );
}