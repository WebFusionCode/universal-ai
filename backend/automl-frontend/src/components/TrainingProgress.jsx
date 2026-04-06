import { useEffect, useState } from "react";

export default function TrainingProgress() {
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    let ws;

    try {
      ws = new WebSocket("ws://127.0.0.1:8000/ws/progress");

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setProgress(data);
      };

      ws.onerror = () => console.log("WS error");
      ws.onclose = () => console.log("WS closed");
    } catch (e) {
      console.log("WS failed");
    }

    return () => {
      if (ws) ws.close();
    };
  }, []);

  if (!progress)
    return <p className="text-gray-400">Waiting for training...</p>;

  return (
    <div className="mt-4">
      <div className="bg-gray-800 h-3 rounded">
        <div
          className="bg-cyan-400 h-3 rounded"
          style={{ width: `${progress.progress || 0}%` }}
        />
      </div>

      <p className="mt-2 text-sm">{progress.status}</p>

      <div className="text-xs text-gray-400 mt-2 space-y-1">
        {progress.logs?.map((log, i) => (
          <p key={i}>• {log}</p>
        ))}
      </div>
    </div>
  );
}
