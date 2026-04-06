import { useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

export default function AudioAI() {
  const [text, setText] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generateAudio = async () => {
    if (!text.trim()) {
      setError("Enter text to generate audio.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const res = await API.post("/text-to-speech", { text });
      setAudioUrl(res.data.audio_url || "");
    } catch (err) {
      setError(err.response?.data?.detail || "Unable to generate audio.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">Audio AI</h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Generate AI speech from plain text for demos, narrations, and
            product walkthroughs.
          </p>
        </div>

        <section className="glass rounded-3xl border border-white/10 p-6">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Enter text for AI speech..."
            className="h-40 w-full rounded-2xl border border-slate-700 bg-black/60 p-4 text-sm"
          />

          <button
            onClick={generateAudio}
            disabled={loading}
            className="mt-4 rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02] disabled:opacity-60"
          >
            {loading ? "Generating..." : "Generate Speech"}
          </button>

          {error ? (
            <p className="mt-4 text-sm text-red-300">{error}</p>
          ) : null}

          {audioUrl ? (
            <div className="mt-6 rounded-2xl border border-cyan-400/20 bg-black/30 p-4">
              <p className="text-sm text-slate-400">Preview</p>
              <audio controls className="mt-3 w-full" src={audioUrl} />
            </div>
          ) : null}
        </section>
      </main>
    </div>
  );
}
