import { useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

export default function VideoAI() {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const checkVideoPipeline = async () => {
    try {
      setLoading(true);
      const res = await API.post("/video-ai");
      setMessage(res.data.message || "Video pipeline is not ready yet.");
    } catch (err) {
      setMessage(
        err.response?.data?.detail || "Unable to reach video pipeline.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">Video AI</h1>
          <p className="max-w-2xl text-sm text-slate-400">
            This workspace reserves a slot for future video-generation and
            video-understanding workflows.
          </p>
        </div>

        <section className="glass max-w-3xl rounded-3xl border border-white/10 p-6">
          <button
            onClick={checkVideoPipeline}
            disabled={loading}
            className="rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02] disabled:opacity-60"
          >
            {loading ? "Checking..." : "Check Video Pipeline"}
          </button>

          <p className="mt-4 text-slate-300">
            {message ||
              "Video AI is scaffolded and ready for the next backend stage."}
          </p>
        </section>
      </main>
    </div>
  );
}
