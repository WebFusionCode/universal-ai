import { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

const statCards = [
  { key: "users", label: "Users" },
  { key: "models", label: "Models" },
  { key: "teams", label: "Teams" },
  { key: "subscriptions", label: "Subscriptions" },
  { key: "usage_events", label: "Usage Events" },
];

export default function Admin() {
  const [stats, setStats] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    const fetchStats = async () => {
      try {
        const res = await API.get("/admin-stats");

        if (isMounted) {
          setStats(res.data);
        }
      } catch (err) {
        if (isMounted) {
          setError(err.response?.data?.detail || "Unable to load admin stats.");
        }
      }
    };

    fetchStats();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">
            Admin Dashboard
          </h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Monitor platform-wide usage, team adoption, and subscription growth.
          </p>
        </div>

        {error ? (
          <div className="glass rounded-2xl border border-red-500/40 p-4 text-red-300">
            {error}
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
          {statCards.map((card) => (
            <div key={card.key} className="glass rounded-3xl border border-white/10 p-5">
              <p className="text-sm text-slate-400">{card.label}</p>
              <p className="mt-3 text-3xl font-semibold">
                {stats[card.key] ?? "—"}
              </p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
