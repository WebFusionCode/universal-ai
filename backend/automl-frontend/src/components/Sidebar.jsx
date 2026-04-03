import { NavLink } from "react-router-dom";

const baseNavItems = [
  { label: "Dashboard", to: "/dashboard" },
  { label: "Train", to: "/train" },
  { label: "Predict", to: "/predict" },
  { label: "Image Generator", to: "/image-ai" },
  { label: "Adversarial Testing", to: "/adversarial-testing" },
  { label: "Audio AI", to: "/audio-ai" },
  { label: "Video AI", to: "/video-ai" },
  { label: "Leaderboard", to: "/leaderboard" },
  { label: "Insights", to: "/insights" },
  { label: "Experiments", to: "/experiments" },
  { label: "Teams", to: "/teams" },
  { label: "Pricing", to: "/pricing" },
  { label: "Download", to: "/download" },
  { label: "Compiler", to: "/compiler" },
  { label: "Profile", to: "/profile" },
];

function getProfileLabel() {
  const storedEmail = localStorage.getItem("user_email");
  const storedUserId = localStorage.getItem("user_id");

  if (!storedEmail) {
    return {
      name: "Guest User",
      subtext: "Local session",
    };
  }

  if (storedEmail === "guest@automl.local") {
    return {
      name: "Guest User",
      subtext: "Explore the workspace",
    };
  }

  return {
    name: storedEmail.split("@")[0].replace(/[._-]+/g, " "),
    subtext: storedUserId || storedEmail,
  };
}

export default function Sidebar() {
  const profile = getProfileLabel();
  const isAdminUser = localStorage.getItem("user_role") === "admin";
  const navItems = isAdminUser
    ? [...baseNavItems, { label: "Admin", to: "/admin" }]
    : baseNavItems;

  return (
    <aside className="glass flex h-screen w-72 flex-col border-r border-white/10 p-5">
      <div className="mb-8 space-y-2">
        <p className="text-xs uppercase tracking-[0.35em] text-slate-500">
          AutoML Platform
        </p>
        <h1 className="text-2xl font-semibold text-cyan-400">AutoML Lab</h1>
        <p className="text-sm text-slate-400">
          Build, compare, and deploy ML workflows from one workspace.
        </p>
      </div>

      <nav className="space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              [
                "block rounded-2xl px-4 py-3 text-sm transition",
                isActive
                  ? "bg-cyan-400/15 text-cyan-300"
                  : "text-slate-300 hover:bg-white/5 hover:text-cyan-300",
              ].join(" ")
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <NavLink
        to="/profile"
        className="mt-auto rounded-2xl border-t border-white/10 pt-4 transition hover:text-cyan-300"
      >
        <p className="text-sm font-medium text-slate-100 capitalize">
          {profile.name}
        </p>
        <p className="mt-1 text-xs text-slate-500">{profile.subtext}</p>
      </NavLink>
    </aside>
  );
}
