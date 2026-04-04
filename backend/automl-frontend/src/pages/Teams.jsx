import { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

export default function Teams() {
  const userId = localStorage.getItem("user_id");
  const [teams, setTeams] = useState([]);
  const [teamName, setTeamName] = useState("");
  const [inviteTeamId, setInviteTeamId] = useState("");
  const [memberUserId, setMemberUserId] = useState("");
  const [message, setMessage] = useState("");

  const loadTeams = async () => {
    if (!userId) {
      return;
    }

    try {
      const res = await API.get(`/teams/${userId}`);
      setTeams(res.data.teams || []);
    } catch (err) {
      setMessage(err.response?.data?.detail || "Unable to load teams.");
    }
  };

  useEffect(() => {
    loadTeams();
  }, []);

  const createTeam = async () => {
    try {
      setMessage("");
      const res = await API.post("/create-team", {
        user_id: userId,
        name: teamName,
      });

      setMessage(`Team created: ${res.data.name}`);
      setTeamName("");
      loadTeams();
    } catch (err) {
      setMessage(err.response?.data?.detail || "Unable to create team.");
    }
  };

  const inviteUser = async () => {
    try {
      setMessage("");
      const res = await API.post("/invite", {
        team_id: inviteTeamId,
        user_id: memberUserId,
      });

      setMessage(res.data.message || "User added");
      setMemberUserId("");
      loadTeams();
    } catch (err) {
      setMessage(err.response?.data?.detail || "Unable to invite user.");
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">Teams</h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Create shared workspaces and invite collaborators into your AutoML
            environment.
          </p>
        </div>

        {message ? (
          <div className="glass rounded-2xl border border-cyan-400/20 p-4 text-cyan-300">
            {message}
          </div>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-2">
          <section className="glass rounded-3xl border border-white/10 p-6">
            <h2 className="text-xl font-semibold text-cyan-300">Create Team</h2>
            <input
              value={teamName}
              onChange={(event) => setTeamName(event.target.value)}
              placeholder="Team name"
              className="mt-4 w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
            />
            <button
              onClick={createTeam}
              className="mt-4 rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02]"
            >
              Create Team
            </button>
          </section>

          <section className="glass rounded-3xl border border-white/10 p-6">
            <h2 className="text-xl font-semibold text-cyan-300">
              Invite Member
            </h2>
            <input
              value={inviteTeamId}
              onChange={(event) => setInviteTeamId(event.target.value)}
              placeholder="Team ID"
              className="mt-4 w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
            />
            <input
              value={memberUserId}
              onChange={(event) => setMemberUserId(event.target.value)}
              placeholder="User ID to invite"
              className="mt-4 w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
            />
            <button
              onClick={inviteUser}
              className="mt-4 rounded-2xl border border-cyan-400/40 px-5 py-3 font-medium text-cyan-300 transition hover:bg-cyan-400/10"
            >
              Invite User
            </button>
          </section>
        </div>

        <section className="glass rounded-3xl border border-white/10 p-6">
          <h2 className="text-xl font-semibold text-cyan-300">Your Teams</h2>
          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            {teams.length > 0 ? (
              teams.map((team) => (
                <div
                  key={team.team_id}
                  className="rounded-2xl border border-white/10 bg-black/30 p-4"
                >
                  <p className="text-lg font-semibold">
                    {team.name || "Untitled Team"}
                  </p>
                  <p className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500">
                    {team.team_id}
                  </p>
                  <p className="mt-3 text-sm text-slate-400">
                    Members: {(team.members || []).length}
                  </p>
                </div>
              ))
            ) : (
              <div className="rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-slate-400">
                No teams yet. Create one to start collaborating.
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
