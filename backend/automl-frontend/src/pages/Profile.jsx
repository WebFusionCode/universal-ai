import { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

const emptyProfile = {
  name: "",
  email: "",
  phone: "",
  dob: "",
  profile_pic: "",
};

export default function Profile() {
  const [user, setUser] = useState(emptyProfile);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const userId = localStorage.getItem("user_id");

  useEffect(() => {
    let isMounted = true;

    const fetchProfile = async () => {
      if (!userId) {
        setLoading(false);
        return;
      }

      try {
        const res = await API.get(`/profile/${userId}`);

        if (isMounted) {
          setUser({ ...emptyProfile, ...res.data });
        }
      } catch (err) {
        if (isMounted) {
          setMessage(
            err.response?.data?.detail || "Unable to load profile details.",
          );
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchProfile();

    return () => {
      isMounted = false;
    };
  }, [userId]);

  const updateProfile = async () => {
    try {
      setSaving(true);
      setMessage("");

      const payload = {
        ...user,
        user_id: userId,
      };

      const res = await API.post("/update-profile", payload);
      setMessage(res.data.message || "Profile updated");
    } catch (err) {
      setMessage(err.response?.data?.detail || "Unable to update profile.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">Profile</h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Manage the account details tied to your AutoML workspace and saved
            training history.
          </p>
        </div>

        <section className="glass max-w-3xl rounded-3xl border border-white/10 p-6">
          {loading ? (
            <p className="text-slate-400">Loading profile...</p>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm text-slate-400">Name</span>
                <input
                  value={user.name || ""}
                  onChange={(event) =>
                    setUser((current) => ({
                      ...current,
                      name: event.target.value,
                    }))
                  }
                  placeholder="Name"
                  className="w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm text-slate-400">Email</span>
                <input
                  value={user.email || ""}
                  disabled
                  className="w-full rounded-2xl border border-slate-800 bg-black/40 p-3 text-slate-500"
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm text-slate-400">Phone</span>
                <input
                  value={user.phone || ""}
                  onChange={(event) =>
                    setUser((current) => ({
                      ...current,
                      phone: event.target.value,
                    }))
                  }
                  placeholder="Phone"
                  className="w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm text-slate-400">Date of Birth</span>
                <input
                  value={user.dob || ""}
                  onChange={(event) =>
                    setUser((current) => ({
                      ...current,
                      dob: event.target.value,
                    }))
                  }
                  placeholder="DOB"
                  className="w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
                />
              </label>

              <label className="space-y-2 md:col-span-2">
                <span className="text-sm text-slate-400">
                  Profile Image URL
                </span>
                <input
                  value={user.profile_pic || ""}
                  onChange={(event) =>
                    setUser((current) => ({
                      ...current,
                      profile_pic: event.target.value,
                    }))
                  }
                  placeholder="https://..."
                  className="w-full rounded-2xl border border-slate-700 bg-black/60 p-3"
                />
              </label>
            </div>
          )}

          {user.profile_pic ? (
            <img
              src={user.profile_pic}
              alt="Profile"
              className="mt-6 h-28 w-28 rounded-3xl border border-cyan-400/30 object-cover"
            />
          ) : null}

          {message ? (
            <p className="mt-4 text-sm text-cyan-300">{message}</p>
          ) : null}

          <button
            onClick={updateProfile}
            disabled={loading || saving || !userId}
            className="mt-6 rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {saving ? "Saving..." : "Save Profile"}
          </button>
        </section>
      </main>
    </div>
  );
}
