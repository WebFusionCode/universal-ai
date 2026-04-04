import { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import API from "../services/api";

const plans = [
  {
    name: "free",
    title: "Free",
    description:
      "Get started with core AutoML workflows and local experimentation.",
  },
  {
    name: "pro",
    title: "Pro",
    description:
      "Unlock advanced AI tooling and a stronger day-to-day workspace.",
  },
  {
    name: "enterprise",
    title: "Enterprise",
    description:
      "Best for teams, collaboration, and production-ready AI operations.",
  },
];

export default function Pricing() {
  const userId = localStorage.getItem("user_id");
  const [currentPlan, setCurrentPlan] = useState("free");
  const [loadingPlan, setLoadingPlan] = useState(true);
  const [subscribingPlan, setSubscribingPlan] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    let isMounted = true;

    const fetchPlan = async () => {
      if (!userId) {
        setLoadingPlan(false);
        return;
      }

      try {
        const res = await API.get(`/get-plan/${userId}`);

        if (isMounted) {
          const nextPlan = res.data.plan || "free";
          setCurrentPlan(nextPlan);
          localStorage.setItem("user_plan", nextPlan);
        }
      } catch (err) {
        if (isMounted) {
          setMessage(
            err.response?.data?.detail || "Unable to load current plan.",
          );
        }
      } finally {
        if (isMounted) {
          setLoadingPlan(false);
        }
      }
    };

    fetchPlan();

    return () => {
      isMounted = false;
    };
  }, [userId]);

  const subscribe = async (plan) => {
    try {
      setSubscribingPlan(plan);
      setMessage("");

      const res = await API.post("/subscribe", {
        user_id: userId,
        plan,
      });

      setCurrentPlan(plan);
      localStorage.setItem("user_plan", plan);
      setMessage(res.data.message || `Subscribed to ${plan}`);
    } catch (err) {
      setMessage(err.response?.data?.detail || "Unable to update plan.");
    } finally {
      setSubscribingPlan("");
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">Pricing</h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Choose the plan that fits your workspace. Payments are simulated for
            now so we can wire the product flow before adding Stripe.
          </p>
        </div>

        <div className="glass rounded-3xl border border-white/10 p-5 text-sm text-slate-400">
          Current plan:{" "}
          <span className="font-semibold uppercase text-cyan-300">
            {loadingPlan ? "Loading..." : currentPlan}
          </span>
        </div>

        {message ? (
          <div className="glass rounded-2xl border border-cyan-400/20 p-4 text-cyan-300">
            {message}
          </div>
        ) : null}

        <div className="grid gap-4 xl:grid-cols-3">
          {plans.map((plan) => {
            const isCurrent = currentPlan === plan.name;

            return (
              <section
                key={plan.name}
                className={[
                  "glass rounded-3xl border p-6 transition",
                  isCurrent ? "border-cyan-400/50" : "border-white/10",
                ].join(" ")}
              >
                <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                  {plan.title}
                </p>
                <p className="mt-4 text-sm text-slate-400">
                  {plan.description}
                </p>

                <button
                  onClick={() => subscribe(plan.name)}
                  disabled={subscribingPlan === plan.name || isCurrent}
                  className="mt-6 rounded-2xl bg-cyan-400 px-5 py-3 font-medium text-black transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isCurrent
                    ? "Current Plan"
                    : subscribingPlan === plan.name
                      ? "Updating..."
                      : `Choose ${plan.title}`}
                </button>
              </section>
            );
          })}
        </div>
      </main>
    </div>
  );
}
