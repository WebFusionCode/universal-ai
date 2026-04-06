import { motion as Motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import ParticlesBG from "../components/ParticlesBG";

const featureCards = [
  {
    title: "Train Fast",
    description:
      "Upload tabular, time-series, or image-ready datasets and launch AutoML workflows in minutes.",
  },
  {
    title: "Explain Results",
    description:
      "Review leaderboard rankings, AI insights, experiment history, and model behavior in one place.",
  },
  {
    title: "Deploy Smarter",
    description:
      "Preview generated pipeline code, export assets, and move from experiment to delivery without context switching.",
  },
];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#0b0f19] text-white">
      <ParticlesBG />

      <section className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-24 text-center">
        <Motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="inline-flex items-center rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-xs uppercase tracking-[0.35em] text-cyan-300"
        >
          Startup-grade AutoML Workspace
        </Motion.div>

        <Motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="mt-8 max-w-5xl text-5xl font-semibold leading-tight text-white md:text-7xl"
        >
          Build AI models without coding, while keeping the power of a real ML
          platform.
        </Motion.h1>

        <Motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.25 }}
          className="mt-6 max-w-2xl text-lg text-slate-400"
        >
          Train, analyze, predict, compare experiments, and export production
          assets from a single workspace designed to feel more like a startup
          product than a classroom demo.
        </Motion.p>

        <Motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-4"
        >
          <button
            onClick={() => navigate("/signup")}
            className="rounded-2xl bg-cyan-400 px-6 py-3 font-medium text-black transition hover:scale-[1.02]"
          >
            Get Started
          </button>

          <button
            onClick={() =>
              document
                .getElementById("learn")
                ?.scrollIntoView({ behavior: "smooth" })
            }
            className="rounded-2xl border border-white/15 px-6 py-3 transition hover:border-cyan-400 hover:text-cyan-300"
          >
            Learn More
          </button>
        </Motion.div>

        <Motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.45 }}
          className="mt-16 grid w-full max-w-5xl gap-4 md:grid-cols-3"
        >
          {featureCards.map((card) => (
            <div
              key={card.title}
              className="glass rounded-3xl border border-white/10 p-6 text-left"
            >
              <h2 className="text-xl font-semibold text-cyan-300">
                {card.title}
              </h2>
              <p className="mt-3 text-sm leading-6 text-slate-400">
                {card.description}
              </p>
            </div>
          ))}
        </Motion.div>
      </section>

      <section
        id="learn"
        className="relative z-10 border-t border-white/10 px-6 py-24"
      >
        <div className="mx-auto grid max-w-6xl gap-10 lg:grid-cols-[1fr,1.2fr]">
          <div>
            <p className="text-sm uppercase tracking-[0.35em] text-slate-500">
              How It Works
            </p>
            <h2 className="mt-4 text-4xl font-semibold text-white">
              A cleaner path from raw dataset to deployed model.
            </h2>
            <p className="mt-6 max-w-xl text-slate-400">
              Upload your data, let the platform detect the problem type, train
              multiple models, compare the leaderboard, explore experiment
              history, and download the best artifacts when you are ready.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="glass rounded-3xl border border-white/10 p-5">
              <p className="text-sm text-slate-500">01</p>
              <h3 className="mt-3 text-xl font-semibold">Upload</h3>
              <p className="mt-2 text-sm text-slate-400">
                Bring in CSV, Excel, ZIP image datasets, and explore your data
                before training.
              </p>
            </div>

            <div className="glass rounded-3xl border border-white/10 p-5">
              <p className="text-sm text-slate-500">02</p>
              <h3 className="mt-3 text-xl font-semibold">Analyze</h3>
              <p className="mt-2 text-sm text-slate-400">
                Use AI guidance, feature suggestions, leaderboard ranking, and
                experiment tracking to understand performance.
              </p>
            </div>

            <div className="glass rounded-3xl border border-white/10 p-5">
              <p className="text-sm text-slate-500">03</p>
              <h3 className="mt-3 text-xl font-semibold">Ship</h3>
              <p className="mt-2 text-sm text-slate-400">
                Preview generated code, download trained models, and keep a
                reusable workflow for future runs.
              </p>
            </div>
          </div>
        </div>
      </section>

      <footer className="relative z-10 border-t border-white/10 px-6 py-6 text-center text-sm text-slate-500">
        © 2026 AutoML Platform | Built by Harsh Singh
      </footer>
    </div>
  );
}
