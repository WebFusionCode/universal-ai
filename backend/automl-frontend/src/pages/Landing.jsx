import { motion as Motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import ParticlesBG from "../components/ParticlesBG";

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen flex flex-col justify-center items-center text-center px-6 overflow-hidden">
      <ParticlesBG />

      <Motion.h1
        initial={{ opacity: 0, y: -80 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 text-6xl font-bold mb-6 text-cyan-400"
      >
        AutoML Lab
      </Motion.h1>

      <Motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="relative z-10 text-gray-400 mb-8 text-lg"
      >
        Train, Analyze & Deploy AI — Instantly
      </Motion.p>

      <Motion.div
        whileHover={{ scale: 1.05 }}
        className="relative z-10 flex gap-4"
      >
        <button
          onClick={() => navigate("/signup")}
          className="bg-cyan-400 text-black px-6 py-3 rounded-lg glow"
        >
          Start Using
        </button>

        <button
          onClick={() => navigate("/dashboard")}
          className="border border-cyan-400 px-6 py-3 rounded-lg hover:bg-cyan-400 hover:text-black transition"
        >
          Learn More
        </button>
      </Motion.div>
      <Motion.div
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 20, ease: "linear" }}
        className="pointer-events-none absolute inset-0 m-auto w-72 h-72 border border-cyan-400 rounded-full opacity-20"
      />
    </div>
  );
}
