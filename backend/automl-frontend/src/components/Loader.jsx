import { motion } from "framer-motion";

export default function Loader() {
  return (
    <div className="flex h-screen items-center justify-center bg-[#0b0f19]">
      <div className="flex flex-col items-center gap-5">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
          className="h-16 w-16 rounded-full border-4 border-cyan-400/40 border-t-cyan-300"
        />

        <p className="text-sm uppercase tracking-[0.35em] text-slate-400">
          Loading AutoML Lab
        </p>
      </div>
    </div>
  );
}
