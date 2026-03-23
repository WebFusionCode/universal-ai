import { motion as Motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";

import Sidebar from "../components/Sidebar";

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen">

      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 p-6 space-y-6">

        {/* 🧠 Header */}
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-cyan-400">
            Dashboard
          </h1>

          <button
            onClick={() => {
              localStorage.removeItem("token");
              navigate("/login");
            }}
            className="px-4 py-2 border border-red-500 rounded hover:bg-red-500 hover:text-white transition"
          >
            Logout
          </button>
        </div>

        {/* 📊 Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          {/* Models Trained */}
          <Motion.div
            whileHover={{ scale: 1.05 }}
            className="glass p-6 rounded-xl glow"
          >
            <h3 className="text-gray-400">Models Trained</h3>
            <p className="text-3xl font-bold mt-2">12</p>
          </Motion.div>

          {/* Best Accuracy */}
          <Motion.div
            whileHover={{ scale: 1.05 }}
            className="glass p-6 rounded-xl glow"
          >
            <h3 className="text-gray-400">Best Score</h3>
            <p className="text-3xl font-bold mt-2">0.91</p>
          </Motion.div>

          {/* Dataset Size */}
          <Motion.div
            whileHover={{ scale: 1.05 }}
            className="glass p-6 rounded-xl glow"
          >
            <h3 className="text-gray-400">Last Dataset Rows</h3>
            <p className="text-3xl font-bold mt-2">5,200</p>
          </Motion.div>

        </div>

        {/* 🚀 Quick Actions */}
        <div className="glass p-6 rounded-xl glow">

          <h2 className="text-xl mb-4 text-purple-400">
            Quick Actions
          </h2>

          <div className="flex flex-wrap gap-4">

            <Link
              to="/train"
              className="bg-cyan-400 text-black px-4 py-2 rounded-lg hover:scale-105 transition"
            >
              Train Model
            </Link>

            <Link
              to="/predict"
              className="bg-purple-500 px-4 py-2 rounded-lg hover:scale-105 transition"
            >
              Predict
            </Link>

            <Link
              to="/leaderboard"
              className="bg-green-500 px-4 py-2 rounded-lg hover:scale-105 transition"
            >
              View Leaderboard
            </Link>

            <Link
              to="/insights"
              className="bg-yellow-500 px-4 py-2 rounded-lg hover:scale-105 transition"
            >
              AI Insights
            </Link>

          </div>

        </div>

        {/* 🧠 Info Section */}
        <div className="glass p-6 rounded-xl glow">

          <h2 className="text-xl mb-3 text-cyan-400">
            Welcome to AutoML Lab
          </h2>

          <p className="text-gray-400">
            Train, analyze, and deploy machine learning models with ease.
            Use the sidebar to explore features like predictions, image AI,
            insights, and model downloads.
          </p>

        </div>

      </div>

    </div>
  );
}
