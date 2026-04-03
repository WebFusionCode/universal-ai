import { Link } from "react-router-dom";

export default function Sidebar() {
  return (
    <div className="w-64 h-screen glass p-4 flex flex-col gap-4">
      <h1 className="text-xl font-bold text-cyan-400 mb-6">AutoML Lab</h1>

      <Link to="/dashboard" className="hover:text-cyan-400">
        Home
      </Link>
      <Link to="/train" className="hover:text-cyan-400">
        Train
      </Link>
      <Link to="/predict" className="hover:text-cyan-400">
        Predict
      </Link>
      <Link to="/image-ai" className="hover:text-cyan-400">
        Image AI
      </Link>
      <Link to="/leaderboard" className="hover:text-cyan-400">
        Leaderboard
      </Link>
      <Link to="/insights" className="hover:text-cyan-400">
        Insights
      </Link>
      <Link to="/experiments" className="hover:text-cyan-400">
        Experiments
      </Link>
      <Link to="/download" className="hover:text-cyan-400">
        Download
      </Link>
    </div>
  );
}
