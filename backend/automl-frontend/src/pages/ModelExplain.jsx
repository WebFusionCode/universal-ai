import { useState, useEffect } from "react";
import API from "../services/api";
import { motion as Motion } from "framer-motion";

export default function ModelExplain() {
  const [explain, setExplain] = useState(null);

  const handleExplain = async () => {
    try {
      const res = await API.get("/model-explain");
      console.log(res.data);
      setExplain(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    const load = async () => {
      await handleExplain();
    };
    load();
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await API.get("/model-explain");
        setExplain(res.data);
      } catch (err) {
        console.error(err);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-3xl text-cyan-400 font-bold">Model Explanation</h2>

      {!explain && <p>Loading explanation...</p>}

      {explain && (
        <Motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass p-6 rounded-xl glow"
        >
          <p className="mb-4">Model: {explain.model_type}</p>

          {explain.feature_importance &&
            Object.entries(explain.feature_importance).map(([key, val]) => (
              <div key={key} className="mb-2">
                <p className="text-sm">{key}</p>
                <div className="bg-gray-800 h-2 rounded">
                  <div
                    className="bg-cyan-400 h-2 rounded"
                    style={{ width: `${val * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}

          {explain.coefficients &&
            Object.entries(explain.coefficients).map(([key, val]) => (
              <p key={key}>
                {key}: {val.toFixed(4)}
              </p>
            ))}
        </Motion.div>
      )}
    </div>
  );
}
