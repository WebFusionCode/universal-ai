import { useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import Loader from "./components/Loader";
import ProtectedRoute from "./components/ProtectedRoute";
import Compiler from "./pages/Compiler";
import Dashboard from "./pages/Dashboard";
import DownloadCenter from "./pages/Download";
import Experiments from "./pages/Experiments";
import ImageAI from "./pages/ImageAI";
import Insights from "./pages/Insights";
import Landing from "./pages/Landing";
import Leaderboard from "./pages/Leaderboard";
import Login from "./pages/Login";
import ModelExplain from "./pages/ModelExplain";
import Predict from "./pages/Predict";
import Signup from "./pages/Signup";
import Train from "./pages/Train";

export default function App() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setLoading(false);
    }, 2000);

    return () => {
      window.clearTimeout(timer);
    };
  }, []);

  if (loading) {
    return <Loader />;
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        <Route
          path="/train"
          element={
            <ProtectedRoute>
              <Train />
            </ProtectedRoute>
          }
        />

        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/predict"
          element={
            <ProtectedRoute>
              <Predict />
            </ProtectedRoute>
          }
        />

        <Route
          path="/model-explain"
          element={
            <ProtectedRoute>
              <ModelExplain />
            </ProtectedRoute>
          }
        />

        <Route
          path="/leaderboard"
          element={
            <ProtectedRoute>
              <Leaderboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/insights"
          element={
            <ProtectedRoute>
              <Insights />
            </ProtectedRoute>
          }
        />

        <Route
          path="/download"
          element={
            <ProtectedRoute>
              <DownloadCenter />
            </ProtectedRoute>
          }
        />

        <Route
          path="/image-ai"
          element={
            <ProtectedRoute>
              <ImageAI />
            </ProtectedRoute>
          }
        />

        <Route
          path="/experiments"
          element={
            <ProtectedRoute>
              <Experiments />
            </ProtectedRoute>
          }
        />

        <Route
          path="/compiler"
          element={
            <ProtectedRoute>
              <Compiler />
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
