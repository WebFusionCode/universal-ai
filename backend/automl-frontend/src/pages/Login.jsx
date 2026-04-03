import { useState } from "react";
import { useNavigate } from "react-router-dom";
import API from "../services/api";

export default function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async () => {
    try {
      const res = await API.post("/login", {
        email,
        password,
      });

      if (res.data.token) {
        localStorage.setItem("token", res.data.token);
        localStorage.setItem("user_email", email);
        navigate("/dashboard");
      } else {
        alert(res.data.error || "Login failed");
      }
    } catch (err) {
      console.error(err);
      alert("Error logging in");
    }
  };

  const handleGuestAccess = () => {
    localStorage.setItem("token", "guest-session");
    localStorage.setItem("user_email", "guest@automl.local");
    navigate("/dashboard");
  };

  return (
    <div className="h-screen flex justify-center items-center">
      <div className="glass p-8 rounded-xl w-96 glow">
        <h2 className="text-2xl mb-4 text-cyan-400">Login</h2>

        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full mb-3 p-2 bg-black border border-gray-700 rounded"
          placeholder="Email"
        />

        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full mb-3 p-2 bg-black border border-gray-700 rounded"
          placeholder="Password"
        />

        <button
          onClick={handleLogin}
          className="w-full bg-cyan-400 text-black px-4 py-2 rounded-lg glow hover:scale-105 transition"
        >
          Login
        </button>

        <button
          onClick={handleGuestAccess}
          className="w-full mt-3 border border-gray-600 p-2 rounded hover:bg-gray-800"
        >
          Continue as Guest
        </button>
      </div>
    </div>
  );
}
