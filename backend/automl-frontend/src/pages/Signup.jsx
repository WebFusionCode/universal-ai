import { useState } from "react";
import { useNavigate } from "react-router-dom";
import API from "../services/api";

export default function Signup() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSignup = async () => {
    try {
      const res = await API.post("/signup", {
        email,
        password,
      });

      if (res.data.message) {
        alert("Signup successful");
        navigate("/login");
      } else {
        alert("Signup failed");
      }
    } catch (err) {
      console.error(err);
      alert("Error signing up");
    }
  };

  return (
    <div className="h-screen flex justify-center items-center">
      <div className="glass p-8 rounded-xl w-96 glow">
        <h2 className="text-2xl mb-4 text-cyan-400">Create Account</h2>

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
          onClick={handleSignup}
          className="w-full bg-cyan-400 text-black px-4 py-2 rounded-lg glow"
        >
          Sign Up
        </button>

        <p
          onClick={() => navigate("/login")}
          className="mt-4 text-sm cursor-pointer text-gray-400 hover:text-cyan-400"
        >
          Already have an account? Login
        </p>
      </div>
    </div>
  );
}
