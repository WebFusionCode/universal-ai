import { useState } from "react";
import API from "../services/api";

export default function Chatbot() {
  const [msg, setMsg] = useState("");
  const [chat, setChat] = useState([]);

  const sendMessage = async () => {
    if (!msg) return;

    const newChat = [...chat, { role: "user", text: msg }];
    setChat(newChat);

    const res = await API.post("/chat", { message: msg });

    setChat([...newChat, { role: "ai", text: res.data.reply }]);

    setMsg("");
  };

  return (
    <div className="p-6">
      <h2 className="text-xl mb-4">AI Assistant</h2>

      <div className="h-96 overflow-y-auto bg-[#111] p-4 rounded-xl mb-4">
        {chat.map((c, i) => (
          <div key={i} className="mb-2">
            <b>{c.role}:</b> {c.text}
          </div>
        ))}
      </div>

      <input
        value={msg}
        onChange={(e) => setMsg(e.target.value)}
        className="border p-2 w-full mb-2"
      />

      <button onClick={sendMessage} className="bg-cyan-400 px-4 py-2 rounded">
        Send
      </button>
    </div>
  );
}
