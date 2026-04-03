import { useEffect, useRef, useState } from "react";
import API from "../services/api";

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { sender: "ai", text: "Hi 👋 I am your AI assistant" },
  ]);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [analyzingDataset, setAnalyzingDataset] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const nextMessage = input.trim();

    if (!nextMessage || sending) {
      return;
    }

    const userMessage = { sender: "user", text: nextMessage };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      setSending(true);

      const res = await API.post("/chat", {
        message: nextMessage,
        dataset_info: datasetInfo ? JSON.stringify(datasetInfo) : "",
      });

      const aiMessage = {
        sender: "ai",
        text: res.data.reply,
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text:
            err.response?.data?.detail ||
            "I couldn't reach the assistant right now. Try again in a moment.",
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 h-[30rem] w-[calc(100vw-2rem)] max-w-sm overflow-hidden rounded-2xl border border-cyan-400/60 bg-[#111]/95 shadow-[0_0_30px_rgba(34,211,238,0.12)] backdrop-blur">
      <div className="border-b border-gray-700 px-4 py-3 text-cyan-400">
        AI Assistant 🤖
      </div>

      <div className="flex h-[calc(100%-8.5rem)] flex-col gap-2 overflow-y-auto p-3">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${
              msg.sender === "user"
                ? "self-end bg-cyan-400 text-black"
                : "self-start bg-gray-800 text-white"
            }`}
          >
            {msg.text}
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      <div className="flex gap-2 border-t border-gray-700 p-3">
        <div className="flex flex-1 flex-col gap-2">
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={async (e) => {
              const file = e.target.files?.[0];

              if (!file) {
                return;
              }

              const formData = new FormData();
              formData.append("file", file);

              try {
                setAnalyzingDataset(true);
                const res = await API.post("/analyze-dataset", formData, {
                  headers: { "Content-Type": "multipart/form-data" },
                });

                setDatasetInfo(res.data);
                setMessages((prev) => [
                  ...prev,
                  {
                    sender: "ai",
                    text: `Dataset loaded: ${res.data.rows} rows and ${res.data.columns.length} columns.`,
                  },
                ]);
              } catch (err) {
                console.error(err);
                setDatasetInfo(null);
                setMessages((prev) => [
                  ...prev,
                  {
                    sender: "ai",
                    text:
                      err.response?.data?.detail ||
                      "I couldn't analyze that dataset. Please upload a CSV or Excel file.",
                  },
                ]);
              } finally {
                setAnalyzingDataset(false);
                e.target.value = "";
              }
            }}
            className="text-xs text-gray-300 file:mr-3 file:rounded-md file:border-0 file:bg-gray-800 file:px-3 file:py-2 file:text-xs file:text-white"
          />

          {datasetInfo && (
            <div className="mt-2 rounded bg-black p-2 text-xs text-gray-300">
              <p>Rows: {datasetInfo.rows}</p>
              <p>Columns: {datasetInfo.columns.length}</p>
            </div>
          )}

          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            className="flex-1 rounded-lg border border-gray-700 bg-black px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
            placeholder={
              analyzingDataset
                ? "Analyzing dataset..."
                : "Ask about models, data, or errors..."
            }
          />
        </div>

        <button
          onClick={sendMessage}
          disabled={sending || analyzingDataset}
          className="rounded-lg bg-cyan-400 px-3 text-sm font-medium text-black disabled:opacity-60"
        >
          {sending ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
