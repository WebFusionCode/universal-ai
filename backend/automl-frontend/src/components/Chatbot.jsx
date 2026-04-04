import { useEffect, useRef, useState } from "react";
import API from "../services/api";

export default function Chatbot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      sender: "ai",
      text: "Hi. I am your AI assistant for datasets, models, and training issues.",
    },
  ]);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [analyzingDataset, setAnalyzingDataset] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (open) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, open]);

  const sendMessage = async () => {
    const nextMessage = input.trim();

    if (!nextMessage || sending) {
      return;
    }

    setMessages((prev) => [...prev, { sender: "user", text: nextMessage }]);
    setInput("");

    try {
      setSending(true);

      const res = await API.post("/chat", {
        message: nextMessage,
        dataset_info: datasetInfo ? JSON.stringify(datasetInfo) : "",
      });

      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: res.data.reply,
        },
      ]);
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
    <>
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="fixed bottom-5 right-5 z-50 rounded-full bg-cyan-400 px-4 py-3 font-medium text-black shadow-[0_0_25px_rgba(34,211,238,0.4)] transition hover:scale-105"
      >
        {open ? "Close AI" : "AI Assistant"}
      </button>

      {open && (
        <div className="fixed bottom-20 right-4 z-50 h-[32rem] w-[calc(100vw-2rem)] max-w-sm overflow-hidden rounded-3xl border border-white/10 bg-[#111827]/95 shadow-[0_25px_80px_rgba(0,0,0,0.45)] backdrop-blur">
          <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
            <div>
              <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
                Assistant
              </p>
              <p className="mt-1 font-medium text-cyan-300">AI Copilot</p>
            </div>

            <button
              onClick={() => setOpen(false)}
              className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300"
            >
              Hide
            </button>
          </div>

          <div className="flex h-[calc(100%-10rem)] flex-col gap-2 overflow-y-auto p-4">
            {messages.map((msg, index) => (
              <div
                key={`${msg.sender}-${index}`}
                className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${
                  msg.sender === "user"
                    ? "self-end bg-cyan-400 text-black"
                    : "self-start bg-white/5 text-slate-100"
                }`}
              >
                {msg.text}
              </div>
            ))}

            <div ref={messagesEndRef} />
          </div>

          <div className="space-y-2 border-t border-white/10 p-3">
            <input
              type="file"
              accept=".csv,.xlsx"
              onChange={async (event) => {
                const file = event.target.files?.[0];

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
                  event.target.value = "";
                }
              }}
              className="text-xs text-slate-300 file:mr-3 file:rounded-xl file:border-0 file:bg-white/10 file:px-3 file:py-2 file:text-xs file:text-slate-200"
            />

            {datasetInfo && (
              <div className="rounded-2xl bg-black/40 p-3 text-xs text-slate-300">
                <p>Rows: {datasetInfo.rows}</p>
                <p>Columns: {datasetInfo.columns.length}</p>
              </div>
            )}

            <div className="flex gap-2">
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                  }
                }}
                className="flex-1 rounded-2xl border border-white/10 bg-black/60 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
                placeholder={
                  analyzingDataset
                    ? "Analyzing dataset..."
                    : "Ask about your models or data..."
                }
              />

              <button
                onClick={sendMessage}
                disabled={sending || analyzingDataset}
                className="rounded-2xl bg-cyan-400 px-4 text-sm font-medium text-black disabled:opacity-60"
              >
                {sending ? "..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
