import { useMemo, useState } from "react";
import Sidebar from "../components/Sidebar";

const starterTemplates = {
  classification: `# Classification starter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
`,
  regression: `# Regression starter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("R2:", r2_score(y_test, preds))
`,
  time_series: `# Time-series starter
import pandas as pd
from prophet import Prophet

df = pd.read_csv("train.csv")
df = df.rename(columns={"date": "ds", "sales": "y"})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)
print(forecast[["ds", "yhat"]].tail(10))
`,
};

export default function Compiler() {
  const [template, setTemplate] = useState("classification");
  const [code, setCode] = useState(starterTemplates.classification);
  const [output, setOutput] = useState(
    "Ready. Choose a template, edit your code, and use this space as a Colab-style drafting area.",
  );

  const lineCount = useMemo(() => code.split("\n").length, [code]);

  const handleTemplateChange = (event) => {
    const nextTemplate = event.target.value;
    setTemplate(nextTemplate);
    setCode(starterTemplates[nextTemplate]);
    setOutput(`Loaded ${nextTemplate.replace("_", " ")} template.`);
  };

  const handleRun = () => {
    setOutput(
      [
        "Playground preview mode",
        "",
        "This editor is ready for drafting ML pipelines, but secure server-side code execution is not wired into the frontend yet.",
        "Use Download Center or generated pipeline export for production artifacts.",
      ].join("\n"),
    );
  };

  return (
    <div className="flex min-h-screen bg-[#0b0f19] text-slate-100">
      <Sidebar />

      <main className="flex-1 space-y-6 p-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-cyan-400">
            ML Playground
          </h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Draft model code in a Colab-style workspace, swap starter templates,
            and prepare pipelines before exporting or integrating them into your
            training flow.
          </p>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.2fr,0.8fr]">
          <section className="glass rounded-3xl border border-white/10 p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-sm text-slate-400">Template</p>
                <select
                  value={template}
                  onChange={handleTemplateChange}
                  className="mt-2 rounded-xl border border-slate-700 bg-black/50 px-3 py-2"
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                  <option value="time_series">Time Series</option>
                </select>
              </div>

              <div className="text-right">
                <p className="text-sm text-slate-400">Lines</p>
                <p className="mt-2 text-lg font-semibold">{lineCount}</p>
              </div>
            </div>

            <textarea
              value={code}
              onChange={(event) => setCode(event.target.value)}
              spellCheck={false}
              className="h-[32rem] w-full rounded-2xl border border-slate-800 bg-black/70 p-4 font-mono text-sm leading-6 text-emerald-300 outline-none focus:border-cyan-400"
            />

            <div className="mt-4 flex flex-wrap gap-3">
              <button
                onClick={handleRun}
                className="rounded-xl bg-cyan-400 px-4 py-2 font-medium text-black transition hover:scale-[1.02]"
              >
                Run Code
              </button>

              <button
                onClick={() => navigator.clipboard.writeText(code)}
                className="rounded-xl border border-slate-700 px-4 py-2 transition hover:border-cyan-400"
              >
                Copy Code
              </button>
            </div>
          </section>

          <aside className="space-y-6">
            <div className="glass rounded-3xl border border-cyan-400/20 p-5">
              <h2 className="text-lg font-semibold text-cyan-300">Console</h2>
              <pre className="mt-4 min-h-[14rem] whitespace-pre-wrap rounded-2xl bg-black/50 p-4 font-mono text-sm text-slate-300">
                {output}
              </pre>
            </div>

            <div className="glass rounded-3xl border border-white/10 p-5">
              <h2 className="text-lg font-semibold text-cyan-300">
                Playground Notes
              </h2>
              <ul className="mt-4 space-y-3 text-sm text-slate-400">
                <li>Use this as a drafting workspace for generated ML code.</li>
                <li>Switch templates to bootstrap classification, regression, or forecasting flows.</li>
                <li>Use the Download Center to export production assets from the backend.</li>
              </ul>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
