import Sidebar from "../components/Sidebar";

export default function Experiments() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />

      <main className="flex-1 p-6">
        <h1 className="text-2xl mb-3">Experiments</h1>
        <p className="text-gray-400">
          Your saved training runs and model versions will appear here.
        </p>
      </main>
    </div>
  );
}
