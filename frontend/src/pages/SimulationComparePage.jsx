import { useNavigate } from 'react-router-dom';

export default function SimulationComparePage() {
  const navigate = useNavigate();
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-white">Sampling comparison</h1>
      <p className="text-sm text-gray-400">
        The legacy HTML comparison view is deprecated. Use the Sampling Explorer to compare runs instead.
      </p>
      <button
        type="button"
        onClick={() => navigate('/results')}
        className="px-3 py-2 rounded-md bg-cyan-600 text-white text-sm"
      >
        Back to results
      </button>
    </div>
  );
}
