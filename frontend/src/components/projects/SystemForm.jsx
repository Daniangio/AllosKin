import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

const defaultForm = {
  name: '',
  description: '',
  activePdb: null,
  inactivePdb: null,
  activeTraj: null,
  inactiveTraj: null,
  activeStride: 1,
  inactiveStride: 1,
  residueSelections: '',
};

export default function SystemForm({ onCreate }) {
  const [form, setForm] = useState(defaultForm);
  const [formKey, setFormKey] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleFileChange = (field, files) => {
    handleChange(field, files?.[0] || null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.activePdb || !form.activeTraj || !form.inactiveTraj) {
      setError('Active PDB, active trajectory, and inactive trajectory are required.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setUploadProgress(null);
    setIsProcessing(false);
    try {
      const payload = new FormData();
      if (form.name) payload.append('name', form.name);
      if (form.description) payload.append('description', form.description);
      payload.append('active_pdb', form.activePdb);
      if (form.inactivePdb) payload.append('inactive_pdb', form.inactivePdb);
      payload.append('active_traj', form.activeTraj);
      payload.append('inactive_traj', form.inactiveTraj);
      payload.append('active_stride', form.activeStride);
      payload.append('inactive_stride', form.inactiveStride);
      const selectionsText = form.residueSelections.trim();
      if (selectionsText) {
        payload.append('residue_selections_text', selectionsText);
      }

      await onCreate(payload, {
        onUploadProgress: (percent) => setUploadProgress(percent),
        onProcessing: (processing) => setIsProcessing(processing),
      });
      setForm(defaultForm);
      setFormKey((prev) => prev + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
      setUploadProgress(null);
      setIsProcessing(false);
    }
  };

  return (
    <form
      key={formKey}
      onSubmit={handleSubmit}
      className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-4"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-300 mb-1">System Name</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => handleChange('name', e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
            placeholder="Inactive vs Active complex"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">Description</label>
          <input
            type="text"
            value={form.description}
            onChange={(e) => handleChange('description', e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FileInput label="Active PDB" required onChange={(files) => handleFileChange('activePdb', files)} />
        <FileInput label="Inactive PDB (optional)" onChange={(files) => handleFileChange('inactivePdb', files)} />
        <FileInput label="Active Trajectory" required onChange={(files) => handleFileChange('activeTraj', files)} />
        <FileInput label="Inactive Trajectory" required onChange={(files) => handleFileChange('inactiveTraj', files)} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StrideInput
          label="Active Stride"
          value={form.activeStride}
          onChange={(value) => handleChange('activeStride', value)}
        />
        <StrideInput
          label="Inactive Stride"
          value={form.inactiveStride}
          onChange={(value) => handleChange('inactiveStride', value)}
        />
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Residue Selections (optional)</label>
        <textarea
          rows={4}
          value={form.residueSelections}
          onChange={(e) => handleChange('residueSelections', e.target.value)}
          placeholder={'resid 50 51\nchain A and resid 10 to 15 [singles]\nsegid CORE and resid 20 to 25 [pairs]'}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        />
        <p className="text-xs text-gray-500 mt-1">
          Enter one selection per line. Use optional [singles] or [pairs] wildcards to expand entries automatically.
        </p>
      </div>

      {(uploadProgress !== null || isProcessing) && (
        <div className="space-y-3">
          {uploadProgress !== null && (
            <div>
              <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                <span>Uploading files</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500 transition-all duration-200"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}
          {isProcessing && (
            <div>
              <p className="text-xs text-gray-400 mb-1">Processing descriptors...</p>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full w-1/3 bg-amber-400 animate-pulse" />
              </div>
              <p className="text-xs text-gray-500 mt-1">This may take a few minutes.</p>
            </div>
          )}
        </div>
      )}

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Processing...' : 'Build Descriptor System'}
      </button>
    </form>
  );
}

function FileInput({ label, onChange, required }) {
  return (
    <div>
      <label className="block text-sm text-gray-300 mb-1">
        {label} {required && <span className="text-red-400">*</span>}
      </label>
      <input
        type="file"
        onChange={(e) => onChange(e.target.files)}
        required={required}
        className="block w-full text-sm text-gray-300 bg-gray-900 border border-gray-700 rounded-md cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-cyan-50 file:text-cyan-700 hover:file:bg-cyan-100"
      />
    </div>
  );
}

function StrideInput({ label, value, onChange }) {
  return (
    <div>
      <label className="block text-sm text-gray-300 mb-1">{label}</label>
      <input
        type="number"
        min={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
      />
    </div>
  );
}
