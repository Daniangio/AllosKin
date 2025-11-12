import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Activity, Server, CheckCircle, XCircle, AlertTriangle, 
  Loader2, Database, FileText, ChevronRight, ArrowLeft, Brain, Sliders, Zap
} from 'lucide-react';

/*
================================================================================
Main Application Component
================================================================================
*/

/**
 * Main App component. Manages routing between pages.
 */
export default function App() {
  // Simple "router" state.
  const [page, setPage] = useState('submit');
  // State to hold the job ID being tracked (the RQ job_id)
  const [pollingJobId, setPollingJobId] = useState(null);
  // State to hold the result ID being viewed (the UUID)
  const [selectedResultId, setSelectedResultId] = useState(null);

  /**
   * Navigate to the status page for a specific job ID.
   */
  const navigateToStatus = (newJobId) => {
    setPollingJobId(newJobId);
    setPage('status');
  };

  /**
   * Navigate to the results list page.
   */
  const navigateToResults = () => {
    setPage('results');
  };

  /**
   * Navigate to a specific result's detail page.
   */
  const navigateToResultDetail = (resultId) => {
    setSelectedResultId(resultId);
    setPage('result_detail');
  };

  /**
   * Render the currently active page based on state.
   */
  const renderPage = () => {
    switch (page) {
      case 'submit':
        return <SubmitJobPage onJobSubmitted={navigateToStatus} />;
      case 'status':
        return <JobStatusPage jobId={pollingJobId} onNavigateToResults={navigateToResults} />;
      case 'health':
        return <HealthCheckPage />;
      case 'results':
        return <ResultsListPage onSelectResult={navigateToResultDetail} />;
      case 'result_detail':
        return <ResultDetailPage resultId={selectedResultId} onBack={() => setPage('results')} />;
      default:
        return <SubmitJobPage onJobSubmitted={navigateToStatus} />;
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-100 font-inter">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 shadow-lg">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Zap className="h-8 w-8 text-cyan-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">AllosKin</h1>
          </div>
          <Navbar setPage={setPage} currentPage={page} />
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow container mx-auto px-4 py-8">
        {renderPage()}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 text-sm text-center py-4 border-t border-gray-700">
        AllosKin Causal Analysis Pipeline
      </footer>
    </div>
  );
}

/*
================================================================================
Navigation Component
================================================================================
*/

const Navbar = ({ setPage, currentPage }) => {
  const navItems = [
    { name: 'submit', label: 'Submit Job', icon: <Upload className="h-4 w-4" /> },
    { name: 'results', label: 'View Results', icon: <Database className="h-4 w-4" /> },
    { name: 'health', label: 'System Health', icon: <Server className="h-4 w-4" /> },
  ];

  const getLinkClasses = (pageName) => {
    // Highlight 'status' or 'result_detail' as if they are part of 'results'
    const isResultsActive = currentPage === 'results' || currentPage === 'status' || currentPage === 'result_detail';
    
    let isActive = currentPage === pageName;
    if (pageName === 'results' && isResultsActive) {
      isActive = true;
    }
    if (pageName === 'submit' && isResultsActive) {
      isActive = false;
    }


    return `flex items-center space-x-2 px-3 py-2 rounded-md font-medium transition-colors ${
      isActive
        ? 'bg-cyan-600 text-white'
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`;
  };

  return (
    <nav className="flex space-x-2">
      {navItems.map((item) => (
        <button
          key={item.name}
          onClick={() => setPage(item.name)}
          className={getLinkClasses(item.name)}
        >
          {item.icon}
          <span>{item.label}</span>
        </button>
      ))}
      {/* Show a special 'Job Status' tab if we are polling a job */}
      {currentPage === 'status' && (
        <button
          onClick={() => setPage('status')}
          className={getLinkClasses('status')}
        >
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Job Status</span>
        </button>
      )}
    </nav>
  );
};

/*
================================================================================
Job Submission Page (Refactored)
================================================================================
*/

const SubmitJobPage = ({ onJobSubmitted }) => {
  const [analysisType, setAnalysisType] = useState('dynamic'); // 'static', 'qubo', 'dynamic'
  
  // State for file inputs (shared)
  const [files, setFiles] = useState({
    active_topo: null,
    active_traj: null,
    inactive_topo: null,
    inactive_traj: null,
    config: null,
  });
  
  // State for parameters
  const [teLag, setTeLag] = useState(10);
  const [targetSwitch, setTargetSwitch] = useState('res_50'); // Example
  
  // State for UI
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const { name, files } = e.target;
    setFiles((prev) => ({ ...prev, [name]: files[0] }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    // 1. Validate all files are present
    if (Object.values(files).some(file => !file)) {
      setError("Please upload all 5 required files.");
      setIsLoading(false);
      return;
    }

    // 2. Create FormData
    const formData = new FormData();
    // Append files (keys match FastAPI endpoint arguments)
    formData.append('active_topo', files.active_topo);
    formData.append('active_traj', files.active_traj);
    formData.append('inactive_topo', files.inactive_topo);
    formData.append('inactive_traj', files.inactive_traj);
    formData.append('config', files.config);

    // 3. Set endpoint and add specific params
    let endpoint = '';
    switch (analysisType) {
      case 'static':
        endpoint = '/api/v1/submit/static';
        // No extra params
        break;
      case 'qubo':
        endpoint = '/api/v1/submit/qubo';
        if (!targetSwitch) {
            setError("Target Switch residue is required for QUBO.");
            setIsLoading(false);
            return;
        }
        formData.append('target_switch', targetSwitch);
        break;
      case 'dynamic':
        endpoint = '/api/v1/submit/dynamic';
        formData.append('te_lag', teLag);
        break;
      default:
        setError("Invalid analysis type selected.");
        setIsLoading(false);
        return;
    }

    // 4. Make the API request
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Job submission failed.");
      }
      
      if (!data.job_id) {
          throw new Error("Job submission failed to return a job ID.");
      }

      // 5. On success, navigate to the status page
      onJobSubmitted(data.job_id); // Pass the RQ job_id for polling

    } catch (err) {
      const errorMsg = typeof err.message === 'object' ? JSON.stringify(err.message) : err.message;
      setError(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-white">Run New Analysis</h2>
      
      {/* Analysis Type Tabs */}
      <div className="flex space-x-1 rounded-lg bg-gray-800 p-1 mb-6">
        <TabButton 
          icon={<Brain />} 
          label="Static Reporters" 
          isActive={analysisType === 'static'} 
          onClick={() => setAnalysisType('static')} 
        />
        <TabButton 
          icon={<Sliders />} 
          label="QUBO Optimal Set" 
          isActive={analysisType === 'qubo'} 
          onClick={() => setAnalysisType('qubo')} 
        />
        <TabButton 
          icon={<Zap />} 
          label="Dynamic TE" 
          isActive={analysisType === 'dynamic'} 
          onClick={() => setAnalysisType('dynamic')} 
        />
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Inputs (Shared) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FileInputGroup
            title="Active State"
            files={[
              { name: 'active_topo', label: 'Topology (PDB, GRO, ...)' },
              { name: 'active_traj', label: 'Trajectory (XTC, TRR, ...)' },
            ]}
            fileState={files}
            onChange={handleFileChange}
          />
          <FileInputGroup
            title="Inactive State"
            files={[
              { name: 'inactive_topo', label: 'Topology (PDB, GRO, ...)' },
              { name: 'inactive_traj', label: 'Trajectory (XTC, TRR, ...)' },
            ]}
            fileState={files}
            onChange={handleFileChange}
          />
        </div>

        {/* Config and Parameters (Dynamic) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h3 className="text-lg font-semibold mb-3 text-cyan-400">Configuration</h3>
            <FileDropzone
              name="config"
              label="Residue Config (config.yml)"
              file={files.config}
              onChange={handleFileChange}
            />
          </div>
          
          {/* Conditional Parameters */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h3 className="text-lg font-semibold mb-3 text-cyan-400">Parameters</h3>
            
            {analysisType === 'static' && (
              <p className="text-sm text-gray-400">No additional parameters required for Static analysis.</p>
            )}

            {analysisType === 'qubo' && (
              <div>
                <label htmlFor="target_switch" className="block text-sm font-medium text-gray-300 mb-1">
                  Target Switch (e.g., res_50)
                </label>
                <input
                  type="text"
                  id="target_switch"
                  value={targetSwitch}
                  onChange={(e) => setTargetSwitch(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="Enter residue key (e.g., res_50)"
                />
              </div>
            )}
            
            {analysisType === 'dynamic' && (
              <div>
                <label htmlFor="te_lag" className="block text-sm font-medium text-gray-300 mb-1">
                  TE Lag Time (frames)
                </label>
                <input
                  type="number"
                  id="te_lag"
                  value={teLag}
                  onChange={(e) => setTeLag(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500"
                />
              </div>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && <ErrorDisplay error={error} />}

        {/* Submit Button */}
        <SubmitButton isLoading={isLoading} />
      </form>
    </div>
  );
};

// --- Submit Page Helpers ---

const TabButton = ({ icon, label, isActive, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    className={`w-full flex justify-center items-center space-x-2 px-3 py-3 font-medium text-sm rounded-md transition-colors ${
      isActive
        ? 'bg-cyan-600 text-white'
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`}
  >
    {icon}
    <span>{label}</span>
  </button>
);

const FileInputGroup = ({ title, files, fileState, onChange }) => (
  <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-4">
    <h3 className="text-lg font-semibold text-cyan-400">{title}</h3>
    {files.map((file) => (
      <FileDropzone
        key={file.name}
        name={file.name}
        label={file.label}
        file={fileState[file.name]}
        onChange={onChange}
      />
    ))}
  </div>
);

const FileDropzone = ({ name, label, file, onChange }) => (
  <div>
    <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
    <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-600 border-dashed rounded-md">
      <div className="space-y-1 text-center">
        <svg className="mx-auto h-10 w-10 text-gray-500" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
          <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="flex text-sm text-gray-400">
          <label htmlFor={name} className="relative cursor-pointer bg-gray-900 rounded-md font-medium text-cyan-500 hover:text-cyan-400 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-offset-gray-900 focus-within:ring-cyan-500">
            <span>Upload a file</span>
            <input id={name} name={name} type="file" className="sr-only" onChange={onChange} />
          </label>
          <p className="pl-1">or drag and drop</p>
        </div>
        {file ? (
          <p className="text-xs text-green-400 truncate max-w-xs">{file.name}</p>
        ) : (
          <p className="text-xs text-gray-500">Up to 500MB</p>
        )}
      </div>
    </div>
  </div>
);

const ErrorDisplay = ({ error }) => (
  <div className="bg-red-900 border border-red-700 text-red-100 p-3 rounded-md flex items-center space-x-2">
    <XCircle className="h-5 w-5" />
    <span>{error}</span>
  </div>
);

const SubmitButton = ({ isLoading }) => (
  <button
    type="submit"
    disabled={isLoading}
    className="w-full flex justify-center items-center space-x-2 bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
  >
    {isLoading ? (
      <Loader2 className="h-6 w-6 animate-spin" />
    ) : (
      <Upload className="h-6 w-6" />
    )}
    <span>{isLoading ? 'Submitting...' : 'Submit Job'}</span>
  </button>
);


/*
================================================================================
Job Status Page
================================================================================
*/

const JobStatusPage = ({ jobId, onNavigateToResults }) => {
  const [status, setStatus] = useState(null); // The full API response
  const [error, setError] = useState(null);
  const pollingInterval = useRef(null);

  useEffect(() => {
    if (!jobId) {
      setError("No Job ID specified. Redirecting to Results.");
      const timer = setTimeout(() => onNavigateToResults(), 2000);
      return () => clearTimeout(timer);
    }

    const pollStatus = async () => {
      try {
        const response = await fetch(`/api/v1/job/status/${jobId}`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Failed to fetch job status.");
        }

        setStatus(data);

        if (data.status === 'finished' || data.status === 'failed') {
          clearInterval(pollingInterval.current);
        }
      } catch (err) {
        setError(err.message);
        clearInterval(pollingInterval.current);
      }
    };

    pollStatus();
    pollingInterval.current = setInterval(pollStatus, 3000);
    return () => clearInterval(pollingInterval.current);
  }, [jobId, onNavigateToResults]);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto text-center">
        <h2 className="text-3xl font-bold mb-4 text-red-500">Error</h2>
        <p className="text-gray-300">{error}</p>
      </div>
    );
  }

  // Handle the different states from the API response
  const jobStatus = status?.status || 'queued';
  const metaStatus = status?.meta?.status || (jobStatus === 'queued' ? 'Waiting in queue...' : 'Initializing...');
  const progress = status?.meta?.progress || (jobStatus === 'queued' ? 0 : 5);
  const resultPayload = status?.result; // This is the full result payload from the worker
  const jobUUID = resultPayload?.job_id; // The persistent UUID

  if (jobStatus === 'finished' && resultPayload) {
    return (
      <StatusDisplay
        icon={<CheckCircle className="h-16 w-16 text-green-500" />}
        title="Analysis Complete"
        jobId={jobId}
        message="Your results are ready."
      >
        <div className="mt-6 text-left">
          <h3 className="text-lg font-semibold text-green-400 mb-2">Results:</h3>
          <pre className="bg-gray-900 p-4 rounded-md text-gray-200 text-xs overflow-auto">
            {JSON.stringify(resultPayload.results || resultPayload, null, 2)}
          </pre>
        </div>
        <button
          onClick={() => onNavigateToResults()}
          className="mt-6 w-full flex justify-center items-center space-x-2 bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
        >
          <span>View All Results</span>
        </button>
      </StatusDisplay>
    );
  }

  if (jobStatus === 'failed') {
    const errorMsg = resultPayload?.error || "An unknown error occurred.";
    return (
      <StatusDisplay
        icon={<XCircle className="h-16 w-16 text-red-500" />}
        title="Analysis Failed"
        jobId={jobId}
        message="An error occurred during processing."
      >
        <div className="mt-6 text-left">
          <h3 className="text-lg font-semibold text-red-400 mb-2">Error Details:</h3>
          <pre className="bg-gray-900 p-4 rounded-md text-red-300 text-xs overflow-auto">
            {errorMsg}
          </pre>
        </div>
      </StatusDisplay>
    );
  }

  // Default: In Progress (queued, started)
  return (
    <StatusDisplay
      icon={<Loader2 className="h-16 w-16 text-cyan-400 animate-spin" />}
      title="Analysis in Progress..."
      jobId={jobId}
      message={metaStatus}
    >
      <div className="w-full bg-gray-700 rounded-full h-2.5 mt-6">
        <div 
          className="bg-cyan-500 h-2.5 rounded-full transition-all" 
          style={{ width: `${progress || 0}%` }}
        ></div>
      </div>
    </StatusDisplay>
  );
};

/**
 * Helper component for the status display card.
 */
const StatusDisplay = ({ icon, title, message, jobId, children }) => (
  <div className="max-w-3xl mx-auto bg-gray-800 rounded-lg border border-gray-700 shadow-xl p-8 text-center">
    <div className="flex justify-center mb-6">{icon}</div>
    <h2 className="text-3xl font-bold mb-3 text-white">{title}</h2>
    <p className="text-gray-300 mb-6">{message}</p>
    <div className="bg-gray-900 p-3 rounded-md text-sm text-gray-400 font-mono">
      Polling Job ID: {jobId}
    </div>
    {children}
  </div>
);


/*
================================================================================
NEW: Results List Page
================================================================================
*/

const ResultsListPage = ({ onSelectResult }) => {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/v1/results');
        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || "Failed to fetch results.");
        }
        const data = await response.json();
        setResults(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchResults();
  }, []);

  const groupResults = (results) => {
    if (!results) return {};
    return results.reduce((acc, result) => {
      const type = result.analysis_type || 'unknown';
      if (!acc[type]) {
        acc[type] = [];
      }
      acc[type].push(result);
      return acc;
    }, {});
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="h-12 w-12 text-cyan-400 animate-spin" />
      </div>
    );
  }

  if (error) {
    return <ErrorDisplay error={error} />;
  }
  
  const groupedResults = groupResults(results);

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-white">Analysis Results</h2>
      
      {Object.keys(groupedResults).length === 0 && (
        <div className="text-center text-gray-400 bg-gray-800 p-8 rounded-lg border border-gray-700">
          <FileText className="h-12 w-12 mx-auto mb-4 text-gray-500" />
          <h3 className="text-xl font-semibold text-white">No Results Found</h3>
          <p className="mt-2">Run a new job from the "Submit Job" page to see results here.</p>
        </div>
      )}

      <div className="space-y-8">
        {Object.entries(groupedResults).map(([type, items]) => (
          <div key={type}>
            <h3 className="text-2xl font-semibold text-cyan-400 mb-4 capitalize">{type} Analysis</h3>
            <div className="bg-gray-800 rounded-lg border border-gray-700 shadow-lg">
              <ul className="divide-y divide-gray-700">
                {items.map((item) => (
                  <ResultItem key={item.job_id} item={item} onSelectResult={onSelectResult} />
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const ResultItem = ({ item, onSelectResult }) => {
  const isFailed = item.status === 'failed';
  const icon = isFailed ? 
    <XCircle className="h-5 w-5 text-red-500" /> : 
    <CheckCircle className="h-5 w-5 text-green-500" />;
  
  const formattedDate = item.completed_at ? 
    new Date(item.completed_at).toLocaleString() : 
    (item.created_at ? `Started ${new Date(item.created_at).toLocaleString()}` : 'No date');

  return (
    <li
      className={`flex items-center justify-between p-4 ${!isFailed ? 'hover:bg-gray-700 cursor-pointer' : 'opacity-60'} transition-colors`}
      onClick={() => !isFailed && onSelectResult(item.job_id)}
    >
      <div className="flex items-center space-x-3">
        {icon}
        <div>
          <p className="text-sm font-medium text-white">{item.job_id}</p>
          <p className="text-sm text-gray-400">{formattedDate}</p>
        </div>
      </div>
      {!isFailed && <ChevronRight className="h-5 w-5 text-gray-500" />}
    </li>
  );
};


/*
================================================================================
NEW: Result Detail Page
================================================================================
*/

const ResultDetailPage = ({ resultId, onBack }) => {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!resultId) {
      setError("No result ID specified.");
      setIsLoading(false);
      return;
    }
    
    const fetchResult = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`/api/v1/results/${resultId}`);
        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || "Failed to fetch result.");
        }
        const data = await response.json();
        setResult(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchResult();
  }, [resultId]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="h-12 w-12 text-cyan-400 animate-spin" />
      </div>
    );
  }

  if (error) {
    return <ErrorDisplay error={error} />;
  }

  if (!result) {
    return <ErrorDisplay error="Result data could not be loaded." />;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <button
        onClick={onBack}
        className="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 mb-4"
      >
        <ArrowLeft className="h-5 w-5" />
        <span>Back to Results List</span>
      </button>

      <div className="bg-gray-800 rounded-lg border border-gray-700 shadow-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-2">Analysis Result</h2>
        <p className="text-sm text-gray-400 mb-1">
          <span className="font-semibold">Job ID:</span> {result.job_id}
        </p>
        <p className="text-sm text-gray-400 mb-4">
          <span className="font-semibold">Type:</span> <span className="capitalize">{result.analysis_type}</span>
        </p>

        {/* As requested, just print the results file */}
        <h3 className="text-lg font-semibold text-cyan-400 mb-2">Raw Result Data</h3>
        <pre className="bg-gray-900 p-4 rounded-md text-gray-200 text-xs overflow-auto">
          {JSON.stringify(result, null, 2)}
        </pre>
      </div>
    </div>
  );
};


/*
================================================================================
System Health Page (Unchanged)
================================================================================
*/

/**
 * Page component for checking system health.
 */
const HealthCheckPage = () => {
  const [healthReport, setHealthReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const runHealthCheck = async () => {
    setIsLoading(true);
    setError(null);
    setHealthReport(null);

    try {
      // Use relative URL
      const response = await fetch('/api/v1/health/check');
      const data = await response.json();

      if (!response.ok) {
        const detail = data.detail || data;
        setError("System status is not fully OK. See details below.");
        setHealthReport(detail);
      } else {
        setHealthReport(data);
      }
    } catch (err) {
      setError("An unknown error occurred while contacting the server.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-white">System Health Check</h2>
      <p className="mb-6 text-gray-300">
        Run an end-to-end test of the system, from the API to Redis
        to the background Worker.
      </p>
      
      <button
        onClick={runHealthCheck}
        disabled={isLoading}
        className="flex items-center justify-center space-x-2 bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed mb-8"
      >
        {isLoading ? (
          <Loader2 className="h-5 w-5 animate-spin" />
        ) : (
          <Activity className="h-5 w-5" />
        )}
        <span>{isLoading ? 'Running Check...' : 'Run E2E Health Check'}</span>
      </button>

      {error && !healthReport && (
        <ErrorDisplay error={error} />
      )}

      {healthReport && (
        <div className="space-y-4">
          <HealthStatusCard title="API Status" status={healthReport.api_status} />
          <HealthStatusCard 
            title="Redis Status" 
            status={healthReport.redis_status?.status} 
            details={healthReport.redis_status} 
          />
          <HealthStatusCard 
            title="Worker Status" 
            status={healthReport.worker_status?.status} 
            details={healthReport.worker_status} 
          />
        </div>
      )}
    </div>
  );
};

/**
 * Helper component to display a single health status card.
 */
const HealthStatusCard = ({ title, status, details }) => {
  const isOk = status === 'ok';
  const displayStatus = status || 'unknown';

  const getStatusIcon = () => {
    if (isOk) {
      return <CheckCircle className="h-8 w-8 text-green-500" />;
    }
    return <AlertTriangle className="h-8 w-8 text-yellow-500" />;
  };

  return (
    <div className={`bg-gray-800 rounded-lg border ${isOk ? 'border-green-700' : 'border-yellow-700'} shadow-lg p-6`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon()}
          <h3 className="text-xl font-semibold text-white">{title}</h3>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm font-bold ${isOk ? 'bg-green-800 text-green-100' : 'bg-yellow-800 text-yellow-100'}`}>
          {displayStatus}
        </span>
      </div>
      {details && details.status !== 'ok' && (
        <pre className="mt-4 bg-gray-900 p-4 rounded-md text-red-300 text-xs overflow-auto">
          {JSON.stringify(details, null, 2)}
        </pre>
      )}
    </div>
  );
};