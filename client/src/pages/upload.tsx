import React, { useState, useRef } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface AnalysisResult {
  known_count: number;
  unknown_count: number;
  known_faces?: { [key: string]: number };
  results?: Array<{
    bbox: number[];
    status: string;
    confidence: number;
    name: string;
    timestamp: string;
  }>;
}

interface RegistrationStatus {
  success: boolean;
  message: string;
}

const VideoAnalysisComponent: React.FC = () => {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [registerName, setRegisterName] = useState<string>('');
  const [registerImage, setRegisterImage] = useState<string | null>(null);
  const [registrationStatus, setRegistrationStatus] = useState<RegistrationStatus | null>(null);
  const [showRegisterPanel, setShowRegisterPanel] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const registerImageInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setAnalysisResult(null);
    }
  };

  const handleAnalyzeClick = async () => {
    if (!videoSrc || !fileInputRef.current?.files?.[0]) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);

    const file = fileInputRef.current.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<AnalysisResult>('http://127.0.0.1:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setAnalysisResult(response.data);
    } catch (error) {
      console.error('Error analyzing video:', error);
      if (error.response) {
        console.error('Backend response:', error.response.data);
        alert(`Error analyzing video: ${error.response.data.error || 'Unknown error'}`);
      } else if (error.request) {
        console.error('No response received:', error.request);
        alert('No response received from the server. Please check your connection.');
      } else {
        console.error('Request setup error:', error.message);
        alert('Error setting up the request. Please try again.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRegisterImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setRegisterImage(url);
      setRegistrationStatus(null);
    }
  };

  const handleRegisterFace = async () => {
    if (!registerImage || !registerName.trim() || !registerImageInputRef.current?.files?.[0]) {
      alert('Please select an image and enter a name');
      return;
    }

    const file = registerImageInputRef.current.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', registerName.trim());

    try {
      const response = await axios.post<{ message: string }>('http://127.0.0.1:5000/register', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setRegistrationStatus({
        success: true,
        message: response.data.message,
      });

      // Clear form
      setRegisterName('');
      setRegisterImage(null);
      if (registerImageInputRef.current) {
        registerImageInputRef.current.value = '';
      }
    } catch (error) {
      console.error('Error registering face:', error);
      setRegistrationStatus({
        success: false,
        message: error.response?.data?.error || 'Error registering face',
      });
    }
  };

  const handleResetDatabase = async () => {
    if (!confirm('Are you sure you want to reset the face database? This will delete all registered faces.')) {
      return;
    }

    try {
      const response = await axios.post<{ message: string }>('http://127.0.0.1:5000/reset');
      alert(response.data.message);
      setRegistrationStatus(null);
    } catch (error) {
      console.error('Error resetting database:', error);
      alert('Error resetting database');
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const triggerRegisterImageInput = () => {
    registerImageInputRef.current?.click();
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setAnalysisResult(null);

      // Update the file input value
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      if (fileInputRef.current) {
        fileInputRef.current.files = dataTransfer.files;
      }
    }
  };

  const prepareChartData = () => {
    if (!analysisResult) return { basic: [] };

    const { known_count, unknown_count } = analysisResult;

    // Basic chart data
    const chartData = [
      {
        name: 'Recognition Results',
        known: known_count,
        unknown: unknown_count,
      }
    ];

    // Add individual known faces if available
    if (analysisResult.known_faces && Object.keys(analysisResult.known_faces).length > 0) {
      const knownFacesData = Object.entries(analysisResult.known_faces).map(([name, count]) => ({
        name,
        count
      }));

      return {
        basic: chartData,
        detailed: knownFacesData
      };
    }

    return { basic: chartData };
  };

  const chartData = analysisResult ? prepareChartData() : { basic: [] };

  const toggleRegisterPanel = () => {
    setShowRegisterPanel(!showRegisterPanel);
    setRegistrationStatus(null);
  };

  const clearResults = () => {
    setAnalysisResult(null);
    setVideoSrc(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="fixed inset-0 flex flex-col w-full h-full bg-gray-100">
      <header className="w-full bg-gray-900 text-white p-4 shadow-md flex justify-between items-center">
        <h1 className="text-xl sm:text-2xl font-bold">Video People Analysis</h1>
        <div className="flex space-x-2">
          <button
            onClick={toggleRegisterPanel}
            className="px-3 py-1 text-sm bg-blue-500 hover:bg-blue-600 rounded text-white transition"
          >
            {showRegisterPanel ? 'Back to Analysis' : 'Register Face'}
          </button>
          <button
            onClick={handleResetDatabase}
            className="px-3 py-1 text-sm bg-red-500 hover:bg-red-600 rounded text-white transition"
          >
            Reset Database
          </button>
        </div>
      </header>

      {showRegisterPanel ? (
        <main className="flex flex-col p-4 flex-1 overflow-auto">
          <div className="bg-white rounded-lg shadow-md p-4 max-w-xl mx-auto w-full">
            <h2 className="text-xl font-semibold mb-4">Register New Face</h2>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                type="text"
                value={registerName}
                onChange={(e) => setRegisterName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter person's name"
              />
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Face Image</label>
              <div
                className="border-2 border-dashed border-gray-300 rounded-lg p-4 flex flex-col items-center justify-center cursor-pointer"
                onClick={triggerRegisterImageInput}
              >
                {registerImage ? (
                  <img
                    src={registerImage}
                    alt="Face to register"
                    className="max-h-64 object-contain mb-2"
                  />
                ) : (
                  <>
                    <svg
                      className="mx-auto h-12 w-12 text-gray-400"
                      stroke="currentColor"
                      fill="none"
                      viewBox="0 0 48 48"
                      aria-hidden="true"
                    >
                      <path
                        d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    <p className="mt-1 text-sm text-gray-500">Click to select or drag and drop a face image</p>
                  </>
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleRegisterImageSelect}
                  className="hidden"
                  ref={registerImageInputRef}
                />
              </div>
            </div>

            <button
              onClick={handleRegisterFace}
              disabled={!registerImage || !registerName.trim()}
              className={`w-full py-2 rounded-md text-white ${
                !registerImage || !registerName.trim()
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-500 hover:bg-green-600'
              }`}
            >
              Register Face
            </button>

            {registrationStatus && (
              <div className={`mt-4 p-3 rounded-md ${
                registrationStatus.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {registrationStatus.message}
              </div>
            )}
          </div>
        </main>
      ) : (
        <main className="flex flex-col md:flex-row flex-1 p-2 sm:p-4 gap-4 overflow-auto w-full">
          {/* Left side - Video upload and preview */}
          <div className="w-full md:w-1/2 bg-white rounded-lg shadow-md p-2 sm:p-4 flex flex-col">
            <h2 className="text-lg sm:text-xl font-semibold mb-2 sm:mb-4">Upload Video</h2>

            <div
              className="flex flex-col items-center justify-center flex-1 border-2 border-dashed border-gray-300 rounded-lg p-2 sm:p-4"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              {videoSrc ? (
                <video
                  src={videoSrc}
                  controls
                  className="w-full max-h-48 sm:max-h-64 lg:max-h-80 mb-4"
                />
              ) : (
                <div className="text-center text-gray-500">
                  <svg
                    className="mx-auto h-8 w-8 sm:h-12 sm:w-12 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                    aria-hidden="true"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <p className="mt-1 text-sm sm:text-base">Drag and drop a video file here or click to upload</p>
                </div>
              )}

              <input
                type="file"
                accept="video/*"
                onChange={handleFileUpload}
                className="hidden"
                ref={fileInputRef}
              />

              <button
                onClick={triggerFileInput}
                className="mt-4 px-3 py-1 sm:px-4 sm:py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition text-sm sm:text-base"
              >
                Select Video
              </button>
            </div>

            <div className="flex space-x-2 mt-4">
              {videoSrc && (
                <button
                  onClick={handleAnalyzeClick}
                  disabled={isAnalyzing}
                  className={`flex-1 px-3 py-1 sm:px-4 sm:py-2 text-white rounded transition text-sm sm:text-base ${
                    isAnalyzing ? 'bg-gray-400' : 'bg-green-500 hover:bg-green-600'
                  }`}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
                </button>
              )}

              {analysisResult && (
                <button
                  onClick={clearResults}
                  className="px-3 py-1 sm:px-4 sm:py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition text-sm sm:text-base"
                >
                  Clear Results
                </button>
              )}
            </div>
          </div>

          {/* Right side - Analysis results */}
          <div className="w-full md:w-1/2 bg-white rounded-lg shadow-md p-2 sm:p-4">
            <h2 className="text-lg sm:text-xl font-semibold mb-2 sm:mb-4">Analysis Results</h2>

            {analysisResult ? (
              <div className="flex flex-col space-y-4">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData.basic}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="known" fill="#8884d8" name="Known Faces" />
                    <Bar dataKey="unknown" fill="#82ca9d" name="Unknown Faces" />
                  </BarChart>
                </ResponsiveContainer>

                {chartData.detailed && (
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Known Faces Breakdown</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={chartData.detailed}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="count" fill="#8884d8" name="Count" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-gray-500">No analysis results to display. Upload and analyze a video first.</p>
            )}
          </div>
        </main>
      )}
    </div>
  );
};

export default VideoAnalysisComponent;