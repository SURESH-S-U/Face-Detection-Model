import React, { useState, useRef } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { NAvigation } from '../components/Layout';

interface AnalysisResult {
  known_count: number;
  unknown_count: number;
  known_faces?: { [key: string]: number };
  total_processed_frames?: number;
  total_frames?: number;
  results?: Array<{
    face_id: string;
    name: string;
    confidence: number;
    timestamp: string;
    face_image: string;
  }>;
}

const VideoAnalysisComponent: React.FC = () => {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<'known' | 'unknown'>('known');
  const [selectedFace, setSelectedFace] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#A4DE6C', '#D0ED57', '#FFC658', '#8DD1E1'];

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setAnalysisResult(null);
      setProgress(0);
      setSelectedFace(null);
    }
  };

  const handleAnalyzeClick = async () => {
    if (!videoSrc || !fileInputRef.current?.files?.[0]) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setProgress(0);
    setSelectedFace(null);

    const file = fileInputRef.current.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const progressInterval = setInterval(() => {
        setProgress(prev => (prev < 95 ? prev + 5 : prev));
      }, 500);

      const response = await axios.post<AnalysisResult>('http://127.0.0.1:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      clearInterval(progressInterval);
      setProgress(100);
      setAnalysisResult(response.data);
    } catch (error) {
      console.error('Error analyzing video:', error);
      alert(`Error analyzing video: ${error.response?.data?.error || 'Server error'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const prepareChartData = () => {
    if (!analysisResult) return { summary: [], detailed: [] };

    const { known_count, unknown_count, known_faces } = analysisResult;

    const summaryData = [
      { name: 'Known', value: known_count },
      { name: 'Unknown', value: unknown_count },
    ];

    const detailedData = known_faces ? Object.entries(known_faces).map(([name, count]) => ({
      name,
      value: count,
    })) : [];

    return { summary: summaryData, detailed: detailedData };
  };

  const chartData = prepareChartData();

  const renderFaceList = () => {
    if (!analysisResult || !analysisResult.results) return null;

    const filteredFaces = analysisResult.results.filter(face =>
      activeTab === 'known' ? face.name !== 'Unknown' : face.name === 'Unknown'
    );

    return (
      <div className="mt-4 flex flex-col space-y-2">
        
        <div className="flex border-b">
          <button
            className={`px-4 py-2 font-medium ${activeTab === 'known' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-500'}`}
            onClick={() => setActiveTab('known')}
          >
            Known Faces ({analysisResult.known_count})
          </button>
          <button
            className={`px-4 py-2 font-medium ${activeTab === 'unknown' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-500'}`}
            onClick={() => setActiveTab('unknown')}
          >
            Unknown Faces ({analysisResult.unknown_count})
          </button>
        </div>

        <div className="overflow-y-auto max-h-96 mt-2">
          {filteredFaces.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {filteredFaces.map((face) => (
                <div
                  key={face.face_id}
                  className={`p-2 border rounded cursor-pointer hover:bg-gray-100 transition ${selectedFace === face.face_id ? 'ring-2 ring-blue-500 bg-blue-50' : ''}`}
                  onClick={() => setSelectedFace(selectedFace === face.face_id ? null : face.face_id)}
                >
                  <img
                    src={`data:image/jpeg;base64,${face.face_image}`}
                    alt={face.name}
                    className="w-full h-32 object-cover mb-1"
                  />
                  <div className="text-xs font-medium truncate">
                    {face.name === 'Unknown' ? 'Unknown Person' : face.name}
                  </div>
                  <div className="text-xs text-gray-500">
                    Confidence: {(face.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 text-center text-gray-500">
              No {activeTab} faces detected in the video.
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 flex flex-col w-full h-full bg-gray-100">
      <NAvigation/>
      <div className="ml-[80px]">
      <header className="w-full bg-gray-900 text-white p-4 shadow-md">
        <h1 className="text-xl sm:text-2xl font-bold">Video Face Recognition</h1>
      </header>

      <main className="flex flex-col lg:flex-row flex-1 p-4 gap-4 overflow-auto">
        {/* Left side - Video upload and preview */}
        <div className="w-full lg:w-1/3 bg-white rounded-lg shadow-md p-4 flex flex-col">
          <h2 className="text-xl font-semibold mb-4">Upload Video</h2>

          <div
            className="flex flex-col items-center justify-center flex-1 border-2 border-dashed border-gray-300 rounded-lg p-4"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const file = e.dataTransfer.files?.[0];
              if (file && file.type.startsWith('video/')) {
                const url = URL.createObjectURL(file);
                setVideoSrc(url);
                setAnalysisResult(null);
                setProgress(0);
                setSelectedFace(null);

                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                if (fileInputRef.current) {
                  fileInputRef.current.files = dataTransfer.files;
                }
              }
            }}
          >
            {videoSrc ? (
              <video src={videoSrc} controls className="w-full max-h-64 mb-4" />
            ) : (
              <div className="text-center text-gray-500 py-12">
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
                <p className="mt-1">Drag and drop a video file here or click to upload</p>
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
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
            >
              Select Video
            </button>
          </div>

          <div className="mt-4">
            {videoSrc && (
              <button
                onClick={handleAnalyzeClick}
                disabled={isAnalyzing}
                className={`w-full py-2 text-white rounded transition ${
                  isAnalyzing ? 'bg-gray-400' : 'bg-green-500 hover:bg-green-600'
                }`}
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
              </button>
            )}

            {isAnalyzing && (
              <div className="w-full mt-4">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="text-center text-xs mt-1 text-gray-500">Processing video...</p>
              </div>
            )}
          </div>
        </div>

        {/* Middle - Charts and graphs */}
        <div className="w-full lg:w-1/3 bg-white rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>

          {analysisResult ? (
            <div className="flex flex-col space-y-6">
              <div>
                <h3 className="text-lg font-medium mb-2">Face Recognition Summary</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={chartData.summary}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {chartData.summary.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value} occurrences`, 'Count']} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {chartData.detailed.length > 0 && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Known Faces Breakdown</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={chartData.detailed}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="value" name="Occurrences" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              No analysis results to display. Upload and analyze a video first.
            </div>
          )}
        </div>

        {/* Right side - Detected Faces */}
        <div className="w-full lg:w-1/3 bg-white rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4">Detected Faces</h2>
          {analysisResult ? renderFaceList() : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              No faces detected yet. Analyze a video to see results.
            </div>
          )}
        </div>
      </main>
      </div>
    </div>
  );
};

export default VideoAnalysisComponent;