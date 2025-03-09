import React, { useState, useRef } from 'react';
import { NAvigation } from '../components/Layout';


interface PersonData {
  id: number;
  status: 'known' | 'unknown';
  confidence: number;
  timestamp: number;
}

interface AnalysisResult {
  totalCount: number;
  knownCount: number;
  unknownCount: number;
  people: PersonData[];
}

const VideoAnalysisPage: React.FC = () => {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setAnalysisResult(null);
    }
  };

  const handleAnalyzeClick = () => {
    setIsAnalyzing(true);
    
    // Simulate video analysis with a timeout
    setTimeout(() => {
      // Mock analysis result
      const mockResult: AnalysisResult = {
        totalCount: 5,
        knownCount: 3,
        unknownCount: 2,
        people: [
          { id: 1, status: 'known', confidence: 0.92, timestamp: 1.2 },
          { id: 2, status: 'known', confidence: 0.88, timestamp: 2.5 },
          { id: 3, status: 'unknown', confidence: 0.76, timestamp: 3.7 },
          { id: 4, status: 'known', confidence: 0.95, timestamp: 5.1 },
          { id: 5, status: 'unknown', confidence: 0.64, timestamp: 8.3 },
        ]
      };
      
      setAnalysisResult(mockResult);
      setIsAnalyzing(false);
    }, 2000);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div>
      <NAvigation/>
    <div className="flex flex-col min-h-screen bg-gray-100 ml-[100px]">
      <header className="bg-gray-900 text-white p-4 shadow-md ml-[-30px]">
        <h1 className="text-2xl font-bold">Video People Analysis</h1>
      </header>
      
      <main className="flex flex-1 p-4 gap-4">
        {/* Left side - Video upload and preview */}
        <div className="w-1/2 bg-white rounded-lg shadow-md p-4 flex flex-col">
          <h2 className="text-xl font-semibold mb-4">Upload Video</h2>
          
          <div className="flex flex-col items-center justify-center flex-1 border-2 border-dashed border-gray-300 rounded-lg p-4">
            {videoSrc ? (
              <video 
                ref={videoRef}
                src={videoSrc} 
                controls 
                className="max-w-full max-h-64 mb-4"
              />
            ) : (
              <div className="text-center text-gray-500">
                <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                  <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
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
              onClick={triggerFileInput}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
            >
              Select Video
            </button>
          </div>
          
          {videoSrc && (
            <button 
              onClick={handleAnalyzeClick}
              disabled={isAnalyzing}
              className={`mt-4 px-4 py-2 text-white rounded transition ${isAnalyzing ? 'bg-gray-400' : 'bg-green-500 hover:bg-green-600'}`}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
            </button>
          )}
        </div>
        
        {/* Right side - Analysis results */}
        <div className="w-1/2 bg-white rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
          
          {isAnalyzing ? (
            <div className="flex flex-col items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
              <p className="mt-4 text-gray-600">Analyzing video...</p>
            </div>
          ) : analysisResult ? (
            <div>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-blue-100 p-4 rounded-lg text-center">
                  <p className="text-sm text-blue-500">Total People</p>
                  <p className="text-3xl font-bold text-blue-700">{analysisResult.totalCount}</p>
                </div>
                <div className="bg-green-100 p-4 rounded-lg text-center">
                  <p className="text-sm text-green-500">Known People</p>
                  <p className="text-3xl font-bold text-green-700">{analysisResult.knownCount}</p>
                </div>
                <div className="bg-orange-100 p-4 rounded-lg text-center">
                  <p className="text-sm text-orange-500">Unknown People</p>
                  <p className="text-3xl font-bold text-orange-700">{analysisResult.unknownCount}</p>
                </div>
              </div>
              
              <h3 className="font-medium mb-2">People Details</h3>
              <div className="overflow-y-auto max-h-64">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {analysisResult.people.map((person) => (
                      <tr key={person.id}>
                        <td className="px-6 py-4 whitespace-nowrap">{person.id}</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            person.status === 'known' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {person.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">{(person.confidence * 100).toFixed(1)}%</td>
                        <td className="px-6 py-4 whitespace-nowrap">{person.timestamp}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
              <svg className="h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="mt-2">Upload and analyze a video to see results</p>
            </div>
          )}
        </div>
      </main>
    </div>
  </div>
  );
};

export default VideoAnalysisPage;