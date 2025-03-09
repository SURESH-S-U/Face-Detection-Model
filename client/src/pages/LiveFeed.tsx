import React, { useState, useEffect, useRef } from 'react';
import { Users, AlertCircle, Power, RefreshCw, Upload } from 'lucide-react';
import { cn } from '../lib/utils';

// Define available cameras
const cameras = [
  { id: 0, name: 'Main Camera' },
  // Add more cameras as needed based on your camera_urls in backend
];

export default function FaceRecognitionDashboard() {
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [isOn, setIsOn] = useState(true);
  const [knownUsers, setKnownUsers] = useState([]);
  const [unknownUsers, setUnknownUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [folderPath, setFolderPath] = useState('');
  const [processingFolder, setProcessingFolder] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [videoFeed, setVideoFeed] = useState(null);
  
  // Reference to set interval for video feed updates
  const videoIntervalRef = useRef(null);

  // Function to fetch users data from backend API
  const fetchUsersData = async () => {
    if (!isOn) return;
    
    setIsLoading(true);
    try {
      // Call the backend API endpoint that connects to MongoDB
      const response = await fetch(`http://localhost:8001/api/recognition?cameraId=${selectedCamera}`);
      
      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.status}`);
      }
      
      const data = await response.json();
      
      setKnownUsers(data.knownUsers || []);
      setUnknownUsers(data.unknownUsers || []);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching user data:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Function to fetch video feed frame
  const fetchVideoFrame = async () => {
    if (!isOn) return;
    
    try {
      const response = await fetch(`http://localhost:8001/api/video-frame?cameraId=${selectedCamera}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch video frame: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.frame) {
        setVideoFeed(data.frame);
      }
    } catch (error) {
      console.error('Error fetching video frame:', error);
    }
  };

  // Function to process folder
  const processFolder = async () => {
    if (!folderPath) {
      alert("Please enter a folder path");
      return;
    }
    
    setProcessingFolder(true);
    try {
      const response = await fetch(
        `http://localhost:8001/detected-faces-from-folder?folder_path=${encodeURIComponent(folderPath)}`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to process folder: ${response.status}`);
      }
      
      const data = await response.json();
      alert(`Processed ${data.length} faces from folder`);
      
      // Refresh the data after processing
      fetchUsersData();
    } catch (error) {
      console.error('Error processing folder:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setProcessingFolder(false);
    }
  };
  
  // Function to clear detected faces
  const clearDetectedFaces = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/clear-detections', {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to clear detections: ${response.status}`);
      }
      
      // Clear the local state
      setKnownUsers([]);
      setUnknownUsers([]);
      setLastUpdated(new Date());
      
    } catch (error) {
      console.error('Error clearing detections:', error);
      alert(`Error: ${error.message}`);
    }
  };
  
  // Function to handle refresh button click
  const handleRefresh = () => {
    // Clear detections first
    clearDetectedFaces();
    // Then fetch new data
    fetchUsersData();
  };

  // Effect to fetch data when component mounts, camera is toggled, or camera selection changes
  useEffect(() => {
    if (isOn) {
      fetchUsersData();
      fetchVideoFrame(); // Get initial frame
      
      // Set up polling to refresh data every 10 seconds
      const dataIntervalId = setInterval(fetchUsersData, 10000);
      
      // Set up polling to refresh video frames every 200ms (5 FPS)
      videoIntervalRef.current = setInterval(fetchVideoFrame, 200);
      
      // Clean up intervals when component unmounts or camera turns off
      return () => {
        clearInterval(dataIntervalId);
        clearInterval(videoIntervalRef.current);
      };
    } else {
      // If camera is off, clear the video feed
      setVideoFeed(null);
      if (videoIntervalRef.current) {
        clearInterval(videoIntervalRef.current);
      }
    }
  }, [isOn, selectedCamera]);

  // Toggle camera state
  const toggleCamera = () => {
    const newState = !isOn;
    setIsOn(newState);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold text-gray-900">Face Recognition System</h1>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Main content area */}
          <div className="lg:col-span-8 space-y-8">
            <div className="bg-white p-6 rounded-xl shadow-md">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold">Live Recognition Feed</h2>
                <div className="flex items-center gap-4">
                  <select 
                    value={selectedCamera}
                    onChange={(e) => setSelectedCamera(Number(e.target.value))}
                    className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {cameras.map(camera => (
                      <option key={camera.id} value={camera.id}>
                        {camera.name}
                      </option>
                    ))}
                  </select>
                  <button 
                    onClick={toggleCamera}
                    className={cn(
                      "p-2 rounded-lg flex items-center gap-2",
                      isOn ? "bg-red-100 text-red-600" : "bg-green-100 text-green-600"
                    )}
                  >
                    <Power className="w-5 h-5" />
                    <span>{isOn ? "Turn Off" : "Turn On"}</span>
                  </button>
                  <button
                    onClick={handleRefresh}
                    disabled={isLoading || !isOn}
                    className="p-2 rounded-lg bg-blue-100 text-blue-600 flex items-center gap-2 disabled:opacity-50"
                  >
                    <RefreshCw className={cn("w-5 h-5", isLoading && "animate-spin")} />
                    <span>Refresh</span>
                  </button>
                </div>
              </div>
              
              <div className="relative aspect-video rounded-lg overflow-hidden bg-gray-900 border border-gray-300">
                {isOn ? (
                  videoFeed ? (
                    <img 
                      src={`data:image/jpeg;base64,${videoFeed}`} 
                      alt="Live camera feed" 
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <p className="text-white text-xl">Loading video feed...</p>
                    </div>
                  )
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-white text-xl">Camera Off</p>
                  </div>
                )}
              </div>
              
              {lastUpdated && (
                <p className="text-sm text-gray-500 mt-2">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </p>
              )}
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md">
              <h2 className="text-xl font-bold mb-4">Process Image Folder</h2>
              <div className="flex gap-3">
                <input
                  type="text"
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder="Enter folder path (e.g., C:\Images)"
                  className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={processFolder}
                  disabled={processingFolder || !folderPath}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg flex items-center gap-2 disabled:opacity-50"
                >
                  {processingFolder ? (
                    <>
                      <span className="animate-spin">⏳</span>
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <Upload className="w-5 h-5" />
                      <span>Process Folder</span>
                    </>
                  )}
                </button>
              </div>
              <p className="text-sm text-gray-500 mt-2">
                Specify a folder containing images to process for face recognition
              </p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md">
              <h2 className="text-xl font-bold mb-4">System Status</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-700">Backend Connection</h3>
                  <p className={cn(
                    "font-bold",
                    isLoading ? "text-yellow-500" : "text-green-500"
                  )}>
                    {isLoading ? "Connecting..." : "Connected"}
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-700">Camera Status</h3>
                  <p className={cn(
                    "font-bold",
                    isOn ? "text-green-500" : "text-red-500"
                  )}>
                    {isOn ? "Active" : "Inactive"}
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-700">Known Faces</h3>
                  <p className="font-bold text-blue-500">{knownUsers.length}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-700">Unknown Faces</h3>
                  <p className="font-bold text-amber-500">{unknownUsers.length}</p>
                </div>
              </div>
            </div>
          </div>
          
          {/* Sidebar */}
          <div className="lg:col-span-4 space-y-8">
            <div className="bg-white p-6 rounded-xl shadow-md">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold">Known Users</h2>
                <div className="flex items-center gap-2 text-green-600">
                  <Users className="w-5 h-5" />
                  <span className="font-semibold">{knownUsers.length}</span>
                </div>
              </div>
              
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {knownUsers.length > 0 ? (
                  knownUsers.map(user => (
                    <div key={user.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                      <div className="w-12 h-12 rounded-full overflow-hidden bg-gray-200 flex-shrink-0">
                        {user.image ? (
                          <img 
                            src={user.image} 
                            alt={user.name} 
                            className="w-full h-full object-cover" 
                            onError={(e) => {
                              e.target.src = "/default-user.png"; // Fallback image
                            }}
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center bg-gray-300 text-gray-600">
                            <Users className="w-6 h-6" />
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium truncate">{user.name}</h3>
                        <p className="text-sm text-gray-500 truncate">{user.time}</p>
                        <p className="text-xs text-gray-400 truncate">{user.camera}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    {isOn ? "No known users detected" : "Turn on camera to detect users"}
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold">Unknown Users</h2>
                <div className="flex items-center gap-2 text-amber-600">
                  <AlertCircle className="w-5 h-5" />
                  <span className="font-semibold">{unknownUsers.length}</span>
                </div>
              </div>
              
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {unknownUsers.length > 0 ? (
                  unknownUsers.map(user => (
                    <div key={user.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                      <div className="w-12 h-12 rounded-full overflow-hidden bg-gray-200 flex-shrink-0">
                        {user.image ? (
                          <img 
                            src={user.image} 
                            alt="Unknown person" 
                            className="w-full h-full object-cover" 
                            onError={(e) => {
                              e.target.src = "/unknown-user.png"; // Fallback image
                            }}
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center bg-gray-300 text-gray-600">
                            <AlertCircle className="w-6 h-6" />
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium truncate">Unknown Person</h3>
                        <p className="text-sm text-gray-500 truncate">{user.time}</p>
                        <p className="text-xs text-gray-400 truncate">{user.camera}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    {isOn ? "No unknown users detected" : "Turn on camera to detect users"}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}