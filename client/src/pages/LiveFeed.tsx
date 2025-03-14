import React, { useState, useEffect, useRef } from 'react';
import { Users, AlertCircle, Power } from 'lucide-react';
import { cn } from '../lib/utils';
import { NAvigation } from '../components/Layout';

const cameras = [
  { id: 0, name: 'Main Entrance' },
];

export default function LiveFeed() {
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [isOn, setIsOn] = useState(true);
  const [knownUsers, setKnownUsers] = useState([]);
  const [unknownUsers, setUnknownUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);

  // Backend URL - using port 5000
  const API_BASE_URL = "http://localhost:5000";

  // Fetch recognition data from backend
  const fetchRecognitionData = async () => {
    if (!isOn) return; // Don't fetch if camera is off

    setLoading(true);
    setError(null);
    
    try {
      console.log("Attempting to fetch data from:", `${API_BASE_URL}/detection_data`);
      
      const response = await fetch(`${API_BASE_URL}/detection_data`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        mode: 'cors',
      });
      
      console.log("Response status:", response.status);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Recognition data received:", data);
      
      if (Array.isArray(data)) {
        // Process the detections and separate known and unknown users
        const known = [];
        const unknown = [];
        
        data.forEach((detection, index) => {
          const userObject = {
            id: detection._id || `unknown-${index}`,
            name: detection.name || "Unknown",
            time: detection.timestamp || new Date().toISOString(),
            camera: `Camera ${detection.camera_id !== undefined ? detection.camera_id : 0}`,
            // Make sure face_image is properly handled
            image: detection.face_image ? `data:image/jpeg;base64,${detection.face_image}` : null,
          };
          
          if (detection.name && detection.name !== "Unknown") {
            known.push(userObject);
          } else {
            unknown.push(userObject);
          }
        });
        
        setKnownUsers(known);
        setUnknownUsers(unknown);
      } else {
        console.error("Unexpected data format:", data);
      }
    } catch (error) {
      console.error("Error fetching recognition data:", error);
      setError(`Failed to fetch recognition data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Stop the backend process
  const stopBackendProcess = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stop`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to stop backend process: ${response.statusText}`);
      }
      
      console.log("Backend process stopped successfully");
    } catch (error) {
      console.error("Error stopping backend process:", error);
      setError(`Failed to stop backend process: ${error.message}`);
    }
  };

  // Handle camera on/off toggle
  const handleCameraToggle = async () => {
    if (isOn) {
      // Stop the backend process when turning off the camera
      await stopBackendProcess();
    }
    setIsOn(!isOn);
  };

  // Setup video stream when camera is toggled or changed
  useEffect(() => {
    if (isOn && videoRef.current) {
      // Get the video URL - using /video_feed
      const videoUrl = `${API_BASE_URL}/video_feed`;
      
      // For MJPEG streams, we need to set the src directly
      videoRef.current.src = videoUrl;
      
      // Add error handler
      const handleVideoError = () => {
        console.error("Error loading video feed");
        setError("Failed to load video feed. Please check if the backend is running.");
      };
      
      videoRef.current.addEventListener('error', handleVideoError);
      
      // Clean up
      return () => {
        if (videoRef.current) {
          videoRef.current.removeEventListener('error', handleVideoError);
          videoRef.current.src = '';
        }
      };
    }
  }, [isOn, selectedCamera]);

  // Handle camera on/off toggle for recognition data
  useEffect(() => {
    if (isOn) {
      // Fetch immediately when turned on
      fetchRecognitionData();
      
      // Set up polling interval
      const intervalId = setInterval(fetchRecognitionData, 5000); // Poll every 5 seconds
      
      // Clean up interval when component unmounts or camera is turned off
      return () => clearInterval(intervalId);
    }
  }, [isOn, selectedCamera]);

  // Format time to 12-hour format
  const formatTime = (dateTimeStr) => {
    try {
      if (!dateTimeStr) return "";
      
      // Extract time part from the datetime string
      // Handle multiple possible formats
      let timePart;
      if (dateTimeStr.includes(' ')) {
        // Format: "2023-01-01 14:30:00"
        timePart = dateTimeStr.split(' ')[1];
      } else if (dateTimeStr.includes('T')) {
        // Format: "2023-01-01T14:30:00"
        timePart = dateTimeStr.split('T')[1].split('.')[0];
      } else {
        return dateTimeStr;
      }
      
      // Extract hours and minutes
      const [hours, minutes] = timePart.split(':');
      const hoursNum = parseInt(hours);
      const amPm = hoursNum >= 12 ? 'PM' : 'AM';
      const hours12 = hoursNum % 12 || 12;
      return `${hours12}:${minutes} ${amPm}`;
    } catch (e) {
      console.error("Error formatting time:", e, dateTimeStr);
      return dateTimeStr || "N/A";
    }
  };

  // Debug function to log and check details
  const debugImageUrl = (imageUrl) => {
    console.log("Image URL:", imageUrl);
    return imageUrl || "/api/placeholder/100/100";
  };

  return (
    <div className="flex">
      <NAvigation />
      <div className="flex flex-col h-screen ml-[100px] w-full">
        <div className="grid grid-cols-12 gap-8 p-5 flex-1">
          <div className="col-span-8 space-y-8">
            <div className="bg-white p-6 rounded-2xl shadow-lg">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold">Live Recognition Feed</h2>
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
                    onClick={handleCameraToggle}
                    className={cn(
                      "p-2 rounded-lg",
                      isOn ? "bg-red-100 text-red-600" : "bg-green-100 text-green-600"
                    )}
                  >
                    <Power className="w-6 h-6" />
                  </button>
                </div>
              </div>
              
              <div className="relative aspect-video rounded-lg overflow-hidden bg-gray-900">
                {isOn ? (
                  <img
                    ref={videoRef}
                    className="w-full h-full object-contain"
                    alt="Live video feed"
                    style={{ width: '100%', height: '100%' }}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-white text-xl">Camera Off</p>
                  </div>
                )}
              </div>
              
              {error && (
                <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
                  {error}
                </div>
              )}
            </div>
          </div>

          <div className="col-span-4 space-y-8">
            <div className="bg-white p-6 rounded-2xl shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold">Known Users</h2>
                <div className="flex items-center gap-2 text-green-600">
                  <Users className="w-5 h-5" />
                  <span className="font-semibold">{knownUsers.length}</span>
                </div>
              </div>
              <div className="space-y-4">
                {loading && knownUsers.length === 0 ? (
                  <p className="text-gray-500 text-center py-2">Loading...</p>
                ) : knownUsers.length > 0 ? (
                  knownUsers.map(user => (
                    <div key={user.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                      {user.image ? (
                        <img 
                          src={user.image}
                          alt={user.name} 
                          className="w-12 h-12 rounded-full object-cover" 
                          onError={(e) => {
                            console.error("Image load error:", e);
                            e.target.src = "/api/placeholder/100/100";
                          }}
                        />
                      ) : (
                        <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
                          <Users className="w-6 h-6 text-gray-500" />
                        </div>
                      )}
                      <div className="flex-1">
                        <h3 className="font-medium">{user.name}</h3>
                        <p className="text-sm text-gray-500">{formatTime(user.time)} - {user.camera}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-500 text-center py-2">No known users detected</p>
                )}
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold">Unknown Users</h2>
                <div className="flex items-center gap-2 text-amber-600">
                  <AlertCircle className="w-5 h-5" />
                  <span className="font-semibold">{unknownUsers.length}</span>
                </div>
              </div>
              <div className="space-y-4">
                {loading && unknownUsers.length === 0 ? (
                  <p className="text-gray-500 text-center py-2">Loading...</p>
                ) : unknownUsers.length > 0 ? (
                  unknownUsers.map(user => (
                    <div key={user.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                      {user.image ? (
                        <img 
                          src={user.image}
                          alt="Unknown person" 
                          className="w-12 h-12 rounded-full object-cover"
                          onError={(e) => {
                            console.error("Image load error:", e);
                            e.target.src = "/api/placeholder/100/100";
                          }}
                        />
                      ) : (
                        <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
                          <AlertCircle className="w-6 h-6 text-gray-500" />
                        </div>
                      )}
                      <div className="flex-1">
                        <h3 className="font-medium">Unknown Person</h3>
                        <p className="text-sm text-gray-500">{formatTime(user.time)} - {user.camera}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-500 text-center py-2">No unknown users detected</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}