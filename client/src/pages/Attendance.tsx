import React, { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Calendar, Search, RefreshCw } from 'lucide-react';
import { NAvigation } from '../components/Layout';

const COLORS = ['#3B82F6', '#10B981'];

interface AttendanceData {
  name: string;
  value: number;
}

interface FaceData {
  face_id: string;
  name: string;
  timestamp: string;
  camera_id: number;
  face_image: string;
  is_known: boolean;
  _id: string;
}

interface QuickStats {
  totalStudents: number;
  presentToday: number;
  lateToday: number;
  onLeave: number;
}

// Default total students
const TOTAL_STUDENTS = 10;

const Attendance: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<string>(format(new Date(), 'yyyy-MM-dd'));
  const [attendanceData, setAttendanceData] = useState<AttendanceData[]>([]);
  const [faceData, setFaceData] = useState<FaceData[]>([]);
  const [quickStats, setQuickStats] = useState<QuickStats>({
    totalStudents: TOTAL_STUDENTS,
    presentToday: 0,
    lateToday: 0,
    onLeave: TOTAL_STUDENTS
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState<boolean>(false);

  // API base URL
  const API_BASE_URL = 'http://localhost:5000';

  // Helper function to calculate attendance data from quick stats
  const calculateAttendanceData = (stats: QuickStats): AttendanceData[] => {
    // Calculate percentages based on the stats - removed late percentage
    const onTimePercentage = Math.round((stats.presentToday / stats.totalStudents) * 100) || 0;
    const absentPercentage = 100 - onTimePercentage;
    
    return [
      { name: 'Present', value: onTimePercentage },
      { name: 'Absent', value: absentPercentage },
    ];
  };

  // Helper function to get status class
  const getStatusClass = (isKnown: boolean): string => {
    return isKnown
      ? "px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800"
      : "px-3 py-1 rounded-full text-sm font-medium bg-amber-100 text-amber-800";
  };

  // Convert face data to attendance logs format for display
  const formatFaceData = (faces: FaceData[]) => {
    return faces
      .filter(face => face.is_known) // Only include known faces
      .filter(face => {
        // If search term is present, filter by name
        if (searchTerm) {
          return face.name.toLowerCase().includes(searchTerm.toLowerCase());
        }
        return true;
      })
      .filter(face => {
        // Filter by selected date if applicable
        if (!face.timestamp) return false;
        const faceDate = format(new Date(face.timestamp), 'yyyy-MM-dd');
        return faceDate === selectedDate;
      })
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  };

  // Fetch face count data
  const fetchFaceCount = async (): Promise<CountData> => {
    try {
      const response = await fetch(`${API_BASE_URL}/face_count`);
      if (!response.ok) throw new Error('Failed to fetch face count');
      return await response.json();
    } catch (err) {
      console.error('Error fetching face count:', err);
      throw err;
    }
  };

  // Fetch all faces data
  const fetchFaceData = async (): Promise<FaceData[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/detection_data`);
      if (!response.ok) throw new Error('Failed to fetch detection data');
      return await response.json();
    } catch (err) {
      console.error('Error fetching detection data:', err);
      throw err;
    }
  };

  // Handle refresh
  const handleRefresh = () => {
    setRefreshing(true);
    fetchAllData();
  };

  // Reset faces in the backend
  const resetFaces = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reset_faces`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) throw new Error('Failed to reset faces');
      
      // Refresh data after reset
      handleRefresh();
    } catch (err) {
      console.error('Error resetting faces:', err);
      setError('Failed to reset faces');
    }
  };

  // Fetch all data
  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const [countData, facesData] = await Promise.all([
        fetchFaceCount(),
        fetchFaceData()
      ]);
      
      // Update quick stats based on count data
      const updatedStats: QuickStats = {
        totalStudents: TOTAL_STUDENTS,
        presentToday: countData.known_count,
        lateToday: 0, // We don't track late students in this system
        onLeave: TOTAL_STUDENTS - countData.known_count
      };
      
      setQuickStats(updatedStats);
      setAttendanceData(calculateAttendanceData(updatedStats));
      setFaceData(facesData);
    } catch (err) {
      setError('Failed to load attendance data');
      console.error('Error fetching all data:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchAllData();
    
    // Set up polling every 10 seconds to get fresh data
    const interval = setInterval(() => {
      fetchAllData();
    }, 10000);
    
    return () => clearInterval(interval);
  }, []);

  // Filter face data when search term or date changes
  useEffect(() => {
    // No need to refetch, just re-filter the existing data
    const filteredData = formatFaceData(faceData);
    
    // Calculate stats based on filtered data for the selected date
    const todayFaces = faceData.filter(face => {
      if (!face.timestamp) return false;
      const faceDate = format(new Date(face.timestamp), 'yyyy-MM-dd');
      return faceDate === selectedDate;
    });
    
    const knownCount = todayFaces.filter(face => face.is_known).length;
    
    // Update quick stats based on filtered data
    const updatedStats: QuickStats = {
      totalStudents: TOTAL_STUDENTS,
      presentToday: knownCount,
      lateToday: 0,
      onLeave: TOTAL_STUDENTS - knownCount
    };
    
    setQuickStats(updatedStats);
    setAttendanceData(calculateAttendanceData(updatedStats));
  }, [selectedDate, searchTerm, faceData]);

  // Apply formatting to face data
  const filteredFaces = formatFaceData(faceData);

  return (
    <div className="space-y-8">
      <NAvigation/>
      <div className="ml-24 mr-4 pb-8">
        <div className="flex justify-between items-center flex-wrap">
          <h1 className="text-2xl font-bold">Attendance Dashboard</h1>
          <div className="flex gap-4 flex-wrap">
            <button
              onClick={resetFaces}
              className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
            >
              Reset Faces
            </button>
            <button
              onClick={handleRefresh}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
              disabled={refreshing}
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search students..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {loading && !refreshing ? (
          <div className="flex justify-center items-center h-64">
            <p className="text-gray-500">Loading attendance data...</p>
          </div>
        ) : error ? (
          <div className="flex justify-center items-center h-64">
            <p className="text-red-500">{error}</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
              <div className="bg-white p-6 rounded-2xl shadow-lg">
                <h2 className="text-xl font-bold mb-6">Today's Attendance Overview</h2>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={attendanceData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        fill="#8884d8"
                        paddingAngle={5}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}%`}
                      >
                        {attendanceData.map((_entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => `${value}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-6">
                  {attendanceData.map((item, index) => (
                    <div key={item.name} className="text-center">
                      <div className="text-2xl font-bold" style={{ color: COLORS[index] }}>
                        {item.value}%
                      </div>
                      <div className="text-gray-600">{item.name}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-lg">
                <h2 className="text-xl font-bold mb-6">Quick Stats</h2>
                <div className="grid grid-cols-2 gap-6">
                  {[
                    { label: 'Total Students', value: quickStats.totalStudents.toString() },
                    { label: 'Present Today', value: quickStats.presentToday.toString() },
                    { label: 'Absent Today', value: quickStats.onLeave.toString() }
                  ].map((stat, index) => (
                    <div key={index} className="bg-gray-50 p-4 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">{stat.value}</div>
                      <div className="text-gray-600">{stat.label}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-6">
                  <h3 className="font-semibold mb-2">Today: {format(new Date(selectedDate), 'MMMM dd, yyyy')}</h3>
                  <p className="text-gray-600">
                    Out of {quickStats.totalStudents} students, {quickStats.presentToday} are present today.
                    {quickStats.onLeave > 0 && ` ${quickStats.onLeave} students are absent.`}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-lg mt-8">
              <h2 className="text-xl font-bold mb-6">Face Recognition Logs</h2>
              <div className="mt-4 mb-6">
                <a
                  href={`${API_BASE_URL}/video_feed`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium"
                >
                  View Live Camera Feed
                </a>
              </div>
              {filteredFaces.length === 0 ? (
                <p className="text-center text-gray-500 py-4">No attendance records found for the selected date</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left border-b border-gray-200">
                        <th className="pb-4 font-semibold text-gray-600">Person</th>
                        <th className="pb-4 font-semibold text-gray-600">Date</th>
                        <th className="pb-4 font-semibold text-gray-600">Time</th>
                        <th className="pb-4 font-semibold text-gray-600">Camera</th>
                        <th className="pb-4 font-semibold text-gray-600">Status</th>
                      </tr>
                    </thead>
                    
                    <tbody className="divide-y divide-gray-100">
                      {filteredFaces.map(face => {
                        const faceDate = new Date(face.timestamp);
                        return (
                          <tr key={face._id} className="hover:bg-gray-50">
                            <td className="py-4">
                              <div className="flex items-center gap-4">
                                <img
                                  src={`data:image/jpeg;base64,${face.face_image}`}
                                  alt={face.name}
                                  className="w-10 h-10rounded-full object-cover"
                                />
                                <span className="font-medium">{face.name}</span>
                              </div>
                            </td>
                            <td className="py-4">{format(faceDate, 'MMM dd, yyyy')}</td>
                            <td className="py-4">{format(faceDate, 'hh:mm:ss a')}</td>
                            <td className="py-4">Camera {face.camera_id + 1}</td>
                            <td className="py-4">
                              <span className={getStatusClass(face.is_known)}>
                                {face.is_known ? 'Present' : 'Unknown'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

interface CountData {
  known_count: number;
  unknown_count: number;
  total_count: number;
}

export default Attendance;