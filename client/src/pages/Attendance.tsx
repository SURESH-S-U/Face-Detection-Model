import React, { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Calendar, Search } from 'lucide-react';
import { NAvigation } from '../components/Layout';

const COLORS = ['#3B82F6', '#F59E0B', '#10B981', '#6366F1'];

interface AttendanceData {
  name: string;
  value: number;
}

interface AttendanceLog {
  id: string;
  name: string;
  date: string;
  timeIn: string;
  timeOut: string;
  camera: string;
  status: string;
  image: string;
}

interface QuickStats {
  totalEmployees: number;
  presentToday: number;
  lateToday: number;
  onLeave: number;
}

// Mock data for fallback when backend is not available
const mockQuickStats: QuickStats = {
  totalEmployees: 120,
  presentToday: 20,
  lateToday: 5,
  onLeave: 100
};

const mockAttendanceLogs: AttendanceLog[] = [
  {
    id: '1',
    name: 'John Doe',
    date: '2025-03-20',
    timeIn: '09:00 AM',
    timeOut: '05:30 PM',
    camera: 'Main Entrance',
    status: 'on-time',
    image: '/api/placeholder/100/100'
  },
  {
    id: '2',
    name: 'Jane Smith',
    date: '2025-03-20',
    timeIn: '09:15 AM',
    timeOut: '05:45 PM',
    camera: 'Side Gate',
    status: 'late',
    image: '/api/placeholder/100/100'
  },
];

const Attendance: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<string>(format(new Date(), 'yyyy-MM-dd'));
  const [attendanceData, setAttendanceData] = useState<AttendanceData[]>([]);
  const [attendanceLogs, setAttendanceLogs] = useState<AttendanceLog[]>([]);
  const [quickStats, setQuickStats] = useState<QuickStats>({
    totalEmployees: 0,
    presentToday: 0,
    lateToday: 0,
    onLeave: 0
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [usingMockData, setUsingMockData] = useState<boolean>(false);

  // Helper function to calculate attendance data from quick stats
  const calculateAttendanceData = (stats: QuickStats): AttendanceData[] => {
    // If total employees is 0, avoid division by zero
    if (stats.totalEmployees === 0) {
      return [
        { name: 'On Time', value: 0 },
        { name: 'Late', value: 0 },
        { name: 'Absent', value: 0 },
      ];
    }
    
    // Calculate percentages based on the stats
    const onTimePercentage = Math.round((stats.presentToday / stats.totalEmployees) * 100);
    const latePercentage = Math.round((stats.lateToday / stats.totalEmployees) * 100);
    const absentPercentage = Math.round((stats.onLeave / stats.totalEmployees) * 100);
    
    return [
      { name: 'On Time', value: onTimePercentage },
      { name: 'Late', value: latePercentage },
      { name: 'Absent', value: absentPercentage },
    ];
  };

  // Helper function to get status class
  const getStatusClass = (status: string): string => {
    if (status === 'on-time') {
      return "px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800";
    } else {
      return "px-3 py-1 rounded-full text-sm font-medium bg-amber-100 text-amber-800";
    }
  };

  // Fetch quick stats
  const fetchQuickStats = async (): Promise<QuickStats> => {
    try {
      const response = await fetch(`/api/attendance/stats?date=${selectedDate}`);
      if (!response.ok) throw new Error('Failed to fetch quick stats');
      return await response.json();
    } catch (err) {
      console.error('Error fetching quick stats:', err);
      // Return mock data as fallback
      setUsingMockData(true);
      return mockQuickStats;
    }
  };

  // Fetch attendance logs with search and date filtering
  const fetchAttendanceLogs = async (): Promise<AttendanceLog[]> => {
    try {
      const response = await fetch(`/api/attendance/logs?date=${selectedDate}&search=${searchTerm}`);
      if (!response.ok) throw new Error('Failed to fetch attendance logs');
      return await response.json();
    } catch (err) {
      console.error('Error fetching attendance logs:', err);
      // Return mock data as fallback
      setUsingMockData(true);
      
      // Filter mock logs by search term if provided
      if (searchTerm) {
        return mockAttendanceLogs.filter(log => 
          log.name.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }
      return mockAttendanceLogs;
    }
  };

  // Fetch all data when date or search term changes
  useEffect(() => {
    const fetchAllData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const [statsData, logsData] = await Promise.all([
          fetchQuickStats(),
          fetchAttendanceLogs()
        ]);
        
        setQuickStats(statsData);
        // Calculate attendance data from quick stats
        setAttendanceData(calculateAttendanceData(statsData));
        setAttendanceLogs(logsData);
      } catch (err) {
        setError('Failed to load attendance data');
        console.error('Error fetching all data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAllData();
  }, [selectedDate, searchTerm]);

  return (
    <div className="space-y-8">
      <NAvigation/>
      <div className="ml-[100px]">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Attendance Dashboard</h1>
        {usingMockData && (
          <div className="bg-amber-100 text-amber-800 px-4 py-2 rounded-lg text-sm">
            Using mock data (backend not connected)
          </div>
        )}
        <div className="flex gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search employees..."
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

      {loading ? (
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
                    >
                      {attendanceData.map((_entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-3 gap-4 mt-6">
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
                  { label: 'Total Employees', value: quickStats.totalEmployees.toString() },
                  { label: 'Present Today', value: quickStats.presentToday.toString() },
                  { label: 'Late Today', value: quickStats.lateToday.toString() },
                  { label: 'On Leave', value: quickStats.onLeave.toString() }
                ].map((stat, index) => (
                  <div key={index} className="bg-gray-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{stat.value}</div>
                    <div className="text-gray-600">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-lg mt-8">
            <h2 className="text-xl font-bold mb-6">Attendance Logs</h2>
            {attendanceLogs.length === 0 ? (
              <p className="text-center text-gray-500 py-4">No attendance records found</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left border-b border-gray-200">
                      <th className="pb-4 font-semibold text-gray-600">Employee</th>
                      <th className="pb-4 font-semibold text-gray-600">Date</th>
                      <th className="pb-4 font-semibold text-gray-600">Time In</th>
                      <th className="pb-4 font-semibold text-gray-600">Time Out</th>
                      <th className="pb-4 font-semibold text-gray-600">Location</th>
                      <th className="pb-4 font-semibold text-gray-600">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {attendanceLogs.map(log => (
                      <tr key={log.id} className="hover:bg-gray-50">
                        <td className="py-4">
                          <div className="flex items-center gap-4">
                            <img
                              src={log.image}
                              alt={log.name}
                              className="w-10 h-10 rounded-full object-cover"
                            />
                            <span className="font-medium">{log.name}</span>
                          </div>
                        </td>
                        <td className="py-4">{format(new Date(log.date), 'MMM dd, yyyy')}</td>
                        <td className="py-4">{log.timeIn}</td>
                        <td className="py-4">{log.timeOut}</td>
                        <td className="py-4">{log.camera}</td>
                        <td className="py-4">
                          <span className={getStatusClass(log.status)}>
                            {log.status.charAt(0).toUpperCase() + log.status.slice(1)}
                          </span>
                        </td>
                      </tr>
                    ))}
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

export default Attendance;