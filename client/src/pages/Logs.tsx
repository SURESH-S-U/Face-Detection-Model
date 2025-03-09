import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { cn } from '../lib/utils';
import { NAvigation } from '../components/Layout';


const mockTimeData = [
  { time: '08:00', known: 4, unknown: 1 },
  { time: '09:00', known: 7, unknown: 2 },
  { time: '10:00', known: 12, unknown: 3 },
  { time: '11:00', known: 15, unknown: 4 },
  { time: '12:00', known: 10, unknown: 2 },
  { time: '13:00', known: 8, unknown: 1 },
];

const mockLogs = [
  { id: 1, name: 'John Doe', status: 'known', time: '13:45:22', image: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100&h=100&fit=crop' },
  { id: 2, status: 'unknown', time: '13:42:15', image: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100&h=100&fit=crop' },
  { id: 3, name: 'Jane Smith', status: 'known', time: '13:40:01', image: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&h=100&fit=crop' },
];

export default function Logs() {
  return (
    <div className="space-y-8">
      <NAvigation/>
    <div className="ml-[100px]">
      <div className="bg-white p-6 rounded-2xl shadow-lg">
        <h2 className="text-2xl font-bold mb-6">Recognition Trends</h2>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="known" stroke="#3B82F6" strokeWidth={2} />
              <Line type="monotone" dataKey="unknown" stroke="#F59E0B" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white p-6 rounded-2xl shadow-lg">
        <h2 className="text-2xl font-bold mb-6">Recognition Log</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-gray-200">
                <th className="pb-4 font-semibold text-gray-600">Time</th>
                <th className="pb-4 font-semibold text-gray-600">Person</th>
                <th className="pb-4 font-semibold text-gray-600">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {mockLogs.map(log => (
                <tr key={log.id} className="hover:bg-gray-50">
                  <td className="py-4">
                    <span className="text-sm text-gray-600">{log.time}</span>
                  </td>
                  <td className="py-4">
                    <div className="flex items-center gap-4">
                      <img
                        src={log.image}
                        alt={log.name || 'Unknown person'}
                        className="w-10 h-10 rounded-full object-cover"
                      />
                      <span className="font-medium">{log.name || 'Unknown Person'}</span>
                    </div>
                  </td>
                  <td className="py-4">
                    <span className={cn(
                      "px-3 py-1 rounded-full text-sm font-medium",
                      log.status === 'known' 
                        ? "bg-green-100 text-green-800"
                        : "bg-amber-100 text-amber-800"
                    )}>
                      {log.status.charAt(0).toUpperCase() + log.status.slice(1)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      </div>
    </div>
  );
}