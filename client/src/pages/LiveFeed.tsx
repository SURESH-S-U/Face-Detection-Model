import React, { useState } from 'react';
import Webcam from 'react-webcam';
import { Users, AlertCircle, Power } from 'lucide-react';
import { cn } from '../lib/utils';
import { NAvigation } from '../components/Layout';

const cameras = [
  { id: 1, name: 'Main Entrance' },
];

export default function LiveFeed() {
  const [selectedCamera, setSelectedCamera] = useState(1);
  const [isOn, setIsOn] = useState(true);
  const [knownUsers] = useState([
    { 
      id: 1, 
      name: 'John Doe', 
      time: '10:30 AM', 
      camera: 'Main Entrance',
      image: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100&h=100&fit=crop' 
    },
    { 
      id: 2, 
      name: 'Jane Smith', 
      time: '10:45 AM', 
      camera: 'Side Gate',
      image: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&h=100&fit=crop' 
    },
  ]);

  const [unknownUsers] = useState([
    { 
      id: 1, 
      time: '10:35 AM', 
      camera: 'Reception',
      image: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100&h=100&fit=crop' 
    },
  ]);

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
                    onClick={() => setIsOn(!isOn)}
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
                  <Webcam
                    className="w-full h-full object-cover"
                    mirrored={true}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-white text-xl">Camera Off</p>
                  </div>
                )}
              </div>
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
                {knownUsers.map(user => (
                  <div key={user.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                    <img src={user.image} alt={user.name} className="w-12 h-12 rounded-full object-cover" />
                    <div className="flex-1">
                      <h3 className="font-medium">{user.name}</h3>
                      <p className="text-sm text-gray-500">{user.time} - {user.camera}</p>
                    </div>
                  </div>
                ))}
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
                {unknownUsers.map(user => (
                  <div key={user.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                    <img src={user.image} alt="Unknown person" className="w-12 h-12 rounded-full object-cover" />
                    <div className="flex-1">
                      <h3 className="font-medium">Unknown Person</h3>
                      <p className="text-sm text-gray-500">{user.time} - {user.camera}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
