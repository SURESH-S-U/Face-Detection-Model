import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import LiveFeed from './pages/LiveFeed';
import Logs from './pages/Logs';
import Attendance from './pages/Attendance';
import VideoAnalysisPage from './pages/upload';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="live" element={<LiveFeed />} />
          <Route path='upload' element={<VideoAnalysisPage/>} />
          <Route path='attendance' element={<Attendance/>} />
          <Route path="logs" element={<Logs />} />
          
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;