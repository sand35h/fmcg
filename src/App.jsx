import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './hooks/useAuth'
import Layout from './components/layout/Layout'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import ForecastExplorer from './pages/ForecastExplorer'
import Inventory from './pages/Inventory'
import ScenarioPlanning from './pages/ScenarioPlanning'
import DataJobs from './pages/DataJobs'
import ModelStatus from './pages/ModelStatus'
import Settings from './pages/Settings'

// Protected Route wrapper
function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full" style={{ minHeight: '100vh' }}>
        <div className="spinner" style={{ width: 40, height: 40 }}></div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return children
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <Layout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/forecasts" element={<ForecastExplorer />} />
                <Route path="/inventory" element={<Inventory />} />
                <Route path="/scenarios" element={<ScenarioPlanning />} />
                <Route path="/data-jobs" element={<DataJobs />} />
                <Route path="/models" element={<ModelStatus />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Layout>
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}

export default App
