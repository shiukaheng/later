import { Component, lazy, Suspense, useLayoutEffect, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router'
import { Feed } from './Feed'
import './style.css'

const PolicyDashboard = lazy(() => import('./PolicyDashboard'))

class DashboardBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return <main className="workspace-message">
      <p role="alert">The policy dashboard couldn’t load. Reload to try again.</p>
      <p><button className="control" onClick={() => window.location.reload()}>Reload dashboard</button> <Link to="/">Back to feed</Link></p>
    </main>
  }
}

function App() {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const onFeed = pathname === '/'

  useLayoutEffect(() => {
    function shortcut(event: KeyboardEvent) {
      if (event.key.toLowerCase() !== 'i' || event.repeat || event.metaKey || event.ctrlKey || event.altKey) return
      if (event.target instanceof HTMLElement &&
          (event.target.isContentEditable || event.target.closest('input, textarea, select'))) return
      if (onFeed) navigate('/dashboard')
    }
    window.addEventListener('keydown', shortcut)
    return () => window.removeEventListener('keydown', shortcut)
  }, [navigate, onFeed])

  return <>
    <Feed paused={!onFeed} />
    <Routes>
      <Route path="/" element={null} />
      <Route path="/dashboard" element={<DashboardBoundary>
        <Suspense fallback={<p className="workspace-message" role="status">Loading policy editor…</p>}>
          <PolicyDashboard />
        </Suspense>
      </DashboardBoundary>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </>
}

createRoot(document.getElementById('root')!).render(<BrowserRouter><App /></BrowserRouter>)
