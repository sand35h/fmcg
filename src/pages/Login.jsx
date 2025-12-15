import { useState } from 'react'
import { useNavigate, Navigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { Activity, Mail, Lock, Eye, EyeOff, AlertCircle } from 'lucide-react'

export default function Login() {
    const { login, isAuthenticated, isLoading } = useAuth()
    const navigate = useNavigate()
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [showPassword, setShowPassword] = useState(false)
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    // Redirect if already authenticated
    if (isAuthenticated) {
        return <Navigate to="/" replace />
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        setLoading(true)

        const result = await login(email, password)

        if (result.success) {
            navigate('/')
        } else {
            setError(result.error)
        }

        setLoading(false)
    }

    return (
        <div
            style={{
                minHeight: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 'var(--space-6)',
                background: 'var(--bg-primary)'
            }}
        >
            {/* Background Effects */}
            <div
                style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: `
            radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.2) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, 0.1) 0%, transparent 60%)
          `,
                    zIndex: 0
                }}
            />

            {/* Login Card */}
            <div
                className="animate-slideUp"
                style={{
                    width: '100%',
                    maxWidth: 420,
                    background: 'var(--bg-card)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid var(--border-primary)',
                    borderRadius: 'var(--radius-2xl)',
                    padding: 'var(--space-10)',
                    boxShadow: 'var(--shadow-xl)',
                    position: 'relative',
                    zIndex: 1
                }}
            >
                {/* Logo */}
                <div style={{ textAlign: 'center', marginBottom: 'var(--space-8)' }}>
                    <div
                        style={{
                            width: 64,
                            height: 64,
                            borderRadius: 'var(--radius-xl)',
                            background: 'var(--gradient-primary)',
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            marginBottom: 'var(--space-4)',
                            boxShadow: 'var(--shadow-glow)'
                        }}
                    >
                        <Activity size={32} color="white" />
                    </div>
                    <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 'var(--space-2)' }}>
                        FMCG Forecast
                    </h1>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: 0 }}>
                        Supply & Demand Forecasting System
                    </p>
                </div>

                {/* Login Form */}
                <form onSubmit={handleSubmit}>
                    {/* Error Alert */}
                    {error && (
                        <div
                            className="alert alert-danger animate-fadeIn"
                            style={{ marginBottom: 'var(--space-5)' }}
                        >
                            <AlertCircle size={18} />
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Email Field */}
                    <div className="form-group">
                        <label className="form-label">Email Address</label>
                        <div style={{ position: 'relative' }}>
                            <Mail
                                size={18}
                                style={{
                                    position: 'absolute',
                                    left: 'var(--space-4)',
                                    top: '50%',
                                    transform: 'translateY(-50%)',
                                    color: 'var(--text-muted)'
                                }}
                            />
                            <input
                                type="email"
                                className="form-input"
                                placeholder="Enter your email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                style={{ paddingLeft: 'var(--space-12)' }}
                            />
                        </div>
                    </div>

                    {/* Password Field */}
                    <div className="form-group">
                        <label className="form-label">Password</label>
                        <div style={{ position: 'relative' }}>
                            <Lock
                                size={18}
                                style={{
                                    position: 'absolute',
                                    left: 'var(--space-4)',
                                    top: '50%',
                                    transform: 'translateY(-50%)',
                                    color: 'var(--text-muted)'
                                }}
                            />
                            <input
                                type={showPassword ? 'text' : 'password'}
                                className="form-input"
                                placeholder="Enter your password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                style={{ paddingLeft: 'var(--space-12)', paddingRight: 'var(--space-12)' }}
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                style={{
                                    position: 'absolute',
                                    right: 'var(--space-3)',
                                    top: '50%',
                                    transform: 'translateY(-50%)',
                                    background: 'transparent',
                                    border: 'none',
                                    cursor: 'pointer',
                                    color: 'var(--text-muted)',
                                    padding: 'var(--space-2)'
                                }}
                            >
                                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                            </button>
                        </div>
                    </div>

                    {/* Submit Button */}
                    <button
                        type="submit"
                        className="btn btn-primary btn-lg"
                        disabled={loading}
                        style={{ width: '100%', marginTop: 'var(--space-4)' }}
                    >
                        {loading ? (
                            <>
                                <span className="spinner" />
                                Signing in...
                            </>
                        ) : (
                            'Sign In'
                        )}
                    </button>
                </form>

                {/* Demo Credentials */}
                <div
                    style={{
                        marginTop: 'var(--space-8)',
                        padding: 'var(--space-4)',
                        background: 'rgba(99, 102, 241, 0.1)',
                        borderRadius: 'var(--radius-lg)',
                        border: '1px solid rgba(99, 102, 241, 0.2)'
                    }}
                >
                    <div style={{
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        color: 'var(--primary-400)',
                        marginBottom: 'var(--space-2)',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                    }}>
                        Demo Credentials
                    </div>
                    <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                        <div style={{ marginBottom: 'var(--space-1)' }}>
                            <strong>Admin:</strong> admin@fmcg.com / admin123
                        </div>
                        <div style={{ marginBottom: 'var(--space-1)' }}>
                            <strong>Planner:</strong> planner@fmcg.com / planner123
                        </div>
                        <div>
                            <strong>Demo:</strong> demo@fmcg.com / demo123
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div
                    style={{
                        marginTop: 'var(--space-6)',
                        textAlign: 'center',
                        fontSize: '0.75rem',
                        color: 'var(--text-muted)'
                    }}
                >
                    Final Year Project • Supply Chain Analytics
                </div>
            </div>
        </div>
    )
}
