import { useState, useEffect } from 'react'
import {
    Brain,
    RefreshCw,
    CheckCircle,
    AlertTriangle,
    Clock,
    TrendingUp,
    Activity,
    Layers,
    Play
} from 'lucide-react'
import api from '../services/api'
import BarChart from '../components/charts/BarChart'
import { formatDate, formatPercent } from '../utils/formatters'
import { CHART_COLORS } from '../utils/constants'

export default function ModelStatus() {
    const [loading, setLoading] = useState(true)
    const [retraining, setRetraining] = useState(false)
    const [modelStatus, setModelStatus] = useState(null)

    useEffect(() => {
        loadModelStatus()
    }, [])

    const loadModelStatus = async () => {
        setLoading(true)
        try {
            const data = await api.getModelStatus()
            setModelStatus(data)
        } catch (error) {
            console.error('Error loading model status:', error)
        }
        setLoading(false)
    }

    const handleRetrain = async () => {
        if (!confirm('Are you sure you want to trigger model retraining? This may take several hours.')) {
            return
        }

        setRetraining(true)
        try {
            const result = await api.triggerRetrain()
            if (result.success) {
                alert(result.message)
                loadModelStatus()
            }
        } catch (error) {
            console.error('Retrain error:', error)
            alert('Failed to trigger retraining. Please try again.')
        }
        setRetraining(false)
    }

    // Feature importance chart data
    const featureData = modelStatus?.featureImportance?.map(f => ({
        name: f.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        importance: f.importance * 100
    })) || []

    return (
        <div className="animate-fadeIn">
            {/* Page Header */}
            <div
                style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    marginBottom: 'var(--space-6)'
                }}
            >
                <div>
                    <h1 style={{ marginBottom: 'var(--space-2)' }}>
                        <Brain size={28} style={{ marginRight: 'var(--space-3)', verticalAlign: 'middle' }} />
                        Model Status
                    </h1>
                    <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                        Monitor model performance, view metrics, and trigger retraining
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                    <button className="btn btn-secondary" onClick={loadModelStatus}>
                        <RefreshCw size={18} />
                        Refresh
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={handleRetrain}
                        disabled={retraining}
                    >
                        {retraining ? (
                            <>
                                <span className="spinner" />
                                Triggering...
                            </>
                        ) : (
                            <>
                                <Play size={18} />
                                Trigger Retrain
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Status Banner */}
            {!loading && modelStatus && (
                <div
                    className="card"
                    style={{
                        marginBottom: 'var(--space-6)',
                        background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%)',
                        borderColor: 'rgba(16, 185, 129, 0.3)'
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
                        <div
                            style={{
                                width: 56,
                                height: 56,
                                borderRadius: 'var(--radius-lg)',
                                background: 'rgba(16, 185, 129, 0.2)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}
                        >
                            <CheckCircle size={28} style={{ color: 'var(--accent-success)' }} />
                        </div>
                        <div style={{ flex: 1 }}>
                            <h3 style={{ marginBottom: 'var(--space-1)', color: 'var(--accent-success)' }}>
                                Model is Healthy
                            </h3>
                            <p style={{ marginBottom: 0, color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                                Current Version: <strong>{modelStatus.currentVersion}</strong> •
                                Last trained: {formatDate(modelStatus.lastTrained)} •
                                Next scheduled: {formatDate(modelStatus.nextScheduledTraining)}
                            </p>
                        </div>
                        <div className="badge badge-success" style={{ padding: 'var(--space-2) var(--space-4)' }}>
                            <Activity size={14} style={{ marginRight: 'var(--space-1)' }} />
                            Active
                        </div>
                    </div>
                </div>
            )}

            {/* Performance Metrics */}
            <div
                className="grid grid-cols-4 gap-4"
                style={{ marginBottom: 'var(--space-6)' }}
            >
                {loading ? (
                    Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="kpi-card">
                            <div className="skeleton" style={{ width: 48, height: 48, marginBottom: 'var(--space-3)' }} />
                            <div className="skeleton" style={{ width: '60%', height: 28, marginBottom: 'var(--space-2)' }} />
                            <div className="skeleton" style={{ width: '80%', height: 14 }} />
                        </div>
                    ))
                ) : (
                    <>
                        <div className="kpi-card success">
                            <div className="kpi-icon success">
                                <TrendingUp size={24} />
                            </div>
                            <div className="kpi-value">{modelStatus?.metrics?.mape}%</div>
                            <div className="kpi-label">MAPE</div>
                        </div>
                        <div className="kpi-card info">
                            <div className="kpi-icon info">
                                <Activity size={24} />
                            </div>
                            <div className="kpi-value">{modelStatus?.metrics?.mae}</div>
                            <div className="kpi-label">MAE</div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-icon primary">
                                <Layers size={24} />
                            </div>
                            <div className="kpi-value">{modelStatus?.metrics?.rmse}</div>
                            <div className="kpi-label">RMSE</div>
                        </div>
                        <div className="kpi-card success">
                            <div className="kpi-icon success">
                                <CheckCircle size={24} />
                            </div>
                            <div className="kpi-value">{modelStatus?.metrics?.r2}</div>
                            <div className="kpi-label">R² Score</div>
                        </div>
                    </>
                )}
            </div>

            {/* Charts Row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)', marginBottom: 'var(--space-6)' }}>
                {/* Feature Importance */}
                <BarChart
                    data={featureData}
                    bars={[{ dataKey: 'importance', name: 'Importance', color: CHART_COLORS.primary }]}
                    xKey="name"
                    height={280}
                    horizontal
                    showLegend={false}
                    title="Feature Importance"
                    loading={loading}
                />

                {/* Training History */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Training History</h3>
                    </div>
                    <div className="table-container" style={{ background: 'transparent', border: 'none' }}>
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>Version</th>
                                    <th>Date</th>
                                    <th style={{ textAlign: 'right' }}>MAPE</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {loading ? (
                                    Array.from({ length: 5 }).map((_, i) => (
                                        <tr key={i}>
                                            {Array.from({ length: 4 }).map((_, j) => (
                                                <td key={j}><div className="skeleton" style={{ height: 16 }} /></td>
                                            ))}
                                        </tr>
                                    ))
                                ) : (
                                    modelStatus?.trainingHistory?.slice(0, 8).map((h, i) => (
                                        <tr key={i}>
                                            <td>
                                                <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem' }}>
                                                    v{h.version}
                                                </code>
                                            </td>
                                            <td style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                                {formatDate(h.date)}
                                            </td>
                                            <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                                                {h.mape}%
                                            </td>
                                            <td>
                                                <span className={`badge ${h.status === 'active' ? 'badge-success' : 'badge-primary'}`}>
                                                    {h.status}
                                                </span>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Info Alerts */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                <div className="alert alert-info">
                    <Clock size={18} />
                    <div>
                        <strong>Scheduled Retraining:</strong> Models are automatically retrained weekly
                        with the latest data. Manual retraining can be triggered when needed.
                    </div>
                </div>
                <div className="alert alert-warning">
                    <AlertTriangle size={18} />
                    <div>
                        <strong>Model Monitoring:</strong> If MAPE increases by more than 10%,
                        consider triggering an immediate retrain or reviewing data quality.
                    </div>
                </div>
            </div>
        </div>
    )
}
