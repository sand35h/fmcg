import { useState, useEffect, useRef } from 'react'
import {
    Database,
    Upload,
    RefreshCw,
    CheckCircle,
    XCircle,
    Clock,
    Play,
    FileText,
    AlertTriangle
} from 'lucide-react'
import api from '../services/api'
import { formatRelativeTime, formatNumber } from '../utils/formatters'

export default function DataJobs() {
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)
    const [uploading, setUploading] = useState(false)
    const fileInputRef = useRef(null)

    useEffect(() => {
        loadJobs()
    }, [])

    const loadJobs = async () => {
        setLoading(true)
        try {
            const data = await api.getDataJobs()
            setJobs(data)
        } catch (error) {
            console.error('Error loading jobs:', error)
        }
        setLoading(false)
    }

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0]
        if (!file) return

        setUploading(true)
        try {
            const result = await api.uploadData(file)
            if (result.success) {
                alert(result.message)
                loadJobs()
            }
        } catch (error) {
            console.error('Upload error:', error)
            alert('Upload failed. Please try again.')
        }
        setUploading(false)
        if (fileInputRef.current) {
            fileInputRef.current.value = ''
        }
    }

    const getStatusIcon = (status) => {
        const icons = {
            completed: <CheckCircle size={18} style={{ color: 'var(--accent-success)' }} />,
            running: <RefreshCw size={18} style={{ color: 'var(--accent-info)' }} className="animate-spin" />,
            failed: <XCircle size={18} style={{ color: 'var(--accent-danger)' }} />,
            pending: <Clock size={18} style={{ color: 'var(--text-muted)' }} />
        }
        return icons[status] || icons.pending
    }

    const getStatusBadge = (status) => {
        const config = {
            completed: 'badge-success',
            running: 'badge-info',
            failed: 'badge-danger',
            pending: 'badge-primary'
        }
        return <span className={`badge ${config[status] || 'badge-primary'}`}>{status}</span>
    }

    // Stats
    const stats = {
        total: jobs.length,
        running: jobs.filter(j => j.status === 'running').length,
        completed: jobs.filter(j => j.status === 'completed').length,
        failed: jobs.filter(j => j.status === 'failed').length
    }

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
                        <Database size={28} style={{ marginRight: 'var(--space-3)', verticalAlign: 'middle' }} />
                        Data Jobs
                    </h1>
                    <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                        Upload data, monitor ETL pipelines, and track job status
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv,.xlsx,.json"
                        onChange={handleFileUpload}
                        style={{ display: 'none' }}
                    />
                    <button
                        className="btn btn-primary"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={uploading}
                    >
                        {uploading ? (
                            <>
                                <span className="spinner" />
                                Uploading...
                            </>
                        ) : (
                            <>
                                <Upload size={18} />
                                Upload Data
                            </>
                        )}
                    </button>
                    <button className="btn btn-secondary" onClick={loadJobs}>
                        <RefreshCw size={18} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Stats */}
            <div
                className="grid grid-cols-4 gap-4"
                style={{ marginBottom: 'var(--space-6)' }}
            >
                <div className="kpi-card">
                    <div className="kpi-icon primary">
                        <Database size={24} />
                    </div>
                    <div className="kpi-value">{stats.total}</div>
                    <div className="kpi-label">Total Jobs</div>
                </div>
                <div className="kpi-card info">
                    <div className="kpi-icon info">
                        <Play size={24} />
                    </div>
                    <div className="kpi-value">{stats.running}</div>
                    <div className="kpi-label">Running</div>
                </div>
                <div className="kpi-card success">
                    <div className="kpi-icon success">
                        <CheckCircle size={24} />
                    </div>
                    <div className="kpi-value">{stats.completed}</div>
                    <div className="kpi-label">Completed</div>
                </div>
                <div className="kpi-card danger">
                    <div className="kpi-icon danger">
                        <XCircle size={24} />
                    </div>
                    <div className="kpi-value">{stats.failed}</div>
                    <div className="kpi-label">Failed</div>
                </div>
            </div>

            {/* Running Jobs */}
            {jobs.filter(j => j.status === 'running').length > 0 && (
                <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                    <div className="card-header">
                        <h3 className="card-title">
                            <RefreshCw size={18} style={{ marginRight: 'var(--space-2)' }} className="animate-spin" />
                            Running Jobs
                        </h3>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                        {jobs.filter(j => j.status === 'running').map(job => (
                            <div
                                key={job.id}
                                style={{
                                    padding: 'var(--space-4)',
                                    background: 'var(--bg-tertiary)',
                                    borderRadius: 'var(--radius-lg)'
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-3)' }}>
                                    <div>
                                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                            {job.type}
                                        </span>
                                        <span style={{ color: 'var(--text-muted)', marginLeft: 'var(--space-2)', fontSize: '0.8125rem' }}>
                                            {job.id}
                                        </span>
                                    </div>
                                    <span style={{ color: 'var(--accent-info)', fontWeight: 600 }}>
                                        {job.progress}%
                                    </span>
                                </div>
                                <div className="progress" style={{ height: 6 }}>
                                    <div
                                        className="progress-bar primary"
                                        style={{ width: `${job.progress}%` }}
                                    />
                                </div>
                                <div style={{ marginTop: 'var(--space-2)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    Started {formatRelativeTime(job.startTime)} • {formatNumber(job.records)} records
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Job History Table */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Job History</h3>
                </div>
                <div className="table-container" style={{ background: 'transparent', border: 'none' }}>
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Job ID</th>
                                <th>Type</th>
                                <th>Status</th>
                                <th>Start Time</th>
                                <th style={{ textAlign: 'right' }}>Duration</th>
                                <th style={{ textAlign: 'right' }}>Records</th>
                            </tr>
                        </thead>
                        <tbody>
                            {loading ? (
                                Array.from({ length: 8 }).map((_, i) => (
                                    <tr key={i}>
                                        {Array.from({ length: 6 }).map((_, j) => (
                                            <td key={j}><div className="skeleton" style={{ height: 20 }} /></td>
                                        ))}
                                    </tr>
                                ))
                            ) : (
                                jobs.map(job => (
                                    <tr key={job.id}>
                                        <td>
                                            <code style={{
                                                fontFamily: 'var(--font-mono)',
                                                fontSize: '0.8125rem',
                                                color: 'var(--text-secondary)'
                                            }}>
                                                {job.id}
                                            </code>
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                                <FileText size={16} style={{ color: 'var(--text-muted)' }} />
                                                <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                                                    {job.type}
                                                </span>
                                            </div>
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                                {getStatusIcon(job.status)}
                                                {getStatusBadge(job.status)}
                                            </div>
                                        </td>
                                        <td style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                            {formatRelativeTime(job.startTime)}
                                        </td>
                                        <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                                            {job.duration ? `${job.duration}m` : '-'}
                                        </td>
                                        <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                                            {formatNumber(job.records)}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Upload Info */}
            <div
                className="alert alert-info"
                style={{ marginTop: 'var(--space-6)' }}
            >
                <AlertTriangle size={18} />
                <div>
                    <strong>Supported Formats:</strong> CSV, XLSX, JSON.
                    Maximum file size: 100MB.
                    Files will be validated and processed through the ETL pipeline.
                </div>
            </div>
        </div>
    )
}
