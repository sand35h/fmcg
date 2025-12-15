import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
    TrendingUp,
    TrendingDown,
    Package,
    AlertTriangle,
    Target,
    BarChart3,
    RefreshCw,
    ArrowRight,
    CheckCircle,
    Clock,
    AlertCircle,
    XCircle
} from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import api from '../services/api'
import KPICard from '../components/charts/KPICard'
import LineChart from '../components/charts/LineChart'
import { formatRelativeTime, formatDate } from '../utils/formatters'
import { CHART_COLORS } from '../utils/constants'

export default function Dashboard() {
    const { user } = useAuth()
    const [kpis, setKpis] = useState(null)
    const [demandTrend, setDemandTrend] = useState([])
    const [alerts, setAlerts] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadDashboardData()
    }, [])

    const loadDashboardData = async () => {
        setLoading(true)
        try {
            const [kpiData, trendData, alertData] = await Promise.all([
                api.getDashboardKPIs(),
                api.getDemandTrend(60),
                api.getAlerts()
            ])
            setKpis(kpiData)
            setDemandTrend(trendData)
            setAlerts(alertData)
        } catch (error) {
            console.error('Error loading dashboard:', error)
        }
        setLoading(false)
    }

    const getAlertIcon = (type) => {
        const icons = {
            stockout: <AlertTriangle size={16} />,
            demand_spike: <TrendingUp size={16} />,
            model_drift: <RefreshCw size={16} />,
            data_quality: <AlertCircle size={16} />,
            forecast: <CheckCircle size={16} />,
            promotion: <Target size={16} />
        }
        return icons[type] || <AlertCircle size={16} />
    }

    const getAlertColor = (severity) => {
        const colors = {
            success: 'var(--accent-success)',
            warning: 'var(--accent-warning)',
            danger: 'var(--accent-danger)',
            info: 'var(--accent-info)'
        }
        return colors[severity] || colors.info
    }

    // Top SKUs mock data
    const topSkus = [
        { name: 'Brand_A_DAIRY_05', category: 'DAIRY', demand: 12450, change: 15.2 },
        { name: 'Brand_B_BEV_12', category: 'BEVERAGES', demand: 10830, change: 8.7 },
        { name: 'Brand_C_SNACK_08', category: 'SNACKS', demand: 9560, change: -3.2 },
        { name: 'Brand_A_BEV_03', category: 'BEVERAGES', demand: 8920, change: 22.1 },
        { name: 'Brand_D_DAIRY_01', category: 'DAIRY', demand: 7650, change: 5.4 }
    ]

    return (
        <div className="animate-fadeIn">
            {/* Page Header */}
            <div style={{ marginBottom: 'var(--space-6)' }}>
                <h1 style={{ marginBottom: 'var(--space-2)' }}>
                    Welcome back, {user?.name?.split(' ')[0] || 'User'}!
                </h1>
                <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                    Here's what's happening with your supply chain today.
                </p>
            </div>

            {/* KPI Cards */}
            <div
                className="grid grid-cols-4 gap-4 mb-6"
                style={{ marginBottom: 'var(--space-6)' }}
            >
                <KPICard
                    title="Forecast MAPE"
                    value={kpis?.mape?.value || 0}
                    change={kpis?.mape?.change}
                    changeLabel="vs baseline"
                    icon={Target}
                    variant="success"
                    suffix="%"
                    loading={loading}
                />
                <KPICard
                    title="Stockout Rate"
                    value={kpis?.stockoutRate?.value || 0}
                    change={kpis?.stockoutRate?.change}
                    changeLabel="vs last month"
                    icon={Package}
                    variant="warning"
                    suffix="%"
                    loading={loading}
                />
                <KPICard
                    title="Inventory Turns"
                    value={kpis?.inventoryTurnover?.value || 0}
                    change={kpis?.inventoryTurnover?.change}
                    changeLabel="vs last quarter"
                    icon={RefreshCw}
                    variant="info"
                    suffix="x"
                    loading={loading}
                />
                <KPICard
                    title="Service Level"
                    value={kpis?.serviceLevel?.value || 0}
                    change={kpis?.serviceLevel?.change}
                    changeLabel="OTIF"
                    icon={CheckCircle}
                    variant="success"
                    suffix="%"
                    loading={loading}
                />
            </div>

            {/* Main Content Grid */}
            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: '2fr 1fr',
                    gap: 'var(--space-6)'
                }}
            >
                {/* Left Column */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
                    {/* Demand Trend Chart */}
                    <LineChart
                        data={demandTrend}
                        lines={[
                            { dataKey: 'actual', name: 'Actual Demand', color: CHART_COLORS.primary },
                            { dataKey: 'forecast', name: 'Forecast', color: CHART_COLORS.cyan, dashed: true }
                        ]}
                        xKey="date"
                        height={280}
                        title="Demand Trend (Last 60 Days)"
                        loading={loading}
                    />

                    {/* Top SKUs */}
                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">Top Performing SKUs</h3>
                            <Link
                                to="/forecasts"
                                style={{
                                    fontSize: '0.8125rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 'var(--space-1)'
                                }}
                            >
                                View all <ArrowRight size={14} />
                            </Link>
                        </div>
                        <div className="table-container" style={{ background: 'transparent', border: 'none' }}>
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>SKU</th>
                                        <th>Category</th>
                                        <th style={{ textAlign: 'right' }}>Demand</th>
                                        <th style={{ textAlign: 'right' }}>Change</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {topSkus.map((sku, index) => (
                                        <tr key={index}>
                                            <td>
                                                <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                                                    {sku.name}
                                                </span>
                                            </td>
                                            <td>
                                                <span className="badge badge-primary">{sku.category}</span>
                                            </td>
                                            <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                                                {sku.demand.toLocaleString()}
                                            </td>
                                            <td style={{ textAlign: 'right' }}>
                                                <span style={{
                                                    display: 'inline-flex',
                                                    alignItems: 'center',
                                                    gap: 'var(--space-1)',
                                                    color: sku.change >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)',
                                                    fontWeight: 500
                                                }}>
                                                    {sku.change >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                                    {sku.change > 0 ? '+' : ''}{sku.change}%
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Right Column - Alerts & Quick Actions */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
                    {/* Alerts */}
                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">Recent Alerts</h3>
                            <span className="badge badge-danger">{alerts.filter(a => !a.read).length} new</span>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                            {loading ? (
                                Array.from({ length: 4 }).map((_, i) => (
                                    <div key={i} className="skeleton" style={{ height: 60 }} />
                                ))
                            ) : (
                                alerts.slice(0, 5).map((alert) => (
                                    <div
                                        key={alert.id}
                                        style={{
                                            display: 'flex',
                                            gap: 'var(--space-3)',
                                            padding: 'var(--space-3)',
                                            background: alert.read ? 'transparent' : 'rgba(99, 102, 241, 0.05)',
                                            borderRadius: 'var(--radius-lg)',
                                            cursor: 'pointer',
                                            transition: 'background var(--transition-fast)'
                                        }}
                                        onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                                        onMouseLeave={(e) => e.currentTarget.style.background = alert.read ? 'transparent' : 'rgba(99, 102, 241, 0.05)'}
                                    >
                                        <div
                                            style={{
                                                width: 32,
                                                height: 32,
                                                borderRadius: 'var(--radius-md)',
                                                background: `${getAlertColor(alert.severity)}20`,
                                                color: getAlertColor(alert.severity),
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                flexShrink: 0
                                            }}
                                        >
                                            {getAlertIcon(alert.type)}
                                        </div>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <p style={{
                                                fontSize: '0.8125rem',
                                                color: 'var(--text-primary)',
                                                marginBottom: 'var(--space-1)',
                                                lineHeight: 1.4
                                            }}>
                                                {alert.message}
                                            </p>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                                {formatRelativeTime(alert.timestamp)}
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">Quick Actions</h3>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                            <Link to="/forecasts" className="btn btn-secondary" style={{ width: '100%', justifyContent: 'flex-start' }}>
                                <BarChart3 size={18} />
                                View Forecasts
                            </Link>
                            <Link to="/scenarios" className="btn btn-secondary" style={{ width: '100%', justifyContent: 'flex-start' }}>
                                <TrendingUp size={18} />
                                Run What-If Scenario
                            </Link>
                            <Link to="/models" className="btn btn-secondary" style={{ width: '100%', justifyContent: 'flex-start' }}>
                                <RefreshCw size={18} />
                                Trigger Model Retrain
                            </Link>
                            <Link to="/data-jobs" className="btn btn-secondary" style={{ width: '100%', justifyContent: 'flex-start' }}>
                                <Package size={18} />
                                Upload New Data
                            </Link>
                        </div>
                    </div>

                    {/* System Status */}
                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">System Status</h3>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                            {[
                                { name: 'Data Pipeline', status: 'healthy', lastRun: '5 mins ago' },
                                { name: 'Forecast Engine', status: 'healthy', lastRun: '2 hours ago' },
                                { name: 'Model Training', status: 'healthy', lastRun: '3 days ago' }
                            ].map((service, i) => (
                                <div
                                    key={i}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between'
                                    }}
                                >
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                        <div
                                            style={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: 'var(--radius-full)',
                                                background: 'var(--accent-success)'
                                            }}
                                        />
                                        <span style={{ fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                                            {service.name}
                                        </span>
                                    </div>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        {service.lastRun}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
