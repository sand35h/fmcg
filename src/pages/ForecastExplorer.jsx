import { useState, useEffect } from 'react'
import {
    Filter,
    Download,
    Calendar,
    MapPin,
    Package,
    TrendingUp,
    Info,
    RefreshCw
} from 'lucide-react'
import api from '../services/api'
import LineChart from '../components/charts/LineChart'
import { formatNumber, formatDate, formatPercent } from '../utils/formatters'
import { CATEGORIES, REGIONS, TIME_HORIZONS, CHART_COLORS } from '../utils/constants'

export default function ForecastExplorer() {
    const [loading, setLoading] = useState(true)
    const [forecasts, setForecasts] = useState([])
    const [skus, setSkus] = useState([])
    const [selectedSku, setSelectedSku] = useState('')
    const [selectedCategory, setSelectedCategory] = useState('')
    const [selectedRegion, setSelectedRegion] = useState('')
    const [timeHorizon, setTimeHorizon] = useState(30)
    const [showFilters, setShowFilters] = useState(true)

    useEffect(() => {
        loadForecastData()
    }, [timeHorizon])

    const loadForecastData = async () => {
        setLoading(true)
        try {
            const data = await api.getForecasts({ days: timeHorizon })
            setForecasts(data.data)
            setSkus(data.skus)
            if (!selectedSku && data.skus.length > 0) {
                setSelectedSku(data.skus[0].id)
            }
        } catch (error) {
            console.error('Error loading forecasts:', error)
        }
        setLoading(false)
    }

    const handleExport = async () => {
        try {
            const result = await api.exportData('csv')
            alert(result.message)
        } catch (error) {
            console.error('Export error:', error)
        }
    }

    // Filter SKUs
    const filteredSkus = skus.filter(sku => {
        if (selectedCategory && sku.category !== selectedCategory) return false
        return true
    })

    // Get selected SKU details
    const selectedSkuDetails = skus.find(s => s.id === selectedSku)

    // Calculate summary stats
    const summaryStats = {
        avgForecast: forecasts.length > 0
            ? Math.round(forecasts.reduce((sum, d) => sum + d.forecast, 0) / forecasts.length)
            : 0,
        totalForecast: forecasts.reduce((sum, d) => sum + d.forecast, 0),
        avgActual: forecasts.filter(d => d.actual).length > 0
            ? Math.round(forecasts.filter(d => d.actual).reduce((sum, d) => sum + d.actual, 0) / forecasts.filter(d => d.actual).length)
            : 0,
        accuracy: 92.3
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
                    <h1 style={{ marginBottom: 'var(--space-2)' }}>Forecast Explorer</h1>
                    <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                        Explore demand forecasts by SKU, location, and time period
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                    <button
                        className="btn btn-secondary"
                        onClick={() => setShowFilters(!showFilters)}
                    >
                        <Filter size={18} />
                        {showFilters ? 'Hide Filters' : 'Show Filters'}
                    </button>
                    <button className="btn btn-secondary" onClick={handleExport}>
                        <Download size={18} />
                        Export
                    </button>
                    <button className="btn btn-primary" onClick={loadForecastData}>
                        <RefreshCw size={18} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Filters */}
            {showFilters && (
                <div
                    className="card animate-slideUp"
                    style={{ marginBottom: 'var(--space-6)', padding: 'var(--space-5)' }}
                >
                    <div
                        style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(4, 1fr)',
                            gap: 'var(--space-4)'
                        }}
                    >
                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label">
                                <Package size={14} style={{ marginRight: 'var(--space-1)' }} />
                                SKU
                            </label>
                            <select
                                className="form-select"
                                value={selectedSku}
                                onChange={(e) => setSelectedSku(e.target.value)}
                            >
                                <option value="">All SKUs</option>
                                {filteredSkus.map(sku => (
                                    <option key={sku.id} value={sku.id}>{sku.name}</option>
                                ))}
                            </select>
                        </div>

                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label">
                                <Filter size={14} style={{ marginRight: 'var(--space-1)' }} />
                                Category
                            </label>
                            <select
                                className="form-select"
                                value={selectedCategory}
                                onChange={(e) => setSelectedCategory(e.target.value)}
                            >
                                <option value="">All Categories</option>
                                {CATEGORIES.map(cat => (
                                    <option key={cat} value={cat}>{cat}</option>
                                ))}
                            </select>
                        </div>

                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label">
                                <MapPin size={14} style={{ marginRight: 'var(--space-1)' }} />
                                Region
                            </label>
                            <select
                                className="form-select"
                                value={selectedRegion}
                                onChange={(e) => setSelectedRegion(e.target.value)}
                            >
                                <option value="">All Regions</option>
                                {REGIONS.map(region => (
                                    <option key={region} value={region}>{region}</option>
                                ))}
                            </select>
                        </div>

                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label">
                                <Calendar size={14} style={{ marginRight: 'var(--space-1)' }} />
                                Time Horizon
                            </label>
                            <select
                                className="form-select"
                                value={timeHorizon}
                                onChange={(e) => setTimeHorizon(Number(e.target.value))}
                            >
                                {TIME_HORIZONS.map(th => (
                                    <option key={th.value} value={th.value}>{th.label}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>
            )}

            {/* Summary Stats */}
            <div
                className="grid grid-cols-4 gap-4"
                style={{ marginBottom: 'var(--space-6)' }}
            >
                <div className="kpi-card info">
                    <div className="kpi-value">{formatNumber(summaryStats.avgForecast)}</div>
                    <div className="kpi-label">Avg Daily Forecast</div>
                </div>
                <div className="kpi-card">
                    <div className="kpi-value">{formatNumber(summaryStats.totalForecast)}</div>
                    <div className="kpi-label">Total Forecast ({timeHorizon}d)</div>
                </div>
                <div className="kpi-card">
                    <div className="kpi-value">{formatNumber(summaryStats.avgActual)}</div>
                    <div className="kpi-label">Avg Actual Demand</div>
                </div>
                <div className="kpi-card success">
                    <div className="kpi-value">{summaryStats.accuracy}%</div>
                    <div className="kpi-label">Forecast Accuracy</div>
                </div>
            </div>

            {/* Main Content */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 'var(--space-6)' }}>
                {/* Forecast Chart */}
                <div>
                    <LineChart
                        data={forecasts}
                        lines={[
                            { dataKey: 'actual', name: 'Actual', color: CHART_COLORS.primary },
                            { dataKey: 'forecast', name: 'Forecast', color: CHART_COLORS.success, dashed: true },
                        ]}
                        lowerBoundKey="lowerBound"
                        upperBoundKey="upperBound"
                        xKey="date"
                        height={350}
                        title={selectedSkuDetails ? `Forecast: ${selectedSkuDetails.name}` : 'Demand Forecast'}
                        loading={loading}
                    />

                    {/* Confidence Interval Info */}
                    <div
                        className="alert alert-info"
                        style={{ marginTop: 'var(--space-4)' }}
                    >
                        <Info size={18} />
                        <div>
                            <strong>Confidence Intervals:</strong> The shaded area represents the 80% prediction interval.
                            Actual demand is expected to fall within this range 80% of the time.
                        </div>
                    </div>
                </div>

                {/* Side Panel - Feature Importance */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Forecast Drivers</h3>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                        {[
                            { feature: 'Historical Demand', importance: 0.28, color: CHART_COLORS.primary },
                            { feature: 'Festival Effects', importance: 0.22, color: CHART_COLORS.warning },
                            { feature: 'Price Changes', importance: 0.18, color: CHART_COLORS.success },
                            { feature: 'Weather Pattern', importance: 0.14, color: CHART_COLORS.cyan },
                            { feature: 'Promotions', importance: 0.10, color: CHART_COLORS.purple },
                            { feature: 'Day of Week', importance: 0.08, color: CHART_COLORS.muted }
                        ].map((item, index) => (
                            <div key={index}>
                                <div
                                    style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        marginBottom: 'var(--space-1)',
                                        fontSize: '0.8125rem'
                                    }}
                                >
                                    <span style={{ color: 'var(--text-primary)' }}>{item.feature}</span>
                                    <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>
                                        {formatPercent(item.importance * 100, 0)}
                                    </span>
                                </div>
                                <div className="progress">
                                    <div
                                        className="progress-bar"
                                        style={{
                                            width: `${item.importance * 100}%`,
                                            background: item.color
                                        }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>

                    <div
                        style={{
                            marginTop: 'var(--space-6)',
                            paddingTop: 'var(--space-4)',
                            borderTop: '1px solid var(--border-secondary)'
                        }}
                    >
                        <h4 style={{ fontSize: '0.875rem', marginBottom: 'var(--space-3)' }}>
                            Key Insights
                        </h4>
                        <ul style={{
                            listStyle: 'none',
                            fontSize: '0.8125rem',
                            color: 'var(--text-secondary)',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 'var(--space-2)'
                        }}>
                            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-2)' }}>
                                <TrendingUp size={14} style={{ color: 'var(--accent-success)', marginTop: 2 }} />
                                Christmas period expected to increase demand by 150%
                            </li>
                            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-2)' }}>
                                <TrendingUp size={14} style={{ color: 'var(--accent-info)', marginTop: 2 }} />
                                Weekend demand typically 20% higher
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    )
}
