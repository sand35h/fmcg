import { useState, useEffect } from 'react'
import {
    GitBranch,
    Play,
    Save,
    Trash2,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    Info
} from 'lucide-react'
import api from '../services/api'
import LineChart from '../components/charts/LineChart'
import { formatNumber, formatPercent } from '../utils/formatters'
import { CHART_COLORS } from '../utils/constants'

export default function ScenarioPlanning() {
    const [loading, setLoading] = useState(false)
    const [scenarioName, setScenarioName] = useState('New Scenario')
    const [results, setResults] = useState(null)

    // Scenario parameters
    const [params, setParams] = useState({
        demandChange: 0,
        priceChange: 0,
        promoIntensity: 0,
        leadTimeChange: 0,
        festivalImpact: 100
    })

    const handleParamChange = (key, value) => {
        setParams(prev => ({ ...prev, [key]: Number(value) }))
    }

    const runScenario = async () => {
        setLoading(true)
        try {
            const result = await api.runScenario(params)
            setResults(result)
        } catch (error) {
            console.error('Error running scenario:', error)
        }
        setLoading(false)
    }

    const resetParams = () => {
        setParams({
            demandChange: 0,
            priceChange: 0,
            promoIntensity: 0,
            leadTimeChange: 0,
            festivalImpact: 100
        })
        setResults(null)
    }

    // Saved scenarios (mock)
    const savedScenarios = [
        { id: 1, name: 'Christmas Period +150%', params: { demandChange: 150, festivalImpact: 150 } },
        { id: 2, name: 'Price Increase 10%', params: { priceChange: 10, demandChange: -5 } },
        { id: 3, name: 'Supply Chain Disruption', params: { leadTimeChange: 50, demandChange: -10 } },
        { id: 4, name: 'Aggressive Promotion', params: { promoIntensity: 30, demandChange: 25 } }
    ]

    // Prepare chart data
    const chartData = results ? results.baseline.map((base, i) => ({
        date: base.date,
        baseline: base.forecast,
        scenario: results.scenario[i]?.forecast || 0
    })) : []

    return (
        <div className="animate-fadeIn">
            {/* Page Header */}
            <div style={{ marginBottom: 'var(--space-6)' }}>
                <h1 style={{ marginBottom: 'var(--space-2)' }}>
                    <GitBranch size={28} style={{ marginRight: 'var(--space-3)', verticalAlign: 'middle' }} />
                    What-If Scenario Planning
                </h1>
                <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                    Simulate different scenarios and see their impact on demand forecasts
                </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 'var(--space-6)' }}>
                {/* Parameters Panel */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                    {/* Scenario Name */}
                    <div className="card">
                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label">Scenario Name</label>
                            <input
                                type="text"
                                className="form-input"
                                value={scenarioName}
                                onChange={(e) => setScenarioName(e.target.value)}
                                placeholder="Enter scenario name"
                            />
                        </div>
                    </div>

                    {/* Parameters */}
                    <div className="card">
                        <h3 style={{ fontSize: '0.9375rem', marginBottom: 'var(--space-4)' }}>
                            Scenario Parameters
                        </h3>

                        {/* Demand Change */}
                        <div className="form-group">
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                                <label className="form-label" style={{ marginBottom: 0 }}>Demand Change</label>
                                <span style={{
                                    color: params.demandChange >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)',
                                    fontWeight: 600,
                                    fontSize: '0.875rem'
                                }}>
                                    {params.demandChange > 0 ? '+' : ''}{params.demandChange}%
                                </span>
                            </div>
                            <input
                                type="range"
                                min="-50"
                                max="200"
                                value={params.demandChange}
                                onChange={(e) => handleParamChange('demandChange', e.target.value)}
                                style={{ width: '100%', accentColor: 'var(--primary-500)' }}
                            />
                        </div>

                        {/* Price Change */}
                        <div className="form-group">
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                                <label className="form-label" style={{ marginBottom: 0 }}>Price Change</label>
                                <span style={{
                                    color: params.priceChange >= 0 ? 'var(--accent-warning)' : 'var(--accent-success)',
                                    fontWeight: 600,
                                    fontSize: '0.875rem'
                                }}>
                                    {params.priceChange > 0 ? '+' : ''}{params.priceChange}%
                                </span>
                            </div>
                            <input
                                type="range"
                                min="-30"
                                max="50"
                                value={params.priceChange}
                                onChange={(e) => handleParamChange('priceChange', e.target.value)}
                                style={{ width: '100%', accentColor: 'var(--primary-500)' }}
                            />
                        </div>

                        {/* Promotion Intensity */}
                        <div className="form-group">
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                                <label className="form-label" style={{ marginBottom: 0 }}>Promotion Intensity</label>
                                <span style={{
                                    color: 'var(--accent-purple)',
                                    fontWeight: 600,
                                    fontSize: '0.875rem'
                                }}>
                                    {params.promoIntensity}%
                                </span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="50"
                                value={params.promoIntensity}
                                onChange={(e) => handleParamChange('promoIntensity', e.target.value)}
                                style={{ width: '100%', accentColor: 'var(--primary-500)' }}
                            />
                        </div>

                        {/* Lead Time Change */}
                        <div className="form-group">
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                                <label className="form-label" style={{ marginBottom: 0 }}>Lead Time Change</label>
                                <span style={{
                                    color: params.leadTimeChange > 0 ? 'var(--accent-danger)' : 'var(--accent-success)',
                                    fontWeight: 600,
                                    fontSize: '0.875rem'
                                }}>
                                    {params.leadTimeChange > 0 ? '+' : ''}{params.leadTimeChange}%
                                </span>
                            </div>
                            <input
                                type="range"
                                min="-30"
                                max="100"
                                value={params.leadTimeChange}
                                onChange={(e) => handleParamChange('leadTimeChange', e.target.value)}
                                style={{ width: '100%', accentColor: 'var(--primary-500)' }}
                            />
                        </div>

                        {/* Festival Impact */}
                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                                <label className="form-label" style={{ marginBottom: 0 }}>Festival Impact</label>
                                <span style={{
                                    color: 'var(--accent-warning)',
                                    fontWeight: 600,
                                    fontSize: '0.875rem'
                                }}>
                                    {params.festivalImpact}%
                                </span>
                            </div>
                            <input
                                type="range"
                                min="50"
                                max="300"
                                value={params.festivalImpact}
                                onChange={(e) => handleParamChange('festivalImpact', e.target.value)}
                                style={{ width: '100%', accentColor: 'var(--primary-500)' }}
                            />
                        </div>
                    </div>

                    {/* Actions */}
                    <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                        <button
                            className="btn btn-primary"
                            style={{ flex: 1 }}
                            onClick={runScenario}
                            disabled={loading}
                        >
                            {loading ? (
                                <>
                                    <span className="spinner" />
                                    Running...
                                </>
                            ) : (
                                <>
                                    <Play size={18} />
                                    Run Scenario
                                </>
                            )}
                        </button>
                        <button
                            className="btn btn-secondary btn-icon"
                            onClick={resetParams}
                            title="Reset"
                        >
                            <Trash2 size={18} />
                        </button>
                    </div>

                    {/* Saved Scenarios */}
                    <div className="card">
                        <h3 style={{ fontSize: '0.9375rem', marginBottom: 'var(--space-3)' }}>
                            Saved Scenarios
                        </h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                            {savedScenarios.map(scenario => (
                                <button
                                    key={scenario.id}
                                    className="btn btn-ghost"
                                    style={{ justifyContent: 'flex-start', textAlign: 'left' }}
                                    onClick={() => {
                                        setScenarioName(scenario.name)
                                        setParams(prev => ({ ...prev, ...scenario.params }))
                                    }}
                                >
                                    <GitBranch size={16} />
                                    <span style={{
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap'
                                    }}>
                                        {scenario.name}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Results Panel */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
                    {!results ? (
                        <div
                            className="card"
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                minHeight: 400,
                                textAlign: 'center'
                            }}
                        >
                            <GitBranch size={48} style={{ color: 'var(--text-muted)', marginBottom: 'var(--space-4)' }} />
                            <h3 style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
                                No Scenario Results Yet
                            </h3>
                            <p style={{ color: 'var(--text-muted)', maxWidth: 400 }}>
                                Adjust the parameters on the left and click "Run Scenario" to see the impact on demand forecasts.
                            </p>
                        </div>
                    ) : (
                        <>
                            {/* Impact Summary */}
                            <div className="grid grid-cols-3 gap-4">
                                <div className="kpi-card success">
                                    <div className="kpi-value">
                                        {results.impact.demandChange > 0 ? '+' : ''}{results.impact.demandChange}%
                                    </div>
                                    <div className="kpi-label">Demand Impact</div>
                                </div>
                                <div className="kpi-card info">
                                    <div className="kpi-value">
                                        {results.impact.revenueImpact > 0 ? '+' : ''}{results.impact.revenueImpact}%
                                    </div>
                                    <div className="kpi-label">Revenue Impact</div>
                                </div>
                                <div className={`kpi-card ${results.impact.stockoutRisk === 'High' ? 'danger' : results.impact.stockoutRisk === 'Medium' ? 'warning' : 'success'}`}>
                                    <div className="kpi-value">{results.impact.stockoutRisk}</div>
                                    <div className="kpi-label">Stockout Risk</div>
                                </div>
                            </div>

                            {/* Comparison Chart */}
                            <LineChart
                                data={chartData}
                                lines={[
                                    { dataKey: 'baseline', name: 'Baseline Forecast', color: CHART_COLORS.muted },
                                    { dataKey: 'scenario', name: `Scenario: ${scenarioName}`, color: CHART_COLORS.primary }
                                ]}
                                xKey="date"
                                height={300}
                                title="Baseline vs Scenario Comparison"
                            />

                            {/* Warnings */}
                            {results.impact.stockoutRisk === 'High' && (
                                <div className="alert alert-danger">
                                    <AlertTriangle size={18} />
                                    <div>
                                        <strong>High Stockout Risk:</strong> The demand increase in this scenario may exceed current
                                        inventory capacity. Consider increasing safety stock levels or expediting replenishment.
                                    </div>
                                </div>
                            )}

                            {results.impact.stockoutRisk === 'Medium' && (
                                <div className="alert alert-warning">
                                    <AlertTriangle size={18} />
                                    <div>
                                        <strong>Moderate Risk:</strong> Some SKUs may face stock constraints under this scenario.
                                        Review reorder points for high-volume items.
                                    </div>
                                </div>
                            )}

                            <div className="alert alert-info">
                                <Info size={18} />
                                <div>
                                    <strong>Scenario Analysis:</strong> This simulation shows a {Math.abs(params.demandChange)}%
                                    {params.demandChange >= 0 ? ' increase' : ' decrease'} in demand.
                                    Adjust parameters to explore different scenarios.
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
