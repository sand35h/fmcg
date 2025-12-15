import {
    LineChart as RechartsLineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Area,
    ComposedChart
} from 'recharts'
import { CHART_COLORS } from '../../utils/constants'
import { formatNumber, formatDate } from '../../utils/formatters'

export default function LineChart({
    data,
    lines = [], // Array of { dataKey, name, color, type }
    xKey = 'date',
    height = 300,
    showGrid = true,
    showLegend = true,
    showArea = false,
    areaKey = null,
    lowerBoundKey = null,
    upperBoundKey = null,
    loading = false,
    title = null
}) {
    if (loading) {
        return (
            <div className="card" style={{ height: height + 60 }}>
                {title && (
                    <div className="card-header">
                        <div className="skeleton" style={{ width: 150, height: 20 }} />
                    </div>
                )}
                <div className="skeleton" style={{ width: '100%', height }} />
            </div>
        )
    }

    const CustomTooltip = ({ active, payload, label }) => {
        if (!active || !payload?.length) return null

        return (
            <div
                style={{
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border-primary)',
                    borderRadius: 'var(--radius-lg)',
                    padding: 'var(--space-3) var(--space-4)',
                    boxShadow: 'var(--shadow-lg)',
                    minWidth: 150
                }}
            >
                <div style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-muted)',
                    marginBottom: 'var(--space-2)',
                    fontWeight: 500
                }}>
                    {formatDate(label)}
                </div>
                {payload.map((entry, index) => (
                    <div
                        key={index}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: 'var(--space-4)',
                            marginBottom: 'var(--space-1)',
                            fontSize: '0.8125rem'
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                            <div
                                style={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    background: entry.color
                                }}
                            />
                            <span style={{ color: 'var(--text-secondary)' }}>{entry.name}</span>
                        </div>
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                            {formatNumber(entry.value)}
                        </span>
                    </div>
                ))}
            </div>
        )
    }

    return (
        <div className="card">
            {title && (
                <div className="card-header">
                    <h3 className="card-title">{title}</h3>
                </div>
            )}
            <ResponsiveContainer width="100%" height={height}>
                <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    {showGrid && (
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="var(--border-secondary)"
                            vertical={false}
                        />
                    )}
                    <XAxis
                        dataKey={xKey}
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        tickFormatter={(val) => formatDate(val, 'MMM dd')}
                        tickMargin={10}
                    />
                    <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        tickFormatter={formatNumber}
                        tickMargin={10}
                        width={50}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    {showLegend && (
                        <Legend
                            wrapperStyle={{ paddingTop: 20 }}
                            formatter={(value) => (
                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8125rem' }}>
                                    {value}
                                </span>
                            )}
                        />
                    )}

                    {/* Confidence Interval Area */}
                    {lowerBoundKey && upperBoundKey && (
                        <Area
                            type="monotone"
                            dataKey={upperBoundKey}
                            stroke="none"
                            fill={CHART_COLORS.primary}
                            fillOpacity={0.1}
                            name="Upper Bound"
                            legendType="none"
                        />
                    )}

                    {/* Main Lines */}
                    {lines.map((line, index) => (
                        <Line
                            key={line.dataKey}
                            type="monotone"
                            dataKey={line.dataKey}
                            name={line.name || line.dataKey}
                            stroke={line.color || Object.values(CHART_COLORS)[index]}
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--bg-tertiary)' }}
                            strokeDasharray={line.dashed ? '5 5' : undefined}
                        />
                    ))}

                    {showArea && areaKey && (
                        <Area
                            type="monotone"
                            dataKey={areaKey}
                            stroke="none"
                            fill={CHART_COLORS.primary}
                            fillOpacity={0.2}
                            legendType="none"
                        />
                    )}
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    )
}
