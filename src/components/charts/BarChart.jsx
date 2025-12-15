import {
    BarChart as RechartsBarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Cell
} from 'recharts'
import { CHART_COLORS } from '../../utils/constants'
import { formatNumber } from '../../utils/formatters'

export default function BarChart({
    data,
    bars = [], // Array of { dataKey, name, color }
    xKey = 'name',
    height = 300,
    showGrid = true,
    showLegend = true,
    horizontal = false,
    stacked = false,
    loading = false,
    title = null,
    colorByValue = false, // Color bars based on value thresholds
    thresholds = { good: 10, warning: 20 }
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

    const getBarColor = (value) => {
        if (!colorByValue) return CHART_COLORS.primary
        if (value <= thresholds.good) return CHART_COLORS.success
        if (value <= thresholds.warning) return CHART_COLORS.warning
        return CHART_COLORS.danger
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
                    boxShadow: 'var(--shadow-lg)'
                }}
            >
                <div style={{
                    fontSize: '0.8125rem',
                    color: 'var(--text-primary)',
                    marginBottom: 'var(--space-2)',
                    fontWeight: 600
                }}>
                    {label}
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
                                    borderRadius: 2,
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

    const ChartComponent = horizontal ? (
        <RechartsBarChart
            data={data}
            layout="vertical"
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
            {showGrid && (
                <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="var(--border-secondary)"
                    horizontal={false}
                />
            )}
            <XAxis
                type="number"
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={formatNumber}
            />
            <YAxis
                type="category"
                dataKey={xKey}
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                width={100}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            {bars.map((bar, index) => (
                <Bar
                    key={bar.dataKey}
                    dataKey={bar.dataKey}
                    name={bar.name || bar.dataKey}
                    fill={bar.color || Object.values(CHART_COLORS)[index]}
                    radius={[0, 4, 4, 0]}
                    stackId={stacked ? 'stack' : undefined}
                >
                    {colorByValue && data.map((entry, i) => (
                        <Cell key={i} fill={getBarColor(entry[bar.dataKey])} />
                    ))}
                </Bar>
            ))}
        </RechartsBarChart>
    ) : (
        <RechartsBarChart
            data={data}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
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
                tickMargin={10}
            />
            <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={formatNumber}
                width={50}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            {bars.map((bar, index) => (
                <Bar
                    key={bar.dataKey}
                    dataKey={bar.dataKey}
                    name={bar.name || bar.dataKey}
                    fill={bar.color || Object.values(CHART_COLORS)[index]}
                    radius={[4, 4, 0, 0]}
                    stackId={stacked ? 'stack' : undefined}
                >
                    {colorByValue && data.map((entry, i) => (
                        <Cell key={i} fill={getBarColor(entry[bar.dataKey])} />
                    ))}
                </Bar>
            ))}
        </RechartsBarChart>
    )

    return (
        <div className="card">
            {title && (
                <div className="card-header">
                    <h3 className="card-title">{title}</h3>
                </div>
            )}
            <ResponsiveContainer width="100%" height={height}>
                {ChartComponent}
            </ResponsiveContainer>
        </div>
    )
}
