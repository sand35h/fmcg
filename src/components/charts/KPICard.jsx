import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { formatNumber, formatPercent, formatChange, getChangeType } from '../../utils/formatters'

export default function KPICard({
    title,
    value,
    change,
    changeLabel = 'vs last period',
    icon: Icon,
    variant = 'primary', // primary, success, warning, danger, info
    format = 'number', // number, percent, currency
    prefix = '',
    suffix = '',
    loading = false
}) {
    const changeType = getChangeType(change)

    const formatValue = (val) => {
        if (format === 'percent') return formatPercent(val)
        if (format === 'currency') return `${prefix}${formatNumber(val)}${suffix}`
        return `${prefix}${formatNumber(val)}${suffix}`
    }

    if (loading) {
        return (
            <div className={`kpi-card ${variant}`}>
                <div className="skeleton" style={{ width: 48, height: 48, marginBottom: 'var(--space-3)' }} />
                <div className="skeleton" style={{ width: '60%', height: 28, marginBottom: 'var(--space-2)' }} />
                <div className="skeleton" style={{ width: '80%', height: 14, marginBottom: 'var(--space-2)' }} />
                <div className="skeleton" style={{ width: '40%', height: 20 }} />
            </div>
        )
    }

    return (
        <div className={`kpi-card ${variant}`}>
            {Icon && (
                <div className={`kpi-icon ${variant}`}>
                    <Icon size={24} />
                </div>
            )}

            <div className="kpi-value">{formatValue(value)}</div>
            <div className="kpi-label">{title}</div>

            {change !== undefined && change !== null && (
                <div className={`kpi-change ${changeType}`}>
                    {changeType === 'positive' && <TrendingUp size={14} />}
                    {changeType === 'negative' && <TrendingDown size={14} />}
                    {changeType === 'neutral' && <Minus size={14} />}
                    <span>{formatChange(change)}</span>
                    {changeLabel && (
                        <span style={{ fontWeight: 400, opacity: 0.8, marginLeft: 4 }}>
                            {changeLabel}
                        </span>
                    )}
                </div>
            )}
        </div>
    )
}
