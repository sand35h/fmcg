import { format, formatDistanceToNow, parseISO } from 'date-fns'

// Number formatting
export function formatNumber(num, decimals = 0) {
    if (num === null || num === undefined) return '-'
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(num)
}

export function formatCompactNumber(num) {
    if (num === null || num === undefined) return '-'
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
}

export function formatPercent(num, decimals = 1) {
    if (num === null || num === undefined) return '-'
    return `${Number(num).toFixed(decimals)}%`
}

export function formatCurrency(num, currency = 'NPR') {
    if (num === null || num === undefined) return '-'
    return new Intl.NumberFormat('en-NP', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(num)
}

// Date formatting
export function formatDate(dateStr, formatStr = 'MMM dd, yyyy') {
    if (!dateStr) return '-'
    try {
        const date = typeof dateStr === 'string' ? parseISO(dateStr) : dateStr
        return format(date, formatStr)
    } catch {
        return dateStr
    }
}

export function formatDateTime(dateStr) {
    return formatDate(dateStr, 'MMM dd, yyyy HH:mm')
}

export function formatRelativeTime(dateStr) {
    if (!dateStr) return '-'
    try {
        const date = typeof dateStr === 'string' ? parseISO(dateStr) : dateStr
        return formatDistanceToNow(date, { addSuffix: true })
    } catch {
        return dateStr
    }
}

// Metric formatting
export function formatMAPE(value) {
    return `${Number(value).toFixed(1)}%`
}

export function formatChange(value) {
    const num = Number(value)
    const prefix = num >= 0 ? '+' : ''
    return `${prefix}${num.toFixed(1)}%`
}

// Data processing helpers
export function getChangeType(value) {
    const num = Number(value)
    if (num > 0) return 'positive'
    if (num < 0) return 'negative'
    return 'neutral'
}

export function getStatusColor(status) {
    const statusColors = {
        healthy: 'success',
        good: 'success',
        warning: 'warning',
        critical: 'danger',
        danger: 'danger',
        overstock: 'info',
        info: 'info'
    }
    return statusColors[status?.toLowerCase()] || 'muted'
}

// Table sorting helper
export function sortData(data, sortKey, sortDirection) {
    return [...data].sort((a, b) => {
        const aVal = a[sortKey]
        const bVal = b[sortKey]

        if (aVal === null || aVal === undefined) return 1
        if (bVal === null || bVal === undefined) return -1

        if (typeof aVal === 'string') {
            return sortDirection === 'asc'
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal)
        }

        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
    })
}

// Get color for chart based on value
export function getColorForValue(value, thresholds = { good: 10, warning: 20 }) {
    if (value <= thresholds.good) return '#10b981' // success
    if (value <= thresholds.warning) return '#f59e0b' // warning
    return '#ef4444' // danger
}
