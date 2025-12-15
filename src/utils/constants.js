// Navigation items
export const NAV_ITEMS = [
    { path: '/', label: 'Dashboard', icon: 'LayoutDashboard' },
    { path: '/forecasts', label: 'Forecast Explorer', icon: 'TrendingUp' },
    { path: '/inventory', label: 'Inventory', icon: 'Package' },
    { path: '/scenarios', label: 'What-If Scenarios', icon: 'GitBranch' },
    { path: '/data-jobs', label: 'Data Jobs', icon: 'Database' },
    { path: '/models', label: 'Model Status', icon: 'Brain' },
    { path: '/settings', label: 'Settings', icon: 'Settings' },
]

// Chart colors
export const CHART_COLORS = {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#3b82f6',
    cyan: '#06b6d4',
    pink: '#ec4899',
    muted: '#64748b'
}

// Status colors
export const STATUS_COLORS = {
    healthy: 'success',
    warning: 'warning',
    critical: 'danger',
    overstock: 'info',
    running: 'info',
    completed: 'success',
    failed: 'danger',
    pending: 'muted'
}

// ABC Class info
export const ABC_CLASS_INFO = {
    A: { label: 'Fast Movers', color: 'success', description: 'Top 20% SKUs generating 80% revenue' },
    B: { label: 'Medium Movers', color: 'warning', description: 'Middle 30% SKUs' },
    C: { label: 'Slow Movers', color: 'info', description: 'Bottom 50% SKUs' }
}

// Categories
export const CATEGORIES = ['DAIRY', 'BEVERAGES', 'SNACKS', 'HOMECARE', 'PERSONALCARE', 'NOODLES', 'BISCUITS']

// Regions (UK)
export const REGIONS = ['Greater London', 'South East', 'North West', 'Midlands', 'Scotland', 'Wales']

// Channels
export const CHANNELS = ['ModernTrade', 'Traditional', 'Ecommerce']

// Time horizons
export const TIME_HORIZONS = [
    { value: 7, label: '7 Days' },
    { value: 14, label: '14 Days' },
    { value: 30, label: '30 Days' },
    { value: 60, label: '60 Days' },
    { value: 90, label: '90 Days' }
]
