import { useState, useEffect } from 'react'
import {
    Package,
    AlertTriangle,
    TrendingUp,
    TrendingDown,
    RefreshCw,
    Filter,
    Download,
    ChevronUp,
    ChevronDown
} from 'lucide-react'
import api from '../services/api'
import { formatNumber, formatDate, sortData } from '../utils/formatters'
import { CATEGORIES, ABC_CLASS_INFO } from '../utils/constants'

export default function Inventory() {
    const [loading, setLoading] = useState(true)
    const [inventory, setInventory] = useState([])
    const [filterCategory, setFilterCategory] = useState('')
    const [filterStatus, setFilterStatus] = useState('')
    const [sortKey, setSortKey] = useState('stockLevel')
    const [sortDirection, setSortDirection] = useState('desc')

    useEffect(() => {
        loadInventory()
    }, [])

    const loadInventory = async () => {
        setLoading(true)
        try {
            const data = await api.getInventory()
            setInventory(data)
        } catch (error) {
            console.error('Error loading inventory:', error)
        }
        setLoading(false)
    }

    // Filter and sort inventory
    const filteredInventory = inventory
        .filter(item => {
            if (filterCategory && item.category !== filterCategory) return false
            if (filterStatus && item.status !== filterStatus) return false
            return true
        })

    const sortedInventory = sortData(filteredInventory, sortKey, sortDirection)

    // Calculate summary stats
    const stats = {
        totalItems: inventory.length,
        lowStock: inventory.filter(i => i.status === 'warning').length,
        critical: inventory.filter(i => i.status === 'critical').length,
        overstock: inventory.filter(i => i.status === 'overstock').length
    }

    const handleSort = (key) => {
        if (sortKey === key) {
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
        } else {
            setSortKey(key)
            setSortDirection('desc')
        }
    }

    const SortIcon = ({ column }) => {
        if (sortKey !== column) return null
        return sortDirection === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
    }

    const getStatusBadge = (status) => {
        const config = {
            healthy: { class: 'badge-success', label: 'Healthy' },
            warning: { class: 'badge-warning', label: 'Low Stock' },
            critical: { class: 'badge-danger', label: 'Critical' },
            overstock: { class: 'badge-info', label: 'Overstock' }
        }
        const { class: badgeClass, label } = config[status] || config.healthy
        return <span className={`badge ${badgeClass}`}>{label}</span>
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
                    <h1 style={{ marginBottom: 'var(--space-2)' }}>Inventory Management</h1>
                    <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                        Monitor stock levels, reorder points, and inventory health
                    </p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                    <button className="btn btn-secondary">
                        <Download size={18} />
                        Export
                    </button>
                    <button className="btn btn-primary" onClick={loadInventory}>
                        <RefreshCw size={18} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Summary Cards */}
            <div
                className="grid grid-cols-4 gap-4"
                style={{ marginBottom: 'var(--space-6)' }}
            >
                <div className="kpi-card">
                    <div className="kpi-icon primary">
                        <Package size={24} />
                    </div>
                    <div className="kpi-value">{stats.totalItems}</div>
                    <div className="kpi-label">Total SKUs</div>
                </div>
                <div className="kpi-card warning">
                    <div className="kpi-icon warning">
                        <TrendingDown size={24} />
                    </div>
                    <div className="kpi-value">{stats.lowStock}</div>
                    <div className="kpi-label">Low Stock Items</div>
                </div>
                <div className="kpi-card danger">
                    <div className="kpi-icon danger">
                        <AlertTriangle size={24} />
                    </div>
                    <div className="kpi-value">{stats.critical}</div>
                    <div className="kpi-label">Critical Stock</div>
                </div>
                <div className="kpi-card info">
                    <div className="kpi-icon info">
                        <TrendingUp size={24} />
                    </div>
                    <div className="kpi-value">{stats.overstock}</div>
                    <div className="kpi-label">Overstock Items</div>
                </div>
            </div>

            {/* Filters */}
            <div
                className="card"
                style={{ marginBottom: 'var(--space-6)', padding: 'var(--space-4)' }}
            >
                <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'flex-end' }}>
                    <div className="form-group" style={{ marginBottom: 0, width: 200 }}>
                        <label className="form-label">Category</label>
                        <select
                            className="form-select"
                            value={filterCategory}
                            onChange={(e) => setFilterCategory(e.target.value)}
                        >
                            <option value="">All Categories</option>
                            {CATEGORIES.map(cat => (
                                <option key={cat} value={cat}>{cat}</option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group" style={{ marginBottom: 0, width: 200 }}>
                        <label className="form-label">Status</label>
                        <select
                            className="form-select"
                            value={filterStatus}
                            onChange={(e) => setFilterStatus(e.target.value)}
                        >
                            <option value="">All Statuses</option>
                            <option value="healthy">Healthy</option>
                            <option value="warning">Low Stock</option>
                            <option value="critical">Critical</option>
                            <option value="overstock">Overstock</option>
                        </select>
                    </div>
                    <div style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Showing {sortedInventory.length} of {inventory.length} items
                    </div>
                </div>
            </div>

            {/* Inventory Table */}
            <div className="table-container">
                <table className="table">
                    <thead>
                        <tr>
                            <th
                                onClick={() => handleSort('name')}
                                style={{ cursor: 'pointer' }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                                    SKU <SortIcon column="name" />
                                </div>
                            </th>
                            <th>Category</th>
                            <th>ABC Class</th>
                            <th
                                onClick={() => handleSort('stockLevel')}
                                style={{ cursor: 'pointer', textAlign: 'right' }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 'var(--space-1)' }}>
                                    Stock Level <SortIcon column="stockLevel" />
                                </div>
                            </th>
                            <th style={{ textAlign: 'right' }}>Reorder Point</th>
                            <th style={{ textAlign: 'right' }}>Safety Stock</th>
                            <th
                                onClick={() => handleSort('mos')}
                                style={{ cursor: 'pointer', textAlign: 'right' }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 'var(--space-1)' }}>
                                    MOS <SortIcon column="mos" />
                                </div>
                            </th>
                            <th>Status</th>
                            <th style={{ textAlign: 'right' }}>Next Replenishment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            Array.from({ length: 10 }).map((_, i) => (
                                <tr key={i}>
                                    {Array.from({ length: 9 }).map((_, j) => (
                                        <td key={j}><div className="skeleton" style={{ height: 20 }} /></td>
                                    ))}
                                </tr>
                            ))
                        ) : sortedInventory.length === 0 ? (
                            <tr>
                                <td colSpan={9} style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                                    No inventory items found matching your filters.
                                </td>
                            </tr>
                        ) : (
                            sortedInventory.slice(0, 20).map((item, index) => (
                                <tr key={index}>
                                    <td>
                                        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                                            {item.name}
                                        </span>
                                    </td>
                                    <td>
                                        <span className="badge badge-primary">{item.category}</span>
                                    </td>
                                    <td>
                                        <span
                                            className={`badge badge-${ABC_CLASS_INFO[item.abcClass]?.color || 'info'}`}
                                        >
                                            {item.abcClass} - {ABC_CLASS_INFO[item.abcClass]?.label}
                                        </span>
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                                        {formatNumber(item.stockLevel)}
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                        {formatNumber(item.reorderPoint)}
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                        {formatNumber(item.safetyStock)}
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                                        {item.mos}
                                    </td>
                                    <td>{getStatusBadge(item.status)}</td>
                                    <td style={{ textAlign: 'right', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                        {formatDate(item.nextReplenishment)}
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
