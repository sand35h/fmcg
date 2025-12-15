import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
    LayoutDashboard,
    TrendingUp,
    Package,
    GitBranch,
    Database,
    Brain,
    Settings,
    ChevronLeft,
    ChevronRight,
    Activity
} from 'lucide-react'
import { NAV_ITEMS } from '../../utils/constants'

const iconMap = {
    LayoutDashboard,
    TrendingUp,
    Package,
    GitBranch,
    Database,
    Brain,
    Settings
}

export default function Sidebar({ collapsed, onToggle }) {
    const location = useLocation()

    return (
        <aside
            className="sidebar"
            style={{
                width: collapsed ? 'var(--sidebar-collapsed-width)' : 'var(--sidebar-width)',
                minHeight: '100vh',
                background: 'var(--bg-secondary)',
                borderRight: '1px solid var(--border-primary)',
                display: 'flex',
                flexDirection: 'column',
                transition: 'width var(--transition-base)',
                position: 'fixed',
                left: 0,
                top: 0,
                zIndex: 50
            }}
        >
            {/* Logo */}
            <div
                className="sidebar-header"
                style={{
                    padding: 'var(--space-5)',
                    borderBottom: '1px solid var(--border-secondary)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-3)',
                    height: 'var(--header-height)'
                }}
            >
                <div
                    style={{
                        width: 40,
                        height: 40,
                        borderRadius: 'var(--radius-lg)',
                        background: 'var(--gradient-primary)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0
                    }}
                >
                    <Activity size={22} color="white" />
                </div>
                {!collapsed && (
                    <div className="animate-fadeIn">
                        <h1 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.2 }}>
                            FMCG Forecast
                        </h1>
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Supply & Demand
                        </span>
                    </div>
                )}
            </div>

            {/* Navigation */}
            <nav style={{ flex: 1, padding: 'var(--space-4)', overflowY: 'auto' }}>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                    {NAV_ITEMS.map((item) => {
                        const Icon = iconMap[item.icon]
                        const isActive = location.pathname === item.path

                        return (
                            <li key={item.path}>
                                <Link
                                    to={item.path}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 'var(--space-3)',
                                        padding: collapsed ? 'var(--space-3)' : 'var(--space-3) var(--space-4)',
                                        borderRadius: 'var(--radius-lg)',
                                        color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
                                        background: isActive ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
                                        textDecoration: 'none',
                                        fontWeight: isActive ? 600 : 400,
                                        fontSize: '0.875rem',
                                        transition: 'all var(--transition-fast)',
                                        justifyContent: collapsed ? 'center' : 'flex-start',
                                        position: 'relative'
                                    }}
                                    onMouseEnter={(e) => {
                                        if (!isActive) {
                                            e.currentTarget.style.background = 'var(--bg-hover)'
                                            e.currentTarget.style.color = 'var(--text-primary)'
                                        }
                                    }}
                                    onMouseLeave={(e) => {
                                        if (!isActive) {
                                            e.currentTarget.style.background = 'transparent'
                                            e.currentTarget.style.color = 'var(--text-muted)'
                                        }
                                    }}
                                    title={collapsed ? item.label : undefined}
                                >
                                    {isActive && (
                                        <div
                                            style={{
                                                position: 'absolute',
                                                left: 0,
                                                top: '50%',
                                                transform: 'translateY(-50%)',
                                                width: 3,
                                                height: '60%',
                                                background: 'var(--primary-500)',
                                                borderRadius: 'var(--radius-full)'
                                            }}
                                        />
                                    )}
                                    {Icon && <Icon size={20} style={{ flexShrink: 0 }} />}
                                    {!collapsed && <span className="animate-fadeIn">{item.label}</span>}
                                </Link>
                            </li>
                        )
                    })}
                </ul>
            </nav>

            {/* Toggle Button */}
            <div
                style={{
                    padding: 'var(--space-4)',
                    borderTop: '1px solid var(--border-secondary)'
                }}
            >
                <button
                    onClick={onToggle}
                    className="btn btn-ghost"
                    style={{
                        width: '100%',
                        justifyContent: collapsed ? 'center' : 'flex-start',
                        gap: 'var(--space-3)'
                    }}
                >
                    {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
                    {!collapsed && <span>Collapse</span>}
                </button>
            </div>
        </aside>
    )
}
