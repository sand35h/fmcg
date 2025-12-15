import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
    Bell,
    Search,
    User,
    LogOut,
    Settings,
    ChevronDown,
    Moon,
    HelpCircle,
    Download
} from 'lucide-react'
import { useAuth } from '../../hooks/useAuth'
import { formatRelativeTime } from '../../utils/formatters'

export default function Header() {
    const { user, logout } = useAuth()
    const navigate = useNavigate()
    const [showUserMenu, setShowUserMenu] = useState(false)
    const [showNotifications, setShowNotifications] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const userMenuRef = useRef(null)
    const notifRef = useRef(null)

    // Mock notifications
    const notifications = [
        { id: 1, type: 'warning', message: 'Low stock alert for 3 SKUs', time: new Date(Date.now() - 1800000), read: false },
        { id: 2, type: 'success', message: 'Weekly forecast generated successfully', time: new Date(Date.now() - 3600000), read: false },
        { id: 3, type: 'info', message: 'Model training completed', time: new Date(Date.now() - 7200000), read: true },
        { id: 4, type: 'danger', message: 'Data pipeline failed - requires attention', time: new Date(Date.now() - 86400000), read: true }
    ]

    const unreadCount = notifications.filter(n => !n.read).length

    // Close menus on outside click
    useEffect(() => {
        function handleClickOutside(event) {
            if (userMenuRef.current && !userMenuRef.current.contains(event.target)) {
                setShowUserMenu(false)
            }
            if (notifRef.current && !notifRef.current.contains(event.target)) {
                setShowNotifications(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const handleLogout = () => {
        logout()
        navigate('/login')
    }

    const getNotifColor = (type) => {
        const colors = {
            success: 'var(--accent-success)',
            warning: 'var(--accent-warning)',
            danger: 'var(--accent-danger)',
            info: 'var(--accent-info)'
        }
        return colors[type] || colors.info
    }

    return (
        <header
            style={{
                height: 'var(--header-height)',
                background: 'var(--bg-secondary)',
                borderBottom: '1px solid var(--border-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 var(--space-6)',
                position: 'sticky',
                top: 0,
                zIndex: 40,
                backdropFilter: 'blur(10px)'
            }}
        >
            {/* Search Bar */}
            <div style={{ flex: 1, maxWidth: 400 }}>
                <div
                    style={{
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center'
                    }}
                >
                    <Search
                        size={18}
                        style={{
                            position: 'absolute',
                            left: 'var(--space-3)',
                            color: 'var(--text-muted)',
                            pointerEvents: 'none'
                        }}
                    />
                    <input
                        type="text"
                        placeholder="Search SKUs, locations, forecasts..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="form-input"
                        style={{
                            paddingLeft: 'var(--space-10)',
                            background: 'var(--bg-tertiary)',
                            border: '1px solid transparent',
                            width: '100%'
                        }}
                    />
                </div>
            </div>

            {/* Right Actions */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                {/* Export Button */}
                <button
                    className="btn btn-ghost btn-icon"
                    title="Export Data"
                >
                    <Download size={20} />
                </button>

                {/* Help */}
                <button
                    className="btn btn-ghost btn-icon"
                    title="Help"
                >
                    <HelpCircle size={20} />
                </button>

                {/* Notifications */}
                <div ref={notifRef} style={{ position: 'relative' }}>
                    <button
                        className="btn btn-ghost btn-icon"
                        onClick={() => setShowNotifications(!showNotifications)}
                        style={{ position: 'relative' }}
                    >
                        <Bell size={20} />
                        {unreadCount > 0 && (
                            <span
                                style={{
                                    position: 'absolute',
                                    top: 4,
                                    right: 4,
                                    width: 16,
                                    height: 16,
                                    borderRadius: 'var(--radius-full)',
                                    background: 'var(--accent-danger)',
                                    color: 'white',
                                    fontSize: '0.65rem',
                                    fontWeight: 700,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }}
                            >
                                {unreadCount}
                            </span>
                        )}
                    </button>

                    {showNotifications && (
                        <div
                            className="animate-slideUp"
                            style={{
                                position: 'absolute',
                                top: 'calc(100% + var(--space-2))',
                                right: 0,
                                width: 360,
                                background: 'var(--bg-secondary)',
                                border: '1px solid var(--border-primary)',
                                borderRadius: 'var(--radius-xl)',
                                boxShadow: 'var(--shadow-xl)',
                                overflow: 'hidden',
                                zIndex: 'var(--z-dropdown)'
                            }}
                        >
                            <div
                                style={{
                                    padding: 'var(--space-4)',
                                    borderBottom: '1px solid var(--border-secondary)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between'
                                }}
                            >
                                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600 }}>Notifications</h3>
                                <button
                                    className="btn btn-ghost btn-sm"
                                    style={{ fontSize: '0.75rem' }}
                                >
                                    Mark all read
                                </button>
                            </div>
                            <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                                {notifications.map((notif) => (
                                    <div
                                        key={notif.id}
                                        style={{
                                            padding: 'var(--space-4)',
                                            borderBottom: '1px solid var(--border-secondary)',
                                            display: 'flex',
                                            gap: 'var(--space-3)',
                                            background: notif.read ? 'transparent' : 'rgba(99, 102, 241, 0.05)',
                                            cursor: 'pointer',
                                            transition: 'background var(--transition-fast)'
                                        }}
                                        onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                                        onMouseLeave={(e) => e.currentTarget.style.background = notif.read ? 'transparent' : 'rgba(99, 102, 241, 0.05)'}
                                    >
                                        <div
                                            style={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: 'var(--radius-full)',
                                                background: getNotifColor(notif.type),
                                                marginTop: 6,
                                                flexShrink: 0
                                            }}
                                        />
                                        <div style={{ flex: 1 }}>
                                            <p style={{
                                                fontSize: '0.8125rem',
                                                color: 'var(--text-primary)',
                                                marginBottom: 'var(--space-1)'
                                            }}>
                                                {notif.message}
                                            </p>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                                {formatRelativeTime(notif.time)}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div
                                style={{
                                    padding: 'var(--space-3)',
                                    textAlign: 'center',
                                    borderTop: '1px solid var(--border-secondary)'
                                }}
                            >
                                <button className="btn btn-ghost btn-sm" style={{ fontSize: '0.8125rem' }}>
                                    View all notifications
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* User Menu */}
                <div ref={userMenuRef} style={{ position: 'relative' }}>
                    <button
                        onClick={() => setShowUserMenu(!showUserMenu)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'var(--space-3)',
                            padding: 'var(--space-2) var(--space-3)',
                            background: 'transparent',
                            border: '1px solid var(--border-primary)',
                            borderRadius: 'var(--radius-lg)',
                            cursor: 'pointer',
                            transition: 'all var(--transition-fast)'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--primary-500)'}
                        onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-primary)'}
                    >
                        <div
                            style={{
                                width: 32,
                                height: 32,
                                borderRadius: 'var(--radius-full)',
                                background: 'var(--gradient-primary)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}
                        >
                            <User size={16} color="white" />
                        </div>
                        <div style={{ textAlign: 'left' }}>
                            <div style={{ fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                                {user?.name || 'User'}
                            </div>
                            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                                {user?.role || 'Viewer'}
                            </div>
                        </div>
                        <ChevronDown size={16} color="var(--text-muted)" />
                    </button>

                    {showUserMenu && (
                        <div
                            className="animate-slideUp"
                            style={{
                                position: 'absolute',
                                top: 'calc(100% + var(--space-2))',
                                right: 0,
                                width: 200,
                                background: 'var(--bg-secondary)',
                                border: '1px solid var(--border-primary)',
                                borderRadius: 'var(--radius-lg)',
                                boxShadow: 'var(--shadow-xl)',
                                overflow: 'hidden',
                                zIndex: 'var(--z-dropdown)'
                            }}
                        >
                            <div style={{ padding: 'var(--space-2)' }}>
                                <button
                                    className="btn btn-ghost"
                                    style={{ width: '100%', justifyContent: 'flex-start', gap: 'var(--space-3)' }}
                                    onClick={() => { navigate('/settings'); setShowUserMenu(false); }}
                                >
                                    <Settings size={16} />
                                    Settings
                                </button>
                                <button
                                    className="btn btn-ghost"
                                    style={{ width: '100%', justifyContent: 'flex-start', gap: 'var(--space-3)' }}
                                >
                                    <Moon size={16} />
                                    Dark Mode
                                </button>
                            </div>
                            <div
                                style={{
                                    borderTop: '1px solid var(--border-secondary)',
                                    padding: 'var(--space-2)'
                                }}
                            >
                                <button
                                    className="btn btn-ghost"
                                    style={{
                                        width: '100%',
                                        justifyContent: 'flex-start',
                                        gap: 'var(--space-3)',
                                        color: 'var(--accent-danger)'
                                    }}
                                    onClick={handleLogout}
                                >
                                    <LogOut size={16} />
                                    Logout
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </header>
    )
}
