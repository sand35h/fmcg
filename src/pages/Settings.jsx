import { useState } from 'react'
import {
    Settings as SettingsIcon,
    User,
    Bell,
    Shield,
    Database,
    Moon,
    Sun,
    Save,
    RefreshCw
} from 'lucide-react'
import { useAuth } from '../hooks/useAuth'

export default function Settings() {
    const { user } = useAuth()
    const [activeTab, setActiveTab] = useState('profile')
    const [saving, setSaving] = useState(false)

    // Settings state
    const [settings, setSettings] = useState({
        profile: {
            name: user?.name || '',
            email: user?.email || '',
            role: user?.role || 'viewer'
        },
        notifications: {
            emailAlerts: true,
            stockoutAlerts: true,
            forecastAlerts: true,
            modelAlerts: true,
            weeklyDigest: true
        },
        preferences: {
            darkMode: true,
            compactView: false,
            defaultTimeHorizon: 30,
            defaultRegion: '',
            autoRefresh: true,
            refreshInterval: 5
        },
        dataSettings: {
            forecastHorizon: 90,
            confidenceLevel: 80,
            retrainFrequency: 'weekly',
            alertThreshold: 15
        }
    })

    const handleSave = async () => {
        setSaving(true)
        // Simulate save
        await new Promise(resolve => setTimeout(resolve, 1000))
        setSaving(false)
        alert('Settings saved successfully!')
    }

    const tabs = [
        { id: 'profile', label: 'Profile', icon: User },
        { id: 'notifications', label: 'Notifications', icon: Bell },
        { id: 'preferences', label: 'Preferences', icon: SettingsIcon },
        { id: 'data', label: 'Data & Model', icon: Database }
    ]

    return (
        <div className="animate-fadeIn">
            {/* Page Header */}
            <div style={{ marginBottom: 'var(--space-6)' }}>
                <h1 style={{ marginBottom: 'var(--space-2)' }}>
                    <SettingsIcon size={28} style={{ marginRight: 'var(--space-3)', verticalAlign: 'middle' }} />
                    Settings
                </h1>
                <p style={{ color: 'var(--text-muted)', marginBottom: 0 }}>
                    Manage your account, notifications, and system preferences
                </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 'var(--space-6)' }}>
                {/* Tabs */}
                <div className="card" style={{ height: 'fit-content' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                className={`btn ${activeTab === tab.id ? 'btn-primary' : 'btn-ghost'}`}
                                style={{ justifyContent: 'flex-start' }}
                                onClick={() => setActiveTab(tab.id)}
                            >
                                <tab.icon size={18} />
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Content */}
                <div className="card">
                    {/* Profile Tab */}
                    {activeTab === 'profile' && (
                        <div className="animate-fadeIn">
                            <h2 style={{ fontSize: '1.25rem', marginBottom: 'var(--space-6)' }}>
                                Profile Settings
                            </h2>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                                <div className="form-group">
                                    <label className="form-label">Full Name</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        value={settings.profile.name}
                                        onChange={(e) => setSettings({
                                            ...settings,
                                            profile: { ...settings.profile, name: e.target.value }
                                        })}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Email Address</label>
                                    <input
                                        type="email"
                                        className="form-input"
                                        value={settings.profile.email}
                                        disabled
                                        style={{ opacity: 0.6 }}
                                    />
                                </div>
                            </div>

                            <div className="form-group">
                                <label className="form-label">Role</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    value={settings.profile.role.charAt(0).toUpperCase() + settings.profile.role.slice(1)}
                                    disabled
                                    style={{ opacity: 0.6, width: 'fit-content', minWidth: 200 }}
                                />
                            </div>

                            <div
                                style={{
                                    marginTop: 'var(--space-6)',
                                    paddingTop: 'var(--space-4)',
                                    borderTop: '1px solid var(--border-secondary)'
                                }}
                            >
                                <h4 style={{ marginBottom: 'var(--space-4)' }}>Account Security</h4>
                                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                                    <button className="btn btn-secondary">
                                        <Shield size={18} />
                                        Change Password
                                    </button>
                                    <button className="btn btn-secondary">
                                        Enable 2FA
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Notifications Tab */}
                    {activeTab === 'notifications' && (
                        <div className="animate-fadeIn">
                            <h2 style={{ fontSize: '1.25rem', marginBottom: 'var(--space-6)' }}>
                                Notification Preferences
                            </h2>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                                {[
                                    { key: 'emailAlerts', label: 'Email Notifications', desc: 'Receive email alerts for important events' },
                                    { key: 'stockoutAlerts', label: 'Stockout Alerts', desc: 'Get notified when stock levels are critical' },
                                    { key: 'forecastAlerts', label: 'Forecast Alerts', desc: 'Notifications for significant demand changes' },
                                    { key: 'modelAlerts', label: 'Model Alerts', desc: 'Alerts when model performance degrades' },
                                    { key: 'weeklyDigest', label: 'Weekly Digest', desc: 'Summary email every Monday morning' }
                                ].map(item => (
                                    <div
                                        key={item.key}
                                        style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            padding: 'var(--space-4)',
                                            background: 'var(--bg-tertiary)',
                                            borderRadius: 'var(--radius-lg)'
                                        }}
                                    >
                                        <div>
                                            <div style={{ fontWeight: 500, color: 'var(--text-primary)', marginBottom: 'var(--space-1)' }}>
                                                {item.label}
                                            </div>
                                            <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                                {item.desc}
                                            </div>
                                        </div>
                                        <label style={{ position: 'relative', display: 'inline-block', width: 48, height: 24 }}>
                                            <input
                                                type="checkbox"
                                                checked={settings.notifications[item.key]}
                                                onChange={(e) => setSettings({
                                                    ...settings,
                                                    notifications: { ...settings.notifications, [item.key]: e.target.checked }
                                                })}
                                                style={{ opacity: 0, width: 0, height: 0 }}
                                            />
                                            <span
                                                style={{
                                                    position: 'absolute',
                                                    cursor: 'pointer',
                                                    top: 0,
                                                    left: 0,
                                                    right: 0,
                                                    bottom: 0,
                                                    background: settings.notifications[item.key] ? 'var(--primary-500)' : 'var(--bg-secondary)',
                                                    borderRadius: 'var(--radius-full)',
                                                    transition: 'var(--transition-fast)'
                                                }}
                                            >
                                                <span
                                                    style={{
                                                        position: 'absolute',
                                                        content: '',
                                                        height: 18,
                                                        width: 18,
                                                        left: settings.notifications[item.key] ? 27 : 3,
                                                        bottom: 3,
                                                        background: 'white',
                                                        borderRadius: 'var(--radius-full)',
                                                        transition: 'var(--transition-fast)'
                                                    }}
                                                />
                                            </span>
                                        </label>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Preferences Tab */}
                    {activeTab === 'preferences' && (
                        <div className="animate-fadeIn">
                            <h2 style={{ fontSize: '1.25rem', marginBottom: 'var(--space-6)' }}>
                                Display Preferences
                            </h2>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                                <div className="form-group">
                                    <label className="form-label">Theme</label>
                                    <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                                        <button
                                            className={`btn ${settings.preferences.darkMode ? 'btn-primary' : 'btn-secondary'}`}
                                            onClick={() => setSettings({
                                                ...settings,
                                                preferences: { ...settings.preferences, darkMode: true }
                                            })}
                                        >
                                            <Moon size={18} />
                                            Dark
                                        </button>
                                        <button
                                            className={`btn ${!settings.preferences.darkMode ? 'btn-primary' : 'btn-secondary'}`}
                                            onClick={() => setSettings({
                                                ...settings,
                                                preferences: { ...settings.preferences, darkMode: false }
                                            })}
                                        >
                                            <Sun size={18} />
                                            Light
                                        </button>
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Default Time Horizon</label>
                                    <select
                                        className="form-select"
                                        value={settings.preferences.defaultTimeHorizon}
                                        onChange={(e) => setSettings({
                                            ...settings,
                                            preferences: { ...settings.preferences, defaultTimeHorizon: Number(e.target.value) }
                                        })}
                                    >
                                        <option value={7}>7 Days</option>
                                        <option value={14}>14 Days</option>
                                        <option value={30}>30 Days</option>
                                        <option value={60}>60 Days</option>
                                        <option value={90}>90 Days</option>
                                    </select>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Auto Refresh</label>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                                        <input
                                            type="checkbox"
                                            checked={settings.preferences.autoRefresh}
                                            onChange={(e) => setSettings({
                                                ...settings,
                                                preferences: { ...settings.preferences, autoRefresh: e.target.checked }
                                            })}
                                            style={{ width: 18, height: 18 }}
                                        />
                                        <span style={{ color: 'var(--text-secondary)' }}>Enable auto-refresh</span>
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Refresh Interval (minutes)</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={settings.preferences.refreshInterval}
                                        min={1}
                                        max={60}
                                        onChange={(e) => setSettings({
                                            ...settings,
                                            preferences: { ...settings.preferences, refreshInterval: Number(e.target.value) }
                                        })}
                                        disabled={!settings.preferences.autoRefresh}
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Data & Model Tab */}
                    {activeTab === 'data' && (
                        <div className="animate-fadeIn">
                            <h2 style={{ fontSize: '1.25rem', marginBottom: 'var(--space-6)' }}>
                                Data & Model Settings
                            </h2>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                                <div className="form-group">
                                    <label className="form-label">Forecast Horizon (days)</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={settings.dataSettings.forecastHorizon}
                                        onChange={(e) => setSettings({
                                            ...settings,
                                            dataSettings: { ...settings.dataSettings, forecastHorizon: Number(e.target.value) }
                                        })}
                                    />
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Confidence Level (%)</label>
                                    <select
                                        className="form-select"
                                        value={settings.dataSettings.confidenceLevel}
                                        onChange={(e) => setSettings({
                                            ...settings,
                                            dataSettings: { ...settings.dataSettings, confidenceLevel: Number(e.target.value) }
                                        })}
                                    >
                                        <option value={80}>80%</option>
                                        <option value={90}>90%</option>
                                        <option value={95}>95%</option>
                                    </select>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Retrain Frequency</label>
                                    <select
                                        className="form-select"
                                        value={settings.dataSettings.retrainFrequency}
                                        onChange={(e) => setSettings({
                                            ...settings,
                                            dataSettings: { ...settings.dataSettings, retrainFrequency: e.target.value }
                                        })}
                                    >
                                        <option value="daily">Daily</option>
                                        <option value="weekly">Weekly</option>
                                        <option value="monthly">Monthly</option>
                                        <option value="manual">Manual Only</option>
                                    </select>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Alert Threshold (% MAPE change)</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={settings.dataSettings.alertThreshold}
                                        onChange={(e) => setSettings({
                                            ...settings,
                                            dataSettings: { ...settings.dataSettings, alertThreshold: Number(e.target.value) }
                                        })}
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Save Button */}
                    <div
                        style={{
                            marginTop: 'var(--space-8)',
                            paddingTop: 'var(--space-4)',
                            borderTop: '1px solid var(--border-secondary)',
                            display: 'flex',
                            justifyContent: 'flex-end',
                            gap: 'var(--space-2)'
                        }}
                    >
                        <button className="btn btn-secondary">
                            <RefreshCw size={18} />
                            Reset to Defaults
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={handleSave}
                            disabled={saving}
                        >
                            {saving ? (
                                <>
                                    <span className="spinner" />
                                    Saving...
                                </>
                            ) : (
                                <>
                                    <Save size={18} />
                                    Save Changes
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
