import { useState } from 'react'
import Sidebar from './Sidebar'
import Header from './Header'

export default function Layout({ children }) {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

    return (
        <div style={{ display: 'flex', minHeight: '100vh' }}>
            {/* Sidebar */}
            <Sidebar
                collapsed={sidebarCollapsed}
                onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
            />

            {/* Main Content Area */}
            <div
                style={{
                    flex: 1,
                    marginLeft: sidebarCollapsed ? 'var(--sidebar-collapsed-width)' : 'var(--sidebar-width)',
                    transition: 'margin-left var(--transition-base)',
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: '100vh'
                }}
            >
                {/* Header */}
                <Header />

                {/* Page Content */}
                <main
                    style={{
                        flex: 1,
                        padding: 'var(--space-6)',
                        overflowY: 'auto'
                    }}
                >
                    {children}
                </main>
            </div>
        </div>
    )
}
