import { createContext, useState, useEffect } from 'react'

export const AuthContext = createContext(null)

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        // Check for existing session
        const storedUser = localStorage.getItem('fmcg_user')
        if (storedUser) {
            setUser(JSON.parse(storedUser))
        }
        setIsLoading(false)
    }, [])

    const login = async (email, password) => {
        // Simulated authentication (replace with actual Cognito integration)
        setIsLoading(true)

        // Demo credentials
        const validCredentials = [
            { email: 'admin@fmcg.com', password: 'admin123', role: 'admin', name: 'Admin User' },
            { email: 'planner@fmcg.com', password: 'planner123', role: 'planner', name: 'Demand Planner' },
            { email: 'demo@fmcg.com', password: 'demo123', role: 'viewer', name: 'Demo User' },
        ]

        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 1000))

        const foundUser = validCredentials.find(
            u => u.email === email && u.password === password
        )

        if (foundUser) {
            const userData = {
                id: Math.random().toString(36).substr(2, 9),
                email: foundUser.email,
                name: foundUser.name,
                role: foundUser.role,
                avatar: null,
                loginTime: new Date().toISOString()
            }
            setUser(userData)
            localStorage.setItem('fmcg_user', JSON.stringify(userData))
            setIsLoading(false)
            return { success: true }
        }

        setIsLoading(false)
        return { success: false, error: 'Invalid email or password' }
    }

    const logout = () => {
        setUser(null)
        localStorage.removeItem('fmcg_user')
    }

    const value = {
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        logout
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
}
