// API Service - Replace with actual API Gateway endpoints in production
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

// Simulated API delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

// Generic fetch wrapper with error handling
async function fetchApi(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`

    const config = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    }

    try {
        const response = await fetch(url, config)

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`)
        }

        return await response.json()
    } catch (error) {
        console.error('API Error:', error)
        throw error
    }
}

// =============================================================================
// MOCK DATA GENERATORS
// =============================================================================

// Generate mock forecast data
function generateMockForecasts(days = 90) {
    const data = []
    const baseDate = new Date()
    baseDate.setDate(baseDate.getDate() - days)

    let baseValue = 1000 + Math.random() * 500

    for (let i = 0; i < days; i++) {
        const date = new Date(baseDate)
        date.setDate(date.getDate() + i)

        // Add seasonality and trend
        const dayOfWeek = date.getDay()
        const weekendEffect = dayOfWeek === 0 || dayOfWeek === 6 ? 0.85 : 1
        const trend = 1 + (i / days) * 0.1
        const seasonality = 1 + 0.15 * Math.sin((i / 7) * Math.PI)

        const actual = Math.round(baseValue * weekendEffect * trend * seasonality * (0.9 + Math.random() * 0.2))
        const forecast = Math.round(actual * (0.95 + Math.random() * 0.1))
        const lowerBound = Math.round(forecast * 0.85)
        const upperBound = Math.round(forecast * 1.15)

        data.push({
            date: date.toISOString().split('T')[0],
            actual: i < days - 7 ? actual : null, // No actuals for future dates
            forecast,
            lowerBound,
            upperBound,
            isFuture: i >= days - 7
        })
    }

    return data
}

// Generate mock SKU data
function generateMockSKUs(count = 50) {
    const categories = ['DAIRY', 'BEVERAGES', 'SNACKS', 'HOMECARE', 'PERSONALCARE']
    const brands = ['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D', 'Brand_E']
    const abcClasses = ['A', 'B', 'C']

    return Array.from({ length: count }, (_, i) => ({
        id: `SKU_${String(i + 1).padStart(4, '0')}`,
        name: `${brands[Math.floor(Math.random() * brands.length)]}_${categories[Math.floor(Math.random() * categories.length)]}_${i + 1}`,
        category: categories[Math.floor(Math.random() * categories.length)],
        brand: brands[Math.floor(Math.random() * brands.length)],
        abcClass: abcClasses[Math.floor(Math.random() * abcClasses.length)],
        basePrice: Math.round(50 + Math.random() * 450),
        avgDemand: Math.round(100 + Math.random() * 900),
        stockLevel: Math.round(500 + Math.random() * 2000),
        reorderPoint: Math.round(200 + Math.random() * 500)
    }))
}

// Generate mock inventory data
function generateMockInventory(skus) {
    return skus.map(sku => {
        const stockLevel = Math.round(100 + Math.random() * 2000)
        const reorderPoint = Math.round(200 + Math.random() * 400)
        const safetyStock = Math.round(100 + Math.random() * 200)
        const avgDailyDemand = Math.round(20 + Math.random() * 80)
        const mos = stockLevel / (avgDailyDemand * 30)

        let status = 'healthy'
        if (stockLevel < safetyStock) status = 'critical'
        else if (stockLevel < reorderPoint) status = 'warning'
        else if (mos > 3) status = 'overstock'

        return {
            ...sku,
            stockLevel,
            reorderPoint,
            safetyStock,
            avgDailyDemand,
            mos: mos.toFixed(2),
            status,
            lastReplenishment: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            nextReplenishment: new Date(Date.now() + Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
        }
    })
}

// Generate mock KPI data
function generateMockKPIs() {
    return {
        mape: {
            value: (8 + Math.random() * 4).toFixed(1),
            change: (-15 - Math.random() * 10).toFixed(1),
            trend: 'positive'
        },
        stockoutRate: {
            value: (2 + Math.random() * 3).toFixed(1),
            change: (-25 - Math.random() * 10).toFixed(1),
            trend: 'positive'
        },
        inventoryTurnover: {
            value: (6 + Math.random() * 4).toFixed(1),
            change: (10 + Math.random() * 5).toFixed(1),
            trend: 'positive'
        },
        serviceLevel: {
            value: (94 + Math.random() * 4).toFixed(1),
            change: (2 + Math.random() * 2).toFixed(1),
            trend: 'positive'
        },
        forecastBias: {
            value: (-2 + Math.random() * 4).toFixed(1),
            change: (-5 - Math.random() * 3).toFixed(1),
            trend: 'positive'
        },
        revenueImpact: {
            value: (12 + Math.random() * 8).toFixed(1),
            change: (8 + Math.random() * 5).toFixed(1),
            trend: 'positive'
        }
    }
}

// Generate mock alerts
function generateMockAlerts() {
    const alertTypes = [
        { type: 'stockout', severity: 'danger', message: 'Critical stock level for SKU_0012 at London DC' },
        { type: 'demand_spike', severity: 'warning', message: 'Unusual demand spike detected for DAIRY category' },
        { type: 'model_drift', severity: 'info', message: 'Model performance degradation detected - consider retraining' },
        { type: 'data_quality', severity: 'warning', message: 'Missing data detected for 3 locations in last 24 hours' },
        { type: 'forecast', severity: 'success', message: 'Weekly forecast generated successfully for all SKUs' },
        { type: 'promotion', severity: 'info', message: 'Upcoming Christmas promotion may increase demand by 150%' }
    ]

    return alertTypes.map((alert, i) => ({
        id: i + 1,
        ...alert,
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
        read: Math.random() > 0.5
    }))
}

// Generate mock model status
function generateMockModelStatus() {
    return {
        currentVersion: '2.4.1',
        lastTrained: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        nextScheduledTraining: new Date(Date.now() + 4 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'healthy',
        metrics: {
            mape: 8.5,
            mae: 45.2,
            rmse: 62.3,
            r2: 0.92
        },
        trainingHistory: Array.from({ length: 10 }, (_, i) => ({
            version: `2.${4 - Math.floor(i / 3)}.${(10 - i) % 3}`,
            date: new Date(Date.now() - (i + 1) * 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            mape: (8 + Math.random() * 4).toFixed(2),
            status: i === 0 ? 'active' : 'archived'
        })),
        featureImportance: [
            { feature: 'lag_7_demand', importance: 0.25 },
            { feature: 'festival_flag', importance: 0.18 },
            { feature: 'price', importance: 0.15 },
            { feature: 'weather_factor', importance: 0.12 },
            { feature: 'promo_flag', importance: 0.10 },
            { feature: 'day_of_week', importance: 0.08 },
            { feature: 'rolling_30_mean', importance: 0.07 },
            { feature: 'competitor_promo', importance: 0.05 }
        ]
    }
}

// Generate mock data jobs
function generateMockDataJobs() {
    const jobTypes = ['ETL Pipeline', 'Forecast Generation', 'Data Ingestion', 'Feature Engineering', 'Model Training']
    const statuses = ['completed', 'running', 'failed', 'pending']

    return Array.from({ length: 15 }, (_, i) => {
        const status = i < 2 ? 'running' : i < 12 ? 'completed' : i < 14 ? 'failed' : 'pending'
        return {
            id: `JOB_${String(i + 1).padStart(6, '0')}`,
            type: jobTypes[Math.floor(Math.random() * jobTypes.length)],
            status,
            startTime: new Date(Date.now() - (i + 1) * 2 * 60 * 60 * 1000).toISOString(),
            endTime: status === 'running' ? null : new Date(Date.now() - i * 2 * 60 * 60 * 1000).toISOString(),
            duration: status === 'running' ? null : Math.round(5 + Math.random() * 55),
            records: Math.round(10000 + Math.random() * 90000),
            progress: status === 'running' ? Math.round(20 + Math.random() * 60) : status === 'completed' ? 100 : 0
        }
    })
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

export const api = {
    // Dashboard
    async getDashboardKPIs() {
        await delay(500)
        return generateMockKPIs()
    },

    async getAlerts() {
        await delay(300)
        return generateMockAlerts()
    },

    async getDemandTrend(days = 30) {
        await delay(400)
        return generateMockForecasts(days)
    },

    // Forecasts
    async getForecasts(params = {}) {
        await delay(600)
        return {
            data: generateMockForecasts(params.days || 90),
            skus: generateMockSKUs(50)
        }
    },

    async getForecastBySKU(skuId) {
        await delay(400)
        return {
            sku: skuId,
            data: generateMockForecasts(90)
        }
    },

    // Inventory
    async getInventory() {
        await delay(500)
        const skus = generateMockSKUs(50)
        return generateMockInventory(skus)
    },

    // Scenarios
    async runScenario(params) {
        await delay(1500) // Simulate longer computation
        const baseline = generateMockForecasts(30)
        const scenario = baseline.map(d => ({
            ...d,
            forecast: Math.round(d.forecast * (1 + (params.demandChange || 0) / 100)),
            lowerBound: Math.round(d.lowerBound * (1 + (params.demandChange || 0) / 100)),
            upperBound: Math.round(d.upperBound * (1 + (params.demandChange || 0) / 100))
        }))

        return {
            baseline,
            scenario,
            impact: {
                demandChange: params.demandChange || 0,
                revenueImpact: ((params.demandChange || 0) * 1.2).toFixed(1),
                stockoutRisk: params.demandChange > 20 ? 'High' : params.demandChange > 10 ? 'Medium' : 'Low'
            }
        }
    },

    // Data Jobs
    async getDataJobs() {
        await delay(400)
        return generateMockDataJobs()
    },

    async uploadData(file) {
        await delay(2000)
        return {
            success: true,
            message: `File ${file.name} uploaded successfully`,
            jobId: `JOB_${Date.now()}`
        }
    },

    // Models
    async getModelStatus() {
        await delay(400)
        return generateMockModelStatus()
    },

    async triggerRetrain() {
        await delay(1000)
        return {
            success: true,
            message: 'Model retraining triggered successfully',
            jobId: `TRAIN_${Date.now()}`
        }
    },

    // Export
    async exportData(format = 'csv') {
        await delay(500)
        // In production, this would return a download URL
        return {
            success: true,
            downloadUrl: '#',
            message: `Export to ${format.toUpperCase()} initiated`
        }
    }
}

export default api
