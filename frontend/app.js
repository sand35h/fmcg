/**
 * FMCG Dashboard JavaScript
 * Handles API communication and UI updates
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const API_BASE_URL = 'http://localhost:8000';

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const elements = {
    apiStatus: document.getElementById('apiStatus'),
    wmapeH1: document.getElementById('wmapeH1'),
    wmapeH7: document.getElementById('wmapeH7'),
    wmapeH14: document.getElementById('wmapeH14'),
    featureCount: document.getElementById('featureCount'),
    forecastForm: document.getElementById('forecastForm'),
    predictionResult: document.getElementById('predictionResult'),
    predictionValue: document.getElementById('predictionValue'),
    predictionMeta: document.getElementById('predictionMeta'),
    forecastDate: document.getElementById('forecastDate')
};

// Chart instance
let featureChart = null;

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error('API not healthy');
        const data = await response.json();
        updateApiStatus(true, `Connected (${data.models_loaded} models)`);
        return data;
    } catch (error) {
        updateApiStatus(false, 'API Offline');
        console.error('Health check failed:', error);
        return null;
    }
}

async function getModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (!response.ok) throw new Error('Failed to get model info');
        return await response.json();
    } catch (error) {
        console.error('Failed to get model info:', error);
        return null;
    }
}

async function getFeatureImportance(topN = 15) {
    try {
        const response = await fetch(`${API_BASE_URL}/feature-importance?top_n=${topN}`);
        if (!response.ok) throw new Error('Failed to get feature importance');
        return await response.json();
    } catch (error) {
        console.error('Failed to get feature importance:', error);
        return null;
    }
}

async function getPrediction(skuId, locationId, date, horizon, priceChangePct = 0, isPromo = false) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sku_id: skuId,
                location_id: locationId,
                date: date,
                horizon: parseInt(horizon),
                price_change_pct: priceChangePct,
                is_promo: isPromo
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Prediction failed:', error);
        throw error;
    }
}

// =============================================================================
// UI UPDATE FUNCTIONS
// =============================================================================

function updateApiStatus(connected, message) {
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    const statusText = elements.apiStatus.querySelector('.status-text');
    statusDot.className = 'status-dot ' + (connected ? 'connected' : 'error');
    statusText.textContent = message;
}

function updateKPIs(modelInfo) {
    if (!modelInfo) return;
    elements.wmapeH1.textContent = modelInfo.wmape_h1 ? `${modelInfo.wmape_h1.toFixed(1)}%` : '--';
    elements.wmapeH7.textContent = modelInfo.wmape_h7 ? `${modelInfo.wmape_h7.toFixed(1)}%` : '--';
    elements.wmapeH14.textContent = modelInfo.wmape_h14 ? `${modelInfo.wmape_h14.toFixed(1)}%` : '--';
    elements.featureCount.textContent = modelInfo.feature_count || '--';
}

function updateFeatureChart(features) {
    if (!features || features.length === 0) return;

    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js not loaded');
        return;
    }

    const canvas = document.getElementById('featureChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Destroy existing chart
    if (featureChart) {
        featureChart.destroy();
    }

    // Prepare data
    const labels = features.map(f => f.feature);
    const values = features.map(f => f.importance);

    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance Score',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.8)', // Solid primary color
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            return `Importance: ${context.raw.toLocaleString()}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                y: {
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function showPrediction(result) {
    document.getElementById('predictionResult').textContent = result.prediction.toLocaleString();

    // Confidence Interval
    const intervalText = `${result.confidence_min.toLocaleString()} - ${result.confidence_max.toLocaleString()}`;
    document.getElementById('confInterval').textContent = intervalText;

    // Meta info
    document.getElementById('resHorizon').textContent = result.horizon;
    document.getElementById('resModel').textContent = result.model_version;

    // Inventory & Status
    const stockEl = document.getElementById('currentStock');
    const statusEl = document.getElementById('stockStatus');
    stockEl.textContent = result.current_stock.toLocaleString();
    statusEl.textContent = result.stock_status;

    // Status styling
    statusEl.className = 'badge'; // reset
    if (result.stock_status.includes('Stockout') || result.stock_status.includes('Risk')) {
        statusEl.classList.add('badge-danger');
    } else if (result.stock_status.includes('Low')) {
        statusEl.classList.add('badge-warning');
    } else {
        statusEl.classList.add('badge-success');
    }

    document.getElementById('daysCover').textContent = `${result.days_of_cover} days cover`;

    // Scenario Impact
    const impactEl = document.getElementById('scenarioImpact');
    const impactVal = result.scenario_impact_pct;
    impactEl.textContent = (impactVal > 0 ? '+' : '') + impactVal + '%';
    impactEl.className = 'text-lg font-bold ' + (impactVal > 0 ? 'text-success' : (impactVal < 0 ? 'text-danger' : 'text-muted'));

    // Show card
    document.getElementById('resultCard').classList.remove('hidden');

    // Scroll to result
    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
}

async function loadProducts(query = '') {
    try {
        const response = await fetch(`${API_BASE_URL}/products?query=${encodeURIComponent(query)}`);
        if (!response.ok) return;
        const data = await response.json();

        const datalist = document.getElementById('skuOptions');
        datalist.innerHTML = '';

        data.products.forEach(p => {
            const option = document.createElement('option');
            option.value = p.sku_id;
            option.textContent = `${p.name} (${p.brand})`;
            datalist.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load products:', error);
    }
}

async function loadLocations() {
    try {
        const response = await fetch(`${API_BASE_URL}/locations`);
        if (!response.ok) return;
        const data = await response.json();

        const datalist = document.getElementById('locOptions');
        datalist.innerHTML = '';

        data.locations.forEach(l => {
            const option = document.createElement('option');
            option.value = l.location_id;
            option.textContent = `${l.city} (${l.type})`;
            datalist.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load locations:', error);
    }
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

// Search debouncing
let searchTimeout;
document.getElementById('skuId').addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        loadProducts(e.target.value);
    }, 300);
});

// Slider visual update
document.getElementById('priceChange').addEventListener('input', (e) => {
    const val = e.target.value;
    document.getElementById('priceVal').textContent = (val > 0 ? '+' : '') + val + '%';
    document.getElementById('priceVal').className = 'text-xs font-mono ' +
        (val > 0 ? 'text-success' : (val < 0 ? 'text-danger' : 'text-primary-400'));
});

elements.forecastForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const skuId = document.getElementById('skuId').value;
    const locationId = document.getElementById('locationId').value;
    const date = document.getElementById('date').value;
    const horizon = document.getElementById('horizon').value;
    const priceChangePct = parseFloat(document.getElementById('priceChange').value);
    const isPromo = document.getElementById('isPromo').checked;

    try {
        const result = await getPrediction(skuId, locationId, date, horizon, priceChangePct, isPromo);
        showPrediction(result);
    } catch (error) {
        alert('Prediction failed: ' + error.message);
    }
});

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    // Set default date
    document.getElementById('date').value = new Date().toISOString().split('T')[0];

    await checkHealth();

    // Load initial data
    loadProducts();
    loadLocations();

    const modelInfo = await getModelInfo();
    updateKPIs(modelInfo);

    const features = await getFeatureImportance(15);
    updateFeatureChart(features);
}

document.addEventListener('DOMContentLoaded', init);
