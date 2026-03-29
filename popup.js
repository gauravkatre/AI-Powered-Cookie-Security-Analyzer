// popup.js

document.addEventListener('DOMContentLoaded', async function () {
    const exportBtn = document.getElementById('exportBtn');
    const statusText = document.getElementById('statusText');
    const spinner = document.getElementById('spinner');
    const currentSite = document.getElementById('currentSite');
    const cookieCount = document.getElementById('cookieCount');
    const sessionCount = document.getElementById('sessionCount');
    const formatJson = document.getElementById('formatJson');
    const analyzerLink = document.getElementById('analyzerLink');

    // Get current tab info and update stats
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab && tab.url) {
            const url = new URL(tab.url);
            currentSite.textContent = url.hostname;
            currentSite.title = tab.url;
            updateCookieStats(tab.url);
        }
    } catch (error) {
        console.error('Error getting tab info:', error);
        currentSite.textContent = 'Unknown site';
    }

    // Analyzer link setup
    analyzerLink.addEventListener('click', function (e) {
        e.preventDefault();
        chrome.tabs.create({ url: 'http://127.0.0.1:5000' });
    });
    
    // --- EXPORT BUTTON LOGIC ---
    exportBtn.addEventListener('click', async () => {
        setStatus('exporting', 'Analyzing page and exporting cookies...');
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            // 1. Get HSTS status from storage (saved by background.js)
            const hstsData = await chrome.storage.session.get(`hsts_status_${tab.id}`);
            const hasHSTS = hstsData[`hsts_status_${tab.id}`] || false;

            // 2. Execute a script in the active tab to get the rest of the page data
            const [scriptResult] = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: getPageContextFromDOM,
            });
            
            const pageSecurityContext = scriptResult.result;
            // Add the HSTS status we retrieved to the context object
            pageSecurityContext.hasHSTS = hasHSTS;
            
            // 3. Get all cookies for the current URL
            const allCookies = await chrome.cookies.getAll({ url: tab.url });
            if (allCookies.length === 0) {
                setStatus('error', 'No cookies found on this site.');
                return;
            }

            // 4. Create the final JSON object for export
            const exportData = {
                pageSecurityContext: pageSecurityContext,
                cookies: allCookies
            };
            const jsonString = formatJson.checked 
                ? JSON.stringify(exportData, null, 2) 
                : JSON.stringify(exportData);

            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const domain = new URL(tab.url).hostname;
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `cookies_${domain}_${timestamp}.json`;

            // 5. Trigger the download
            chrome.downloads.download({
                url: url,
                filename: filename,
                saveAs: true
            }, () => {
                URL.revokeObjectURL(url);
                if (chrome.runtime.lastError) {
                    setStatus('error', 'Export failed: ' + chrome.runtime.lastError.message);
                } else {
                    setStatus('success', `${allCookies.length} cookies exported successfully!`);
                }
            });

        } catch (error) {
            setStatus('error', 'Failed to export: ' + error.message);
            console.error(error);
        }
    });

    // --- HELPER FUNCTIONS ---
    async function updateCookieStats(url) {
        const cookies = await chrome.cookies.getAll({ url });
        cookieCount.textContent = cookies.length;
        sessionCount.textContent = cookies.filter(c => c.session).length;
    }

    function setStatus(type, message) {
        statusText.textContent = message;
        spinner.style.display = type === 'exporting' ? 'block' : 'none';
        statusText.className = type;
    }
});

// This function is injected into the webpage to get data from the DOM.
function getPageContextFromDOM() {
    const hasCSP = document.querySelector("meta[http-equiv='Content-Security-Policy']") !== null;
    
    const scripts = Array.from(document.querySelectorAll('script[src]'));
    const stylesheets = Array.from(document.querySelectorAll('link[rel="stylesheet"][href]'));
    const externalResources = scripts.concat(stylesheets);
    const resourcesWithIntegrity = externalResources.filter(el => el.hasAttribute('integrity'));
    
    const sriCoverage = externalResources.length > 0 
        ? resourcesWithIntegrity.length / externalResources.length 
        : 1.0;

    return {
        hostname: window.location.hostname,
        hasCSP: hasCSP,
        sriCoverage: sriCoverage,
        sslType: 'dv' // This remains a placeholder
    };
}