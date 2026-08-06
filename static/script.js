document.addEventListener('DOMContentLoaded', () => {
    // API Configuration (Base URL dynamically resolved)
    const API_BASE = window.location.origin;

    // Elements
    const queryInput = document.getElementById('query-input');
    const searchBtn = document.getElementById('search-btn');
    const btnText = document.getElementById('btn-text');
    const searchSpinner = document.getElementById('search-spinner');
    
    const classSelect = document.getElementById('class-select');
    const subjectSelect = document.getElementById('subject-select');
    const clearFiltersBtn = document.getElementById('clear-filters-btn');
    
    const serverStatus = document.getElementById('server-status');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    const resultsPanel = document.getElementById('results-panel');
    const statLatency = document.getElementById('stat-latency');
    const cacheBadgeContainer = document.getElementById('cache-badge-container');
    const answerContent = document.getElementById('answer-content');
    const sourcesContainer = document.getElementById('sources-container');

    // Load Metadata Filters
    async function loadMetadata() {
        try {
            const response = await fetch(`${API_BASE}/api/metadata`);
            if (!response.ok) throw new Error('Metadata load failed');
            const data = await response.json();
            
            // Populate classes
            data.classes.forEach(cls => {
                const opt = document.createElement('option');
                opt.value = cls;
                opt.textContent = `Class ${cls}`;
                classSelect.appendChild(opt);
            });
            
            // Populate subjects
            data.subjects.forEach(sub => {
                const opt = document.createElement('option');
                opt.value = sub;
                // Capitalize first letter
                opt.textContent = sub.charAt(0).toUpperCase() + sub.slice(1);
                subjectSelect.appendChild(opt);
            });
            
            // Connection successful status update
            serverStatus.textContent = 'Mitra Online';
            serverStatus.classList.add('online');
        } catch (err) {
            console.error('Failed to load server metadata:', err);
            serverStatus.textContent = 'Server Offline';
            serverStatus.style.borderColor = 'rgba(235, 94, 85, 0.2)';
            serverStatus.style.color = '#eb5e55';
            serverStatus.style.backgroundColor = 'rgba(235, 94, 85, 0.1)';
        }
    }

    // Execute Search
    async function executeSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        // Reset UI States
        searchBtn.disabled = true;
        queryInput.disabled = true;
        btnText.style.display = 'none';
        searchSpinner.style.display = 'block';
        loadingIndicator.style.display = 'flex';
        resultsPanel.style.display = 'none';
        cacheBadgeContainer.innerHTML = '';

        // Build Payload
        const payload = {
            query: query,
            filters: {
                class: classSelect.value || null,
                subject: subjectSelect.value || null
            }
        };

        try {
            const response = await fetch(`${API_BASE}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Query execution failed');
            }

            const data = await response.json();
            renderResults(data);
        } catch (err) {
            console.error('Mitra query error:', err);
            alert(`Error: ${err.message}`);
        } finally {
            // Restore UI States
            searchBtn.disabled = false;
            queryInput.disabled = false;
            btnText.style.display = 'inline';
            searchSpinner.style.display = 'none';
            loadingIndicator.style.display = 'none';
        }
    }

    // Render Results on Page
    function renderResults(data) {
        // 1. Latency & Caching Stats
        statLatency.textContent = `${data.metadata.latency_ms} ms`;
        
        if (data.metadata.is_cached) {
            cacheBadgeContainer.innerHTML = `
                <span class="cache-badge" title="Retrieved instantly from SQLite local database. Similarity: ${data.metadata.similarity}">
                    ⚡ Semantic Cache Hit (${Math.round(data.metadata.similarity * 100)}% match)
                </span>
            `;
        }

        // 2. Render Answer (Basic formatting helper)
        answerContent.innerHTML = formatAnswer(data.answer);

        // 3. Render Source Materials
        sourcesContainer.innerHTML = '';
        if (data.sources && data.sources.length > 0) {
            data.sources.forEach(src => {
                const card = document.createElement('div');
                card.className = 'source-card';
                card.innerHTML = `
                    <div class="source-header">
                        <div class="pdf-icon">PDF</div>
                        <div class="source-title" title="${src}">${src}</div>
                    </div>
                    <div class="source-meta">Chapter reference context</div>
                `;
                sourcesContainer.appendChild(card);
            });
        } else {
            sourcesContainer.innerHTML = '<p class="text-muted">No sources referenced.</p>';
        }

        resultsPanel.style.display = 'block';
    }

    // Simple formatting for bold, bullets, and linebreaks
    function formatAnswer(text) {
        // Escape HTML
        let escaped = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
            
        // Render Bolds: **text** -> <strong>text</strong>
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Render Bullet Points
        escaped = escaped.split('\n').map(line => {
            if (line.trim().startsWith('* ')) {
                return `<li>${line.trim().substring(2)}</li>`;
            }
            if (line.trim().startsWith('- ')) {
                return `<li>${line.trim().substring(2)}</li>`;
            }
            return line;
        }).join('\n');

        return escaped;
    }

    // Event Bindings
    searchBtn.addEventListener('click', executeSearch);
    
    // Command/Ctrl + Enter to trigger ask
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            executeSearch();
        }
    });

    clearFiltersBtn.addEventListener('click', () => {
        classSelect.value = '';
        subjectSelect.value = '';
    });

    // Initialize
    loadMetadata();
});
