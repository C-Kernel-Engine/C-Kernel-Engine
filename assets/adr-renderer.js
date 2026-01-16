/**
 * ADR Renderer - Dynamic Architecture Decision Records viewer
 *
 * Features:
 * - Load ADRs from JSON
 * - Search by title, context, decision
 * - Filter by category and status
 * - Pagination for long lists
 * - Collapsible sections
 * - URL hash navigation
 */

class ADRRenderer {
    constructor(containerId, jsonPath) {
        this.container = document.getElementById(containerId);
        this.jsonPath = jsonPath || 'assets/adr-data.json';
        this.adrs = [];
        this.filteredAdrs = [];
        this.currentPage = 1;
        this.perPage = 5;
        this.searchQuery = '';
        this.categoryFilter = 'all';
        this.statusFilter = 'all';
    }

    async init() {
        try {
            const response = await fetch(this.jsonPath);
            const data = await response.json();
            this.adrs = data.adrs;
            this.metadata = data.metadata;
            this.filteredAdrs = [...this.adrs];
            this.render();
            this.handleHashNavigation();
        } catch (error) {
            console.error('Failed to load ADR data:', error);
            this.container.innerHTML = '<div class="alert alert-error">Failed to load ADR data.</div>';
        }
    }

    render() {
        this.container.innerHTML = `
            ${this.renderHeader()}
            ${this.renderFilters()}
            ${this.renderIndex()}
            ${this.renderADRs()}
            ${this.renderPagination()}
        `;
        this.bindEvents();
    }

    renderHeader() {
        return `
            <h1>Architecture Decision Records</h1>
            <p>Key architectural decisions for the <strong>${this.metadata.project}</strong>
               (${this.metadata.version}). Last updated: ${this.metadata.lastUpdated}.</p>

            <div class="alert alert-info">
                <div class="alert-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>
                        <line x1="12" y1="8" x2="12.01" y2="8"/>
                    </svg>
                </div>
                <div>
                    <strong>About ADRs</strong><br>
                    Architecture Decision Records capture decisions along with their context and consequences.
                    They help prevent re-litigating settled discussions and provide context for new contributors.
                </div>
            </div>
        `;
    }

    renderFilters() {
        const categories = [...new Set(this.adrs.map(a => a.category))];
        const statuses = [...new Set(this.adrs.map(a => a.status))];

        return `
            <div class="adr-filters" style="display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; align-items: center;">
                <div style="flex: 1; min-width: 200px;">
                    <input type="text" id="adr-search" placeholder="Search ADRs..."
                           value="${this.searchQuery}"
                           style="width: 100%; padding: 0.5rem; border: 1px solid #e2e8f0; border-radius: 6px;">
                </div>
                <div>
                    <select id="adr-category" style="padding: 0.5rem; border: 1px solid #e2e8f0; border-radius: 6px;">
                        <option value="all">All Categories</option>
                        ${categories.map(c => `<option value="${c}" ${this.categoryFilter === c ? 'selected' : ''}>${this.capitalize(c)}</option>`).join('')}
                    </select>
                </div>
                <div>
                    <select id="adr-status" style="padding: 0.5rem; border: 1px solid #e2e8f0; border-radius: 6px;">
                        <option value="all">All Status</option>
                        ${statuses.map(s => `<option value="${s}" ${this.statusFilter === s ? 'selected' : ''}>${this.capitalize(s)}</option>`).join('')}
                    </select>
                </div>
                <div style="color: #64748b; font-size: 0.9rem;">
                    Showing ${this.filteredAdrs.length} of ${this.adrs.length} ADRs
                </div>
            </div>
        `;
    }

    renderIndex() {
        const start = (this.currentPage - 1) * this.perPage;
        const pageAdrs = this.filteredAdrs.slice(start, start + this.perPage);

        return `
            <h2>Decision Index</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Title</th>
                        <th>Category</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${pageAdrs.map(adr => `
                        <tr>
                            <td><a href="#adr-${adr.id}">ADR-${adr.id}</a></td>
                            <td>${adr.title}</td>
                            <td>${this.renderBadge(adr.category, 'category')}</td>
                            <td>${this.renderBadge(adr.status, 'status')}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    renderADRs() {
        const start = (this.currentPage - 1) * this.perPage;
        const pageAdrs = this.filteredAdrs.slice(start, start + this.perPage);

        return pageAdrs.map(adr => this.renderADR(adr)).join('');
    }

    renderADR(adr) {
        return `
            <details class="adr" id="adr-${adr.id}">
                <summary>
                    <span class="adr-title">ADR-${adr.id}: ${adr.title}</span>
                    <span class="adr-status">${this.renderBadge(adr.status, 'status')}</span>
                </summary>
                <div class="adr-content">
                    <div class="adr-grid">
                        <div><strong>Status:</strong> ${this.capitalize(adr.status)}</div>
                        <div><strong>Date:</strong> ${adr.date}</div>
                        <div><strong>Category:</strong> ${this.renderBadge(adr.category, 'category')}</div>
                        <div><strong>Related:</strong> ${adr.related.map(r => `<a href="#adr-${r}">ADR-${r}</a>`).join(', ') || 'None'}</div>
                    </div>

                    ${adr.supersedes.length ? `
                        <div class="alert alert-warning" style="margin-bottom: 1rem;">
                            <strong>Supersedes:</strong> ${adr.supersedes.join(', ')}
                        </div>
                    ` : ''}

                    <h3>Context</h3>
                    <p>${adr.context}</p>

                    <h3>Decision</h3>
                    <p>${adr.decision}</p>

                    ${adr.pipeline ? this.renderPipeline(adr.pipeline) : ''}

                    ${adr.diagram ? `
                        <div class="card" style="background: var(--dark-card, #323232); border: 1px solid var(--grey, #454545);">
                            <pre style="background: var(--code-bg, #1a1a1a); margin: 0; color: #e2e8f0; white-space: pre; overflow-x: auto;">${adr.diagram}</pre>
                        </div>
                    ` : ''}

                    ${adr.codeExample ? `
                        <h4>Example</h4>
                        <div class="card">
                            <pre><code>${this.escapeHtml(adr.codeExample)}</code></pre>
                        </div>
                    ` : ''}

                    <h3>Consequences</h3>
                    <div class="consequences-grid">
                        <div class="consequence-card benefit">
                            <h4>Benefits</h4>
                            <ul>
                                ${adr.consequences.benefits.map(b => `<li>${b}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="consequence-card cost">
                            <h4>Costs</h4>
                            <ul>
                                ${adr.consequences.costs.map(c => `<li>${c}</li>`).join('')}
                            </ul>
                        </div>
                    </div>

                    ${adr.metrics ? `
                        <h4>Metrics</h4>
                        <table>
                            ${Object.entries(adr.metrics).map(([k, v]) => `<tr><td>${k}</td><td><code>${v}</code></td></tr>`).join('')}
                        </table>
                    ` : ''}

                    ${adr.kernelExamples ? `
                        <h4>Kernel Selection Examples</h4>
                        <table>
                            <thead><tr><th>Context</th><th>Selected Kernel</th></tr></thead>
                            <tbody>
                                ${Object.entries(adr.kernelExamples).map(([k, v]) => `<tr><td>${k}</td><td><code>${v}</code></td></tr>`).join('')}
                            </tbody>
                        </table>
                    ` : ''}
                </div>
            </details>
        `;
    }

    renderPipeline(stages) {
        const colorMap = {
            purple: { border: '#8b5cf6', header: '#a78bfa' },
            orange: { border: '#f59e0b', header: '#fbbf24' },
            green: { border: '#22c55e', header: '#4ade80' },
            blue: { border: '#07adf8', header: '#38bdf8' }
        };

        return `
            <div class="pipeline-flowchart">
                <div class="pipeline-title">IR v4 PIPELINE</div>
                <div class="pipeline-input">
                    <code>config.json</code> + <code>template.yaml</code>
                </div>
                ${stages.map((stage, i) => {
                    const colors = colorMap[stage.color] || colorMap.orange;
                    return `
                        <div class="pipeline-arrow">▼</div>
                        <div class="pipeline-stage" style="border-left: 3px solid ${colors.border};">
                            <div class="stage-header" style="color: ${colors.header};">${stage.name}</div>
                            <div class="stage-subtitle">${stage.subtitle}</div>
                            <div class="stage-desc">${stage.desc}</div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    renderPagination() {
        const totalPages = Math.ceil(this.filteredAdrs.length / this.perPage);
        if (totalPages <= 1) return '';

        const pages = [];
        for (let i = 1; i <= totalPages; i++) {
            pages.push(i);
        }

        return `
            <div class="adr-pagination" style="display: flex; justify-content: center; gap: 0.5rem; margin-top: 2rem;">
                <button class="adr-page-btn" data-page="prev" ${this.currentPage === 1 ? 'disabled' : ''}>
                    ← Prev
                </button>
                ${pages.map(p => `
                    <button class="adr-page-btn ${p === this.currentPage ? 'active' : ''}" data-page="${p}">
                        ${p}
                    </button>
                `).join('')}
                <button class="adr-page-btn" data-page="next" ${this.currentPage === totalPages ? 'disabled' : ''}>
                    Next →
                </button>
            </div>
        `;
    }

    renderBadge(value, type) {
        const colors = {
            category: {
                design: '#8b5cf6',
                performance: '#f59e0b',
                security: '#ef4444',
                infrastructure: '#0ea5e9'
            },
            status: {
                accepted: '#22c55e',
                proposed: '#f59e0b',
                deprecated: '#ef4444',
                superseded: '#64748b'
            }
        };
        const color = colors[type]?.[value] || '#64748b';
        return `<span class="badge" style="background: ${color}; color: white;">${this.capitalize(value)}</span>`;
    }

    bindEvents() {
        // Search
        const searchInput = document.getElementById('adr-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchQuery = e.target.value.toLowerCase();
                this.applyFilters();
            });
        }

        // Category filter
        const categorySelect = document.getElementById('adr-category');
        if (categorySelect) {
            categorySelect.addEventListener('change', (e) => {
                this.categoryFilter = e.target.value;
                this.applyFilters();
            });
        }

        // Status filter
        const statusSelect = document.getElementById('adr-status');
        if (statusSelect) {
            statusSelect.addEventListener('change', (e) => {
                this.statusFilter = e.target.value;
                this.applyFilters();
            });
        }

        // Pagination
        document.querySelectorAll('.adr-page-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const page = e.target.dataset.page;
                if (page === 'prev') {
                    this.currentPage = Math.max(1, this.currentPage - 1);
                } else if (page === 'next') {
                    const totalPages = Math.ceil(this.filteredAdrs.length / this.perPage);
                    this.currentPage = Math.min(totalPages, this.currentPage + 1);
                } else {
                    this.currentPage = parseInt(page);
                }
                this.render();
            });
        });

        // Hash navigation for direct links
        document.querySelectorAll('a[href^="#adr-"]').forEach(link => {
            link.addEventListener('click', (e) => {
                const id = e.target.getAttribute('href').replace('#adr-', '');
                this.openADR(id);
            });
        });
    }

    applyFilters() {
        this.filteredAdrs = this.adrs.filter(adr => {
            // Search filter
            if (this.searchQuery) {
                const searchText = `${adr.title} ${adr.context} ${adr.decision}`.toLowerCase();
                if (!searchText.includes(this.searchQuery)) return false;
            }
            // Category filter
            if (this.categoryFilter !== 'all' && adr.category !== this.categoryFilter) return false;
            // Status filter
            if (this.statusFilter !== 'all' && adr.status !== this.statusFilter) return false;
            return true;
        });
        this.currentPage = 1;
        this.render();
    }

    openADR(id) {
        const details = document.getElementById(`adr-${id}`);
        if (details) {
            details.open = true;
            details.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    handleHashNavigation() {
        const hash = window.location.hash;
        if (hash && hash.startsWith('#adr-')) {
            const id = hash.replace('#adr-', '');
            setTimeout(() => this.openADR(id), 100);
        }
    }

    capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('adr-container');
    if (container) {
        const renderer = new ADRRenderer('adr-container');
        renderer.init();
    }
});
