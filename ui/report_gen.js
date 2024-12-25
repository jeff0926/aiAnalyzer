class ReportGenerator {
    constructor() {
        this.templates = {
            executive: this.executiveSummaryTemplate,
            technical: this.technicalReportTemplate,
            security: this.securityReportTemplate
        };
    }

    generateReport(analysisResults, type = 'executive') {
        const template = this.templates[type] || this.templates.executive;
        return template(analysisResults);
    }

    executiveSummaryTemplate(results) {
        return `
            <div class="max-w-4xl mx-auto">
                <h1 class="text-3xl font-bold mb-8">Executive Summary</h1>
                
                <div class="space-y-6">
                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Overview</h2>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <p class="text-sm text-gray-500">Repository</p>
                                <p class="font-medium">${results.repoName}</p>
                            </div>
                            <div>
                                <p class="text-sm text-gray-500">Analysis Date</p>
                                <p class="font-medium">
                                    ${new Date(results.timestamp).toLocaleDateString()}
                                </p>
                            </div>
                            <div>
                                <p class="text-sm text-gray-500">Components Analyzed</p>
                                <p class="font-medium">${results.componentCount}</p>
                            </div>
                            <div>
                                <p class="text-sm text-gray-500">Issues Found</p>
                                <p class="font-medium">${results.issueCount}</p>
                            </div>
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Key Findings</h2>
                        <div class="space-y-4">
                            ${results.keyFindings.map(finding => `
                                <div class="border-l-4 border-blue-500 pl-4">
                                    <p>${finding}</p>
                                </div>
                            `).join('')}
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Recommendations</h2>
                        <div class="space-y-4">
                            ${results.recommendations.map(rec => `
                                <div class="border-l-4 border-${this.getPriorityColor(rec.priority)} pl-4">
                                    <h3 class="font-medium">${rec.title}</h3>
                                    <p class="text-gray-600 mt-1">${rec.description}</p>
                                </div>
                            `).join('')}
                        </div>
                    </section>
                </div>
            </div>
        `;
    }

    technicalReportTemplate(results) {
        return `
            <div class="max-w-4xl mx-auto">
                <h1 class="text-3xl font-bold mb-8">Technical Analysis Report</h1>
                
                <div class="space-y-8">
                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Architecture Analysis</h2>
                        <div class="space-y-4">
                            <div>
                                <h3 class="font-medium text-gray-700">Pattern</h3>
                                <p class="mt-1">${results.architecture.pattern}</p>
                            </div>
                            
                            <div>
                                <h3 class="font-medium text-gray-700">Components</h3>
                                <ul class="mt-2 space-y-2">
                                    ${results.architecture.components.map(comp => `
                                        <li class="flex items-center">
                                            <span class="w-3 h-3 rounded-full bg-blue-500 mr-2"></span>
                                            ${comp.name} (${comp.type})
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>

                            <div>
                                <h3 class="font-medium text-gray-700">Metrics</h3>
                                <div class="mt-2 grid grid-cols-2 gap-4">
                                    ${Object.entries(results.architecture.metrics).map(([key, value]) => `
                                        <div>
                                            <p class="text-sm text-gray-500">
                                                ${this.formatMetricName(key)}
                                            </p>
                                            <p class="font-medium">${value}</p>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Code Quality</h2>
                        <div class="space-y-4">
                            ${this.generateCodeQualitySection(results.codeQuality)}
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Performance Analysis</h2>
                        <div class="space-y-4">
                            ${this.generatePerformanceSection(results.performance)}
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Technical Debt</h2>
                        <div class="space-y-4">
                            ${this.generateTechnicalDebtSection(results.technicalDebt)}
                        </div>
                    </section>
                </div>
            </div>
        `;
    }

    securityReportTemplate(results) {
        return `
            <div class="max-w-4xl mx-auto">
                <h1 class="text-3xl font-bold mb-8">Security Analysis Report</h1>
                
                <div class="space-y-8">
                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Security Overview</h2>
                        <div class="grid grid-cols-3 gap-4">
                            ${Object.entries(results.security.summary).map(([severity, count]) => `
                                <div>
                                    <p class="text-sm text-gray-500">${severity} Issues</p>
                                    <p class="text-2xl font-semibold ${this.getSeverityColor(severity)}">
                                        ${count}
                                    </p>
                                </div>
                            `).join('')}
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Vulnerabilities</h2>
                        <div class="space-y-4">
                            ${results.security.vulnerabilities.map(vuln => `
                                <div class="border-l-4 border-${this.getSeverityColor(vuln.severity)} p-4">
                                    <div class="flex justify-between items-start">
                                        <div>
                                            <h3 class="font-medium">${vuln.title}</h3>
                                            <p class="text-gray-600 mt-1">${vuln.description}</p>
                                        </div>
                                        <span class="px-2 py-1 text-sm rounded-full bg-${this.getSeverityColor(vuln.severity)}-100 text-${this.getSeverityColor(vuln.severity)}-800">
                                            ${vuln.severity}
                                        </span>
                                    </div>
                                    ${vuln.location ? `
                                        <p class="text-sm text-gray-500 mt-2">
                                            Location: ${vuln.location}
                                        </p>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Compliance Analysis</h2>
                        <div class="space-y-4">
                            ${this.generateComplianceSection(results.security.compliance)}
                        </div>
                    </section>

                    <section class="bg-white p-6 rounded-lg shadow">
                        <h2 class="text-xl font-semibold mb-4">Security Recommendations</h2>
                        <div class="space-y-4">
                            ${results.security.recommendations.map(rec => `
                                <div class="border-l-4 border-${this.getPriorityColor(rec.priority)} p-4">
                                    <h3 class="font-medium">${rec.title}</h3>
                                    <p class="text-gray-600 mt-1">${rec.description}</p>
                                    ${rec.steps ? `
                                        <div class="mt-3">
                                            <h4 class="text-sm font-medium">Implementation Steps</h4>
                                            <ul class="mt-2 list-disc list-inside space-y-1">
                                                ${rec.steps.map(step => `
                                                    <li>${step}</li>
                                                `).join('')}
                                            </ul>
                                        </div>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </section>
                </div>
            </div>
        `;
    }

    generateCodeQualitySection(codeQuality) {
        return `
            <div>
                <h3 class="font-medium text-gray-700">Metrics</h3>
                <div class="mt-2 grid grid-cols-2 gap-4">
                    ${Object.entries(codeQuality.metrics).map(([key, value]) => `
                        <div>
                            <p class="text-sm text-gray-500">${this.formatMetricName(key)}</p>
                            <p class="font-medium">${value}</p>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div>
                <h3 class="font-medium text-gray-700">Issues</h3>
                <div class="mt-2 space-y-3">
                    ${codeQuality.issues.map(issue => `
                        <div>
                            <p class="font-medium">${issue.type}</p>
                            <p class="text-sm text-gray-600">${issue.description}</p>
                            <p class="text-sm text-gray-500 mt-1">Files affected: ${issue.fileCount}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    generatePerformanceSection(performance) {
        return `
            <div>
                <h3 class="font-medium text-gray-700">Performance Metrics</h3>
                <div class="mt-2 grid grid-cols-2 gap-4">
                    ${Object.entries(performance.metrics).map(([key, value]) => `
                        <div>
                            <p class="text-sm text-gray-500">${this.formatMetricName(key)}</p>
                            <p class="font-medium">${value}</p>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div>
                <h3 class="font-medium text-gray-700">Bottlenecks</h3>
                <div class="mt-2 space-y-3">
                    ${performance.bottlenecks.map(bottleneck => `
                        <div>
                            <p class="font-medium">${bottleneck.location}</p>
                            <p class="text-sm text-gray-600">${bottleneck.description}</p>
                            ${bottleneck.impact ? `
                                <p class="text-sm text-gray-500 mt-1">Impact: ${bottleneck.impact}</p>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    generateTechnicalDebtSection(technicalDebt) {
        return `
            <div>
                <h3 class="font-medium text-gray-700">Technical Debt Score</h3>
                <div class="mt-2">
                    <div class="flex items-center">
                        <div class="flex-1 bg-gray-200 rounded-full h-2">
                            <div class="bg-blue-600 h-2 rounded-full" 
                                 style="width: ${technicalDebt.score}%"></div>
                        </div>
                        <span class="ml-4 font-medium">${technicalDebt.score}%</span>
                    </div>
                </div>
            </div>

            <div>
                <h3 class="font-medium text-gray-700">Areas of Concern</h3>
                <div class="mt-2 space-y-3">
                    ${technicalDebt.areas.map(area => `
                        <div>
                            <p class="font-medium">${area.name}</p>
                            <p class="text-sm text-gray-600">${area.description}</p>
                            <div class="mt-2 flex items-center">
                                <div class="flex-1 bg-gray-200 rounded-full h-1">
                                    <div class="bg-blue-600 h-1 rounded-full" 
                                         style="width: ${area.score}%"></div>
                                </div>
                                <span class="ml-4 text-sm font-medium">${area.score}%</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    generateComplianceSection(compliance) {
        return `
            <div>
                ${Object.entries(compliance).map(([framework, data]) => `
                    <div class="mb-6">
                        <h3 class="font-medium text-gray-700">${framework}</h3>
                        <div class="mt-2">
                            <div class="flex items-center">
                                <div class="flex-1 bg-gray-200 rounded-full h-2">
                                    <div class="bg-blue-600 h-2 rounded-full" 
                                         style="width: ${data.score}%"></div>
                                </div>
                                <span class="ml-4 font-medium">${data.score}%</span>
                            </div>
                            ${data.violations.length > 0 ? `
                                <div class="mt-4">
                                    <h4 class="text-sm font-medium">Violations</h4>
                                    <ul class="mt-2 list-disc list-inside space-y-1">
                                        ${data.violations.map(v => `
                                            <li>${v}</li>
                                        `).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    getPriorityColor(priority) {
        switch (priority.toLowerCase()) {
            case 'high':
                return 'red';
            case 'medium':
                return 'yellow';
            case 'low':
                return 'blue';
            default:
                return 'gray';
        }
    }

    getSeverityColor(severity) {
        switch (severity.toLowerCase()) {
            case 'critical':
                return 'red';
            case 'high':
                return 'yellow';
            case 'medium':
                return 'orange';
            case 'low':
                return 'blue';
            default:
                return 'gray';
        }
    }

    formatMetricName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, char => char.toUpperCase());
    }
}

// Export the ReportGenerator class
export default ReportGenerator;
