/**
 * FaceAttend AI - Dashboard Charts Manager
 * Renders modern, responsive Chart.js visual analytics.
 */

class DashboardCharts {
    static async initTrendChart(canvasId = 'trendChart') {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        try {
            const res = await fetch('/api/analytics/daily-trend');
            const data = await res.json();
            if (!data.success) return;

            const trend = data.trend || [];
            const labels = trend.map(t => `${t.session_date} (${t.course_code})`);
            const values = trend.map(t => t.present_count);

            const ctx = canvas.getContext('2d');
            const gradient = ctx.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
            gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels.length > 0 ? labels : ['No Data'],
                    datasets: [{
                        label: 'Present Students',
                        data: values.length > 0 ? values : [0],
                        borderColor: '#6366F1',
                        backgroundColor: gradient,
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: '#6366F1',
                        pointBorderColor: '#FFFFFF',
                        pointBorderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#1E293B',
                            titleColor: '#F8FAFC',
                            bodyColor: '#94A3B8',
                            borderColor: '#334155',
                            borderWidth: 1,
                            padding: 10,
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.04)' },
                            ticks: { color: '#94A3B8', font: { size: 11 } }
                        },
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.04)' },
                            ticks: { color: '#94A3B8', stepSize: 1, font: { size: 11 } }
                        }
                    }
                }
            });
        } catch (e) {
            console.error('Failed to initialize trend chart:', e);
        }
    }

    static async initDepartmentChart(canvasId = 'deptChart') {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        try {
            const res = await fetch('/api/analytics/department-breakdown');
            const data = await res.json();
            if (!data.success) return;

            const depts = data.departments || [];
            const labels = depts.map(d => d.department);
            const values = depts.map(d => d.attendance_count);

            const ctx = canvas.getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels.length > 0 ? labels : ['No Data'],
                    datasets: [{
                        data: values.length > 0 ? values : [1],
                        backgroundColor: [
                            '#6366F1',
                            '#10B981',
                            '#06B6D4',
                            '#F59E0B',
                            '#EC4899',
                            '#8B5CF6'
                        ],
                        borderWidth: 2,
                        borderColor: '#1E293B',
                        hoverOffset: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#94A3B8',
                                padding: 16,
                                font: { size: 11 }
                            }
                        }
                    },
                    cutout: '72%'
                }
            });
        } catch (e) {
            console.error('Failed to initialize department chart:', e);
        }
    }
}

