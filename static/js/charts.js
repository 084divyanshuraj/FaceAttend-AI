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
            gradient.addColorStop(0, 'rgba(0, 144, 231, 0.45)');
            gradient.addColorStop(0.6, 'rgba(255, 20, 147, 0.12)');
            gradient.addColorStop(1, 'rgba(255, 20, 147, 0.0)');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels.length > 0 ? labels : ['No Data'],
                    datasets: [{
                        label: 'Present Students',
                        data: values.length > 0 ? values : [0],
                        borderColor: '#0090E7',
                        backgroundColor: gradient,
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: '#FF1493',
                        pointBorderColor: '#FFFFFF',
                        pointBorderWidth: 2,
                        pointRadius: 4.5,
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#00D2FF',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#11141D',
                            titleColor: '#00D2FF',
                            bodyColor: '#F8FAFC',
                            borderColor: '#1E2330',
                            borderWidth: 1.5,
                            padding: 12,
                            displayColors: false,
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.03)' },
                            ticks: { color: '#94A3B8', font: { size: 11 } }
                        },
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.03)' },
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
                            '#0090E7', // Cerulean Blue
                            '#FF1493', // Neon Rose
                            '#00D2FF', // Electric Cyan
                            '#FF4DA6', // Bright Rose
                            '#00B4D8', // Light Cerulean
                            '#FF85C0'  // Soft Rose
                        ],
                        borderWidth: 2,
                        borderColor: '#11141D',
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

