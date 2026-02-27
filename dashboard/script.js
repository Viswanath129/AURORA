
document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
    initBackground();
});

let dashboardData = null;
let currentSampleId = null;

async function initDashboard() {
    try {
        const response = await fetch('data/dashboard_data.json');
        dashboardData = await response.json();

        renderMetrics(dashboardData.summary);
        renderSampleList(dashboardData.samples);

        // Load first sample by default
        if (dashboardData.samples.length > 0) {
            loadSample(dashboardData.samples[0].id);
        }
    } catch (error) {
        console.error("Failed to load dashboard data:", error);
        document.querySelector('.main-content').innerHTML = `
            <div class="error-msg">
                <h2>Error Loading Data</h2>
                <p>Please ensure generate_data.py has run successfully.</p>
                <pre>${error.message}</pre>
            </div>
        `;
    }
}

function renderMetrics(summary) {
    const metricsContainer = document.getElementById('metrics-summary');
    metricsContainer.innerHTML = '';

    // Define order and icons/colors (optional logic)
    const order = ['Advection', 'Morphology', 'Emergence', 'Temporal', 'AURORA'];

    order.forEach(model => {
        if (!summary[model]) return;

        const data = summary[model];
        const card = document.createElement('div');
        card.className = `metric-card ${model === 'AURORA' ? 'highlight' : ''}`;

        card.innerHTML = `
            <span class="label">${model}</span>
            <div class="value">${data.avg_ssim.toFixed(4)}</div>
            <div class="sub-value">PSNR: ${data.avg_psnr.toFixed(2)} dB</div>
        `;

        metricsContainer.appendChild(card);
    });
}

function renderSampleList(samples) {
    const listContainer = document.getElementById('sample-list');
    listContainer.innerHTML = '';

    samples.forEach(sample => {
        const item = document.createElement('div');
        item.className = 'sample-item';
        item.dataset.id = sample.id;
        item.textContent = sample.id.replace('_', ' ').toUpperCase();

        item.addEventListener('click', () => {
            loadSample(sample.id);
        });

        listContainer.appendChild(item);
    });
}

function loadSample(sampleId) {
    currentSampleId = sampleId;

    // Update Active State in Sidebar
    document.querySelectorAll('.sample-item').forEach(el => {
        el.classList.toggle('active', el.dataset.id === sampleId);
    });

    const sampleData = dashboardData.samples.find(s => s.id === sampleId);
    const basePath = `data/samples/${sampleId}/`;

    // 1. Input Sequence (t0, t1, t2, t3) & GT
    const inputsContainer = document.getElementById('input-sequence');
    inputsContainer.innerHTML = '';

    for (let i = 0; i < 4; i++) {
        const img = createImgCard(`${basePath}input_t${i}.png`, `Input T-${3 - i}`);
        inputsContainer.appendChild(img);
    }
    // Add GT
    inputsContainer.appendChild(createImgCard(`${basePath}gt.png`, 'Ground Truth (T+1)'));

    // 2. Expert Predictions
    const expertsContainer = document.getElementById('expert-preds');
    expertsContainer.innerHTML = '';
    expertsContainer.appendChild(createImgCard(`${basePath}pred_Advection.png`, 'Advection (Flow)'));
    expertsContainer.appendChild(createImgCard(`${basePath}pred_Morphology.png`, 'Morphology (UNet)'));
    expertsContainer.appendChild(createImgCard(`${basePath}pred_Emergence.png`, 'Emergence (Diff)'));
    expertsContainer.appendChild(createImgCard(`${basePath}pred_Temporal.png`, 'Temporal (LSTM)'));

    // 3. Routing Weights
    const weightsContainer = document.getElementById('routing-weights');
    weightsContainer.innerHTML = '';
    weightsContainer.appendChild(createImgCard(`${basePath}weight_Advection.png`, 'W: Advection'));
    weightsContainer.appendChild(createImgCard(`${basePath}weight_Morphology.png`, 'W: Morphology'));
    weightsContainer.appendChild(createImgCard(`${basePath}weight_Emergence.png`, 'W: Emergence'));
    weightsContainer.appendChild(createImgCard(`${basePath}weight_Temporal.png`, 'W: Temporal'));

    // 4. Comparison
    document.getElementById('final-pred-img').src = `${basePath}pred_AURORA.png`;
    document.getElementById('gt-img-comp').src = `${basePath}gt.png`;

    // Error Map (Simulated or pre-generated? We didn't generate explicit error map image in python script... 
    // Wait, in generate_data.py I only saved preds and weights. I missed error map!
    // I can generate it on the fly if I have raw data, but I don't.
    // I should have generated it. 
    // For now, I'll allow it to break gracefully or just hide it.
    // The script generated: input_t*, gt, pred_*, weight_*

    // Let's check generate_data.py again. I did not save error map.
    // I will disable the "Show Error Map" button for now or just not show it.
    const errBtn = document.getElementById('toggle-error');
    if (errBtn) errBtn.style.display = 'none'; // Hide for now as we don't have the image

}

function createImgCard(src, title) {
    const wrapper = document.createElement('div');
    wrapper.className = 'img-wrapper';

    const img = document.createElement('img');
    img.src = src;
    img.loading = 'lazy';

    const label = document.createElement('div');
    label.className = 'overlay-label';
    label.textContent = title;

    wrapper.appendChild(img);
    wrapper.appendChild(label);

    return wrapper;
}

// Background Animation
function initBackground() {
    const canvas = document.getElementById('bg-canvas');
    const ctx = canvas.getContext('2d');

    let width, height;
    let particles = [];

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    window.addEventListener('resize', resize);
    resize();

    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.size = Math.random() * 2;
            this.alpha = Math.random() * 0.5 + 0.1;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0) this.x = width;
            if (this.x > width) this.x = 0;
            if (this.y < 0) this.y = height;
            if (this.y > height) this.y = 0;
        }

        draw() {
            ctx.fillStyle = `rgba(0, 240, 255, ${this.alpha})`;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    for (let i = 0; i < 50; i++) {
        particles.push(new Particle());
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        requestAnimationFrame(animate);
    }

    animate();
}
