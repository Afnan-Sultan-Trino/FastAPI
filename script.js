// script.js

// Must match ALLOWED_FOODS in main.py
const FOOD_NAMES = ["ভাত", "ইলিশ", "লাল শাক", "ডিম", "ডাল", "আমলকি", "মুরগি", "গরু", "দুধ চা"];
// Must match VALID_FREQUENCIES in main.py
const FREQUENCIES = ["০ দিন", "১-২ দিন", "৩-৪ দিন", "৫-৬ দিন", "প্রতিদিন"];

// Build one dropdown per food, defaulting to "০ দিন" (never)
function buildFoodFrequencyInputs() {
    const container = document.getElementById('food-frequency-list');
    FOOD_NAMES.forEach(food => {
        const row = document.createElement('div');
        row.className = 'food-row';

        const label = document.createElement('span');
        label.textContent = food;

        const select = document.createElement('select');
        select.id = `freq-${food}`;
        FREQUENCIES.forEach(freq => {
            const opt = document.createElement('option');
            opt.value = freq;
            opt.textContent = freq;
            if (freq === "০ দিন") opt.selected = true;
            select.appendChild(opt);
        });

        row.appendChild(label);
        row.appendChild(select);
        container.appendChild(row);
    });
}
buildFoodFrequencyInputs();

// ---- Global variable to store chart instance ----
let deficiencyChart = null;

async function sendData() {
    const name = document.getElementById('name').value.trim() || 'বন্ধু';
    const age = parseInt(document.getElementById('age').value);
    const gender = document.getElementById('gender').value;
    const weight = parseFloat(document.getElementById('weight').value);
    const height = parseFloat(document.getElementById('height').value);

    // Build {food_name: frequency} for every food, SKIPPING "০ দিন" entries
    const foods = {};
    FOOD_NAMES.forEach(food => {
        const freq = document.getElementById(`freq-${food}`).value;
        if (freq !== "০ দিন") {
            foods[food] = freq;
        }
    });

    const payload = { name, age, gender, weight, height, foods };

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.textContent = '⏳ ডেটা পাঠানো হচ্ছে, দয়া করে অপেক্ষা করুন...';
    resultDiv.className = '';

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
            // ---- Render the result with chart and reminder ----
            renderResult(data);
            resultDiv.className = '';
        } else {
            resultDiv.innerHTML = `<strong>❌ সার্ভার এরর:</strong> <pre>${JSON.stringify(data, null, 2)}</pre>`;
            resultDiv.className = 'error';
        }
    } catch (error) {
        resultDiv.innerHTML = `❌ কানেকশন ফেল! সার্ভার কি চালু আছে? (Error: ${error.message})`;
        resultDiv.className = 'error';
    }
}

// ---- Render Result with Chart + Reminder ----
function renderResult(data) {
    const resultDiv = document.getElementById('result');

    let html = `<strong>✅ সফল!</strong><br>`;
    html += `<strong>মেসেজ:</strong> ${data.message}<br>`;
    html += `<strong>BMI:</strong> ${data.bmi}<br><br>`;

    // ---- Food Images ----
    html += `<strong>আপনার খাবারগুলো:</strong>`;
    html += `<div class="food-grid">`;
    data.foods_with_images.forEach(food => {
        const imgSrc = food.image_url || 'https://via.placeholder.com/100x100?text=No+Image';
        html += `
            <div class="food-item">
                <img src="${imgSrc}" alt="${food.name}"
                    onerror="this.src='https://via.placeholder.com/100x100?text=Image+Not+Found'">
                <span>${food.name}</span>
            </div>
        `;
    });
    html += `</div><br>`;

    // ---- Deficiency Report as Table ----
    html += `<strong>📊 ঘাটতি রিপোর্ট:</strong>`;
    html += `<div style="margin: 10px 0;">`;
    for (const [nutrient, info] of Object.entries(data.deficiencies)) {
        const cls = info.status === 'Deficient' ? 'deficient' : 'adequate';
        html += `
            <div class="deficiency-row">
                <span>${nutrient}</span>
                <span class="${cls}">${info.deficit_percentage}% (${info.status})</span>
            </div>
        `;
    }
    html += `</div>`;

    // ---- Deficiency Bar Chart (Canvas) ----
    html += `<div style="margin: 15px 0; height: 250px;">`;
    html += `<canvas id="deficiencyChart" width="400" height="200"></canvas>`;
    html += `</div>`;

    // ---- Recommendations ----
    if (Object.keys(data.recommendations).length > 0) {
        html += `<br><strong>🍽️ প্রস্তাবিত খাবার:</strong><ul>`;
        for (const [nutrient, foodsRec] of Object.entries(data.recommendations)) {
            if (foodsRec.length > 0) {
                html += `<li>${nutrient}: ${foodsRec.join(', ')}</li>`;
            }
        }
        html += `</ul>`;
    }

    // ---- Reminder / Notification Section ----
    html += `
        <br><hr>
        <div style="margin: 10px 0; padding: 12px; background: #f1f5f9; border-radius: 8px;">
            <strong>⏰ রিমাইন্ডার সেট করুন:</strong><br>
            <div style="display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap;">
                <input type="time" id="reminderTime" value="08:00" style="flex:1; min-width:120px;">
                <button onclick="setReminder()" style="flex:0; padding: 8px 20px; background:#2563eb; color:white; border:none; border-radius:6px; cursor:pointer;">
                    🔔 রিমাইন্ডার সেট করো
                </button>
            </div>
            <div id="reminderStatus" style="margin-top: 6px; font-size: 13px; color: #4a4a6a;"></div>
        </div>
    `;

    resultDiv.innerHTML = html;

    // ---- Draw the chart after DOM update ----
    drawDeficiencyChart(data.deficiencies);
}

// ---- Draw Deficiency Bar Chart using Chart.js ----
function drawDeficiencyChart(deficiencies) {
    const ctx = document.getElementById('deficiencyChart');
    if (!ctx) return;

    // Destroy previous chart if exists
    if (deficiencyChart) {
        deficiencyChart.destroy();
        deficiencyChart = null;
    }

    const labels = Object.keys(deficiencies);
    const values = Object.values(deficiencies).map(d => d.deficit_percentage);
    const statuses = Object.values(deficiencies).map(d => d.status);

    const colors = statuses.map(status =>
        status === 'Deficient' ? '#b60909' : '#22c55e'
    );

    deficiencyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'ঘাটতি (%)',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'ঘাটতি (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// ---- Simple Reminder / Notification System ----
function setReminder() {
    const timeInput = document.getElementById('reminderTime');
    const statusDiv = document.getElementById('reminderStatus');

    if (!timeInput) return;

    const [hours, minutes] = timeInput.value.split(':').map(Number);

    const now = new Date();
    const target = new Date();
    target.setHours(hours, minutes, 0, 0);

    // If the time is already past today, set for tomorrow
    if (target <= now) {
        target.setDate(target.getDate() + 1);
    }

    const delayMs = target - now;
    const delayMinutes = Math.round(delayMs / 60000);

    statusDiv.textContent = `⏳ রিমাইন্ডার সেট করা হয়েছে ${target.toLocaleTimeString('bn-BD', { hour: '2-digit', minute: '2-digit' })}-এর জন্য (${delayMinutes} মিনিট পরে)`;

    // Request permission if not already granted
    if (Notification.permission === 'default') {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                scheduleNotification(target);
            } else {
                statusDiv.textContent = '⚠️ নোটিফিকেশন অনুমতি দেওয়া হয়নি। ব্রাউজার সেটিংস থেকে অনুমতি দিন।';
            }
        });
    } else if (Notification.permission === 'granted') {
        scheduleNotification(target);
    } else {
        statusDiv.textContent = '⚠️ নোটিফিকেশন ব্লক করা আছে। ব্রাউজার সেটিংস থেকে অনুমতি দিন।';
    }
}

function scheduleNotification(targetTime) {
    const now = new Date();
    const delay = targetTime - now;

    if (delay <= 0) {
        document.getElementById('reminderStatus').textContent = '❌ সময় অতিক্রান্ত হয়েছে, আবার চেষ্টা করুন।';
        return;
    }

    // Store reminder in localStorage for persistence across page reloads
    localStorage.setItem('reminderTime', targetTime.toISOString());
    localStorage.setItem('reminderSet', 'true');

    setTimeout(() => {
        // Check if we still have a reminder set (user might have cleared it)
        if (localStorage.getItem('reminderSet') !== 'true') return;

        new Notification('🍽️ পুষ্টি রিমাইন্ডার', {
            body: 'আপনার ঘাটতি পূরণ করতে প্রস্তাবিত খাবারগুলো খেতে মনে রাখবেন!',
            icon: '🍽️'
        });

        // Clear the reminder after showing
        localStorage.removeItem('reminderSet');
        localStorage.removeItem('reminderTime');

        const statusDiv = document.getElementById('reminderStatus');
        if (statusDiv) {
            statusDiv.textContent = '🔔 রিমাইন্ডার দেখানো হয়েছে!';
        }
    }, delay);

    // Also show a banner reminder on page load if a reminder is pending
    checkPendingReminder();
}

// ---- Check if there's a pending reminder on page load ----
function checkPendingReminder() {
    const reminderSet = localStorage.getItem('reminderSet');
    const reminderTime = localStorage.getItem('reminderTime');

    if (reminderSet === 'true' && reminderTime) {
        const target = new Date(reminderTime);
        const now = new Date();

        if (target > now) {
            const delay = target - now;
            const minutes = Math.round(delay / 60000);
            // Show a subtle reminder banner on the page
            const resultDiv = document.getElementById('result');
            if (resultDiv) {
                const banner = document.createElement('div');
                banner.id = 'pendingReminderBanner';
                banner.style.cssText = `
                    margin: 8px 0; padding: 10px; background: #dbeafe; 
                    border-radius: 8px; border-left: 4px solid #2563eb;
                    font-size: 13px; display: flex; justify-content: space-between;
                    align-items: center;
                `;
                banner.innerHTML = `
                    <span>⏰ একটি রিমাইন্ডার ${minutes} মিনিট পরে দেখানো হবে (${target.toLocaleTimeString('bn-BD', { hour: '2-digit', minute: '2-digit' })})</span>
                    <button onclick="cancelReminder()" style="background: #ff0000; color:white; border:none; border-radius:4px; padding:4px 12px; cursor:pointer;">বাতিল</button>
                `;
                resultDiv.prepend(banner);
            }
        } else {
            // Reminder time passed, clean up
            localStorage.removeItem('reminderSet');
            localStorage.removeItem('reminderTime');
        }
    }
}

// ---- Cancel pending reminder ----
function cancelReminder() {
    localStorage.removeItem('reminderSet');
    localStorage.removeItem('reminderTime');
    const banner = document.getElementById('pendingReminderBanner');
    if (banner) banner.remove();
    const statusDiv = document.getElementById('reminderStatus');
    if (statusDiv) statusDiv.textContent = '❌ রিমাইন্ডার বাতিল করা হয়েছে।';
}

// ---- Check for pending reminder on page load ----
document.addEventListener('DOMContentLoaded', checkPendingReminder);

// ---- Request notification permission early (optional, better UX) ----
// This will show the browser's permission prompt when user interacts with the page
document.addEventListener('click', function requestNotificationPermission() {
    if (Notification.permission === 'default') {
        Notification.requestPermission();
    }
}, { once: true });