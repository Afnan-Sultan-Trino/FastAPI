// script.js
//
// Updated to match the full survey/ontology schema used in data_importer.py:
//   - 22 foods (was 12) -- labels copied EXACTLY from BANGLA_TO_ONTOLOGY keys
//   - Frequency labels switched to the FREQ_MAP style ("একদমই না" ... "প্রতিদিন")
//   - NEW: portion-size dropdown per food (PORTION_MAP style)
//   - NEW: symptom checkboxes (11 labels, exactly as SYMPTOM_LABELS)
//   - (life-stage field considered, then dropped -- out of scope for a
//      university-student-only survey; gender-based RDA is sufficient)
//
// ⚠️ main.py's ALLOWED_FOODS / VALID_FREQUENCIES / request schema must be
// updated to match these exact strings, or /predict will reject the payload.
// This file was updated without sight of main.py -- please cross-check.

// Must match ALLOWED_FOODS in main.py (needs updating -- see note above)
const FOOD_NAMES = [
    "ইলিশ মাছ", "রুই মাছ", "গরুর মাংস", "মুরগির মাংস",
    "ডিম", "দুধ", "দই", "দুধ চা",
    "লাল শাক", "পালং শাক",
    "বেগুন", "আলু", "গাজর", "মিষ্টি কুমড়া",
    "আম", "কমলালেবু", "আমলকী", "পেঁপে",
    "ছোলা", "মসুর ডাল", "আটা/রুটি", "ভাত"
];

// Must match VALID_FREQUENCIES in main.py (needs updating -- see note above)
const FREQUENCIES = ["একদমই না", "সপ্তাহে ১–২ বার", "সপ্তাহে ৩–৪ বার", "সপ্তাহে ৫–৬ বার", "প্রতিদিন"];

// NEW: portion-size multiplier options (must match PORTION_MAP in main.py)
const PORTIONS = ["খাই না", "ছোট (স্বাভাবিকের চেয়ে কম)", "স্বাভাবিক পরিমাণ", "বড় (স্বাভাবিকের চেয়ে বেশি)"];

// NEW: the 11 symptom checkboxes -- labels copied EXACTLY from SYMPTOM_LABELS
// in data_importer.py so the backend can map them the same way it maps the
// Google Forms checkbox cell.
const SYMPTOM_LABELS = [
    "সহজেই ক্লান্ত হয়ে পড়া / সবসময় দুর্বল লাগা (Fatigue)",
    "অতিরিক্ত চুল পড়া (Hair Loss)",
    "মাড়ি দিয়ে রক্ত পড়া (Gum Bleeding)",
    "মাড়ি ফোলা/ব্যথা/দাঁতের গোড়া দুর্বল হওয়া (Periodontitis)",
    "রাতে/আবছা আলোতে ঠিকমতো দেখতে সমস্যা হওয়া (Night Blindness)",
    "ত্বক শুষ্ক ও খসখসে হয়ে যাওয়া (Dry/Flaky Skin)",
    "হাড় বা জয়েন্টে ব্যথা (Bone/Joint Pain)",
    "ঘন ঘন ঠান্ডা-জ্বর/অসুস্থ হয়ে পড়া (Frequent Illness)",
    "পড়াশোনায় মনোযোগ ধরে রাখতে কষ্ট হওয়া (Poor Concentration)",
    "চোখের নিচের পাতার ভেতরের অংশ ফ্যাকাশে মনে হওয়া (Pale Conjunctiva)",
    "নখ ভঙ্গুর হয়ে যাওয়া বা চামচের মতো বেঁকে যাওয়া (Koilonychia)",
];

// NOTE: life-stage (pregnant/lactating/child/elderly) was considered but
// removed -- this tool targets university students only, so gender-based
// RDA (already handled by the backend) is sufficient. No separate
// life-stage field needed.

// Build one (frequency + portion) row per food, defaulting to "একদমই না" /
// "স্বাভাবিক পরিমাণ"
function buildFoodFrequencyInputs() {
    const container = document.getElementById('food-frequency-list');
    FOOD_NAMES.forEach(food => {
        const row = document.createElement('div');
        row.className = 'food-row';

        const label = document.createElement('span');
        label.textContent = food;

        const freqSelect = document.createElement('select');
        freqSelect.id = `freq-${food}`;
        FREQUENCIES.forEach(freq => {
            const opt = document.createElement('option');
            opt.value = freq;
            opt.textContent = freq;
            if (freq === "একদমই না") opt.selected = true;
            freqSelect.appendChild(opt);
        });

        const portionSelect = document.createElement('select');
        portionSelect.id = `portion-${food}`;
        PORTIONS.forEach(portion => {
            const opt = document.createElement('option');
            opt.value = portion;
            opt.textContent = portion;
            if (portion === "স্বাভাবিক পরিমাণ") opt.selected = true;
            portionSelect.appendChild(opt);
        });

        row.appendChild(label);
        row.appendChild(freqSelect);
        row.appendChild(portionSelect);
        container.appendChild(row);
    });
}
buildFoodFrequencyInputs();

// NEW: build the 11 symptom checkboxes
function buildSymptomChecklist() {
    const container = document.getElementById('symptom-list');
    if (!container) return;
    SYMPTOM_LABELS.forEach((label, idx) => {
        const row = document.createElement('label');
        row.className = 'symptom-row';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `symptom-${idx}`;
        checkbox.value = label;

        row.appendChild(checkbox);
        row.appendChild(document.createTextNode(' ' + label));
        container.appendChild(row);
    });
}
buildSymptomChecklist();

// (life-stage dropdown removed -- see note above)

// ---- Global variable to store chart instance ----
let deficiencyChart = null;

async function sendData() {
    const name = document.getElementById('name').value.trim() || 'বন্ধু';
    const age = parseInt(document.getElementById('age').value);
    const gender = document.getElementById('gender').value;
    const religion = document.getElementById('religion').value;
    const weight = parseFloat(document.getElementById('weight').value);
    const height = parseFloat(document.getElementById('height').value);

    // NEW: lifestyle fields
    const livingArrangement = document.getElementById('living-arrangement').value;
    const sleepCategory = document.getElementById('sleep-category').value;
    const screenBeforeSleep = document.getElementById('screen-before-sleep').value;
    const sunExposure = document.getElementById('sun-exposure').value;

    // Build {food_name: {frequency, portion}} for every food actually eaten,
    // SKIPPING "একদমই না" entries
    const foods = {};
    FOOD_NAMES.forEach(food => {
        const freq = document.getElementById(`freq-${food}`).value;
        if (freq !== "একদমই না") {
            const portion = document.getElementById(`portion-${food}`).value;
            foods[food] = { frequency: freq, portion: portion };
        }
    });

    // NEW: collect checked symptoms
    const symptoms = SYMPTOM_LABELS.filter((label, idx) => {
        const cb = document.getElementById(`symptom-${idx}`);
        return cb && cb.checked;
    });

    const payload = {
        name, age, gender, religion, weight, height, foods, symptoms,
        living_arrangement: livingArrangement,
        sleep_category: sleepCategory,
        screen_before_sleep_category: screenBeforeSleep,
        sun_exposure_category: sunExposure,
    };

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
    html += `<strong>BMI:</strong> ${data.bmi} — <strong>${data.bmi_category_bangla}</strong> (${data.bmi_category})<br><br>`;

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

    // ---- Recommendations (now includes portion/quantity text) ----
    if (Object.keys(data.recommendations).length > 0) {
        html += `<br><strong>🍽️ প্রস্তাবিত খাবার:</strong><ul>`;
        for (const [nutrient, foodsRec] of Object.entries(data.recommendations)) {
            if (foodsRec.length > 0) {
                const itemsText = foodsRec.map(r => {
                    const costNote = r.budget_friendly === false ? ' 💰 তুলনামূলক ব্যয়বহুল' : '';
                    return `${r.food} (${r.quantity_text})${costNote}`;
                }).join(', ');
                html += `<li>${nutrient}: ${itemsText}</li>`;
            }
        }
        html += `</ul>`;
    }

    // ---- Reported Symptoms (NEW) ----
    if (data.reported_symptoms && data.reported_symptoms.length > 0) {
        html += `<br><strong>🩺 রিপোর্ট করা উপসর্গ:</strong><ul>`;
        data.reported_symptoms.forEach(s => {
            html += `<li>${s}</li>`;
        });
        html += `</ul>`;
    }

    // ---- Lifestyle Tips (NEW) ----
    if (data.lifestyle_tips && data.lifestyle_tips.length > 0) {
        html += `<br><strong>🌤️ জীবনযাত্রা সংক্রান্ত পরামর্শ:</strong><ul>`;
        data.lifestyle_tips.forEach(tip => {
            html += `<li>${tip}</li>`;
        });
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

// ---- Color scale by SEVERITY, not just binary deficient/adequate ----
// Gives a much more informative (and less alarming) chart: a 22% gap and
// a 95% gap both used to show as the same red -- now they don't.
function getColorForDeficiency(pct) {
    if (pct <= 20) return '#22c55e';   // green  - fine
    if (pct <= 40) return '#eab308';   // yellow - mild gap
    if (pct <= 60) return '#f97316';   // orange - moderate gap
    return '#dc2626';                  // red    - severe gap
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
    const colors = values.map(getColorForDeficiency);

    deficiencyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'ঘাটতি (%)',
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',            // horizontal bars -- easier to read nutrient labels
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'ঘাটতি (%)' }
                }
            },
            plugins: {
                legend: { display: false }
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
            // Show a subtle reminder banner on the page (remove any existing
            // one first so repeated calls don't stack up duplicate banners)
            const existingBanner = document.getElementById('pendingReminderBanner');
            if (existingBanner) existingBanner.remove();

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
