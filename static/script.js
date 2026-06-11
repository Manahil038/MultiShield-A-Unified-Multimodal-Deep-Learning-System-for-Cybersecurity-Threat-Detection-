// -------------------------------------------------------------
// MultiShield Client Script - Zero-Trust Intelligence Portal
// -------------------------------------------------------------

// Global state for tabular features (UNSW-NB15 dataset alignments)
let currentTabularPayload = {
    "dur": 15.28, "proto": "tcp", "service": "ssh", "state": "FIN",
    "spkts": 624, "dpkts": 682, "sbytes": 48776, "dbytes": 85554,
    "rate": 85.40, "sttl": 31, "dttl": 29, "sload": 25494.9,
    "dload": 44724.8, "sloss": 144, "dloss": 199, "sinpkt": 24.6,
    "dinpkt": 22.47, "sjit": 1726.6, "djit": 1650.8, "swin": 255,
    "stcpb": 3645792739, "dtcpb": 3646109452, "dwin": 255, "tcprtt": 0.00075,
    "synack": 0.00061, "ackdat": 0.00013, "smean": 78, "dmean": 125,
    "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1,
    "ct_state_ttl": 0, "ct_dst_ltm": 4, "ct_src_dport_ltm": 1,
    "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0,
    "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 3,
    "ct_srv_dst": 1, "is_sm_ips_ports": 0
};

// Preset Scenarios
const PRESETS = {
    tabular: {
        normal: {
            desc: "Safe Connection: A standard, safe web connection with normal data transfer rates.",
            features: {
                "dur": 15.28, "proto": "tcp", "service": "ssh", "state": "FIN",
                "spkts": 624, "dpkts": 682, "sbytes": 48776, "dbytes": 85554,
                "rate": 85.40, "sttl": 31, "dttl": 29, "sload": 25494.9,
                "dload": 44724.8, "sloss": 144, "dloss": 199, "sinpkt": 24.6,
                "dinpkt": 22.47, "sjit": 1726.6, "djit": 1650.8, "swin": 255,
                "stcpb": 3645792739, "dtcpb": 3646109452, "dwin": 255, "tcprtt": 0.00075,
                "synack": 0.00061, "ackdat": 0.00013, "smean": 78, "dmean": 125,
                "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1,
                "ct_state_ttl": 0, "ct_dst_ltm": 4, "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0,
                "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 3,
                "ct_srv_dst": 1, "is_sm_ips_ports": 0
            }
        },
        anomaly: {
            desc: "DDoS Attack Flow: A massive wave of fake data traffic designed to crash your server.",
            features: {
                "dur": 58.06, "proto": "ospf", "service": "-", "state": "REQ",
                "spkts": 52, "dpkts": 0, "sbytes": 5616, "dbytes": 0,
                "rate": 0.87, "sttl": 254, "dttl": 0, "sload": 758.9,
                "dload": 0.0, "sloss": 0, "dloss": 0, "sinpkt": 1170.5,
                "dinpkt": 0.0, "sjit": 1428.3, "djit": 0.0, "swin": 0,
                "stcpb": 0, "dtcpb": 0, "dwin": 0, "tcprtt": 0.0,
                "synack": 0.0, "ackdat": 0.0, "smean": 108, "dmean": 0,
                "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1,
                "ct_state_ttl": 6, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0,
                "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 2,
                "ct_srv_dst": 1, "is_sm_ips_ports": 0
            }
        }
    },
    news: {
        real: {
            title: "Trump administration imposing new email security protocols for agencies",
            text: "WASHINGTON (Reuters) - The Trump administration on Monday will order federal agencies to adopt common email security standards in an effort to better protect against hackers."
        },
        fake: {
            title: "LOL! ANTI-TRUMP ACTORS George Clooney and Matt Damon’s ‘Suburbicon’ Movie TANKS...",
            text: "In fly-over country, and pretty much anywhere in between the east and west coasts of America, people are sick and tired supporting the arrogant, condescending"
        }
    },
    phishing: {
        safe: {
            text: "projects happy chat convenience available 5 00 pm cst today tomorrow time 1 30 cst project list fountain valley psco closing sale black hills late march early april"
        },
        phish: {
            text: "ester pritchett susanneergmckennaseltzerandsodacom papoose"
        }
    }
};

// Diagnostic Verdict Exponents
const EXPLANATIONS = {
    network: {
        normal: "✓ SAFE CONNECTION: This connection has normal data traffic. No signs of server flood or DDoS attacks were found.",
        anomaly: "⚠️ THREAT DETECTED: This looks like a DDoS attack! The network traffic matches patterns used to overwhelm and crash servers."
    },
    news: {
        normal: "✓ TRUSTWORTHY ARTICLE: The writing style, tone, and grammar match verified news publications.",
        anomaly: "⚠️ SUSPICIOUS CONTENT: The AI detected sensational phrasing, false claims, and urgent tones typical of fake news."
    },
    phishing: {
        safe: "✓ SAFE EMAIL: This looks like a normal, safe message. There are no suspicious links or urgent requests for passwords.",
        anomaly: "⚠️ PHISHING SCAM: This email is dangerous! It uses fake urgency and suspicious phrasing to try to steal your passwords or personal details."
    },
    deepfake: {
        normal: "✓ REAL PHOTO: The skin texture and lighting detail match a real camera photo. No AI-generation traits found.",
        anomaly: "⚠️ DEEPFAKE DETECTED: This is an AI-generated face! The scanner detected typical artificial pixel patterns and lighting glitches."
    }
};

// Global variables
let currentUploadedFile = null;
let lenis = null;

// -------------------------------------------------------------
// 1. Initializers & UI Boot Sequence
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    // Registers GSAP ScrollTrigger plugin
    gsap.registerPlugin(ScrollTrigger);

    // Initial log setups
    setupLoggerConsole();

    // Setup active listeners
    setupSliders();
    setupAccordions();
    setupMobileMenu();
    setupHeaderScroll();
    setupDragAndDrop();
    initTooltipBehavior();
    initHowItWorksTimeline();
    initHero3DCanvas();
    setupTerminalTabs();
    setupScrollReveal();
    initDemoTerminal();
    initDemoCounters();

    // Check backend health
    checkServerHealth();
    setInterval(checkServerHealth, 10000);

    // Start with tabular normal default preset
    loadTabularPreset('normal');
});

// -------------------------------------------------------------
// 2. Lucien GSAP Preloader Sequence
// -------------------------------------------------------------
function initPreloaderLoader() {
    const tl = gsap.timeline();

    gsap.defaults({
        ease: "Expo.easeInOut",
        duration: 1.6,
    });

    const rand1 = gsap.utils.random([2, 3, 4]);
    const rand2 = gsap.utils.random([5, 6]);
    const rand3 = gsap.utils.random([1, 4]);
    const rand4 = gsap.utils.random([7, 8, 9]);

    // Initial preloading states
    tl.set(".home-preloader", { display: "block", opacity: 1 });
    tl.set(".loading-screen", { display: "flex", opacity: 1 });
    tl.set(".loading-screen > *", { opacity: 0, y: "1.5rem" });
    tl.set(".loading__progress-inner", { scaleX: 0 });
    tl.set(".loading__number-group.is--first .loading__number-wrap, .loading__percentage", { yPercent: 100 });
    tl.set(".loading__number-group.is--second .loading__number-wrap, .loading__number-group.is--third .loading__number-wrap", { yPercent: 10 });

    // Fade Elements In
    tl.to(".loading-screen > *", {
        opacity: 1,
        y: 0,
        duration: 0.6,
        ease: "power3.out",
        stagger: 0.1
    });

    // Animate stage 1 progress
    tl.to(".loading__progress-inner", {
        scaleX: (rand1 + "" + rand3) / 100
    });

    tl.to(".loading__percentage", {
        yPercent: 0
    }, "<");

    tl.to(".loading__number-group.is--second .loading__number-wrap", {
        yPercent: (rand1 - 1) * -10
    }, "<");

    tl.to(".loading__number-group.is--third .loading__number-wrap", {
        yPercent: (rand3 - 1) * -10
    }, "<");

    // Animate stage 2 progress
    tl.to(".loading__progress-inner", {
        scaleX: (rand2 + "" + rand4) / 100
    });

    tl.to(".loading__number-group.is--second .loading__number-wrap", {
        yPercent: (rand2 - 1) * -10
    }, "<");

    tl.to(".loading__number-group.is--third .loading__number-wrap", {
        yPercent: (rand4 - 1) * -10
    }, "<");

    // Animate stage 3 completion (100%)
    tl.to(".loading__progress-inner", {
        scaleX: 1
    });

    tl.to(".loading__number-group.is--second .loading__number-wrap", {
        yPercent: -90 // Scrolls to 0
    }, "<");

    tl.to(".loading__number-group.is--third .loading__number-wrap", {
        yPercent: -90 // Scrolls to 0
    }, "<");

    tl.to(".loading__number-group.is--first .loading__number-wrap", {
        yPercent: 0 // Shows 1
    }, "<");

    // Fade loader out
    tl.to(".loading-screen > *", {
        opacity: 0,
        y: "-1.5rem",
        duration: 0.6,
        ease: "power3.out",
        stagger: 0.08
    });

    tl.to(".home-preloader", {
        opacity: 0,
        duration: 0.4
    });

    tl.set(".home-preloader", { display: "none" });

    // Initialize typography scramble triggers on completion
    tl.add(() => {
        initScrambleReveal();
        initLenisScroll();
        appendConsoleLog("system", "MultiShield Framework: Core diagnostic runtime active.");
    });
}

// Trigger preloader load sequence on window trigger
window.addEventListener("load", () => {
    initPreloaderLoader();
});

// -------------------------------------------------------------
// 3. Typographic Scramble Character Morph Effect
// -------------------------------------------------------------
function scrambleTextMorph(element, finalString, duration = 1.6) {
    const characters = "XYZ//0123456789%@#!*?+=-_";
    const textLength = finalString.length;
    const tracker = { progress: 0 };
    
    gsap.to(tracker, {
        progress: textLength,
        duration: duration,
        ease: "power2.out",
        onUpdate: () => {
            const currentPosition = Math.floor(tracker.progress);
            let displayString = "";
            for (let i = 0; i < textLength; i++) {
                if (i < currentPosition) {
                    displayString += finalString[i];
                } else if (finalString[i] === " ") {
                    displayString += " ";
                } else {
                    displayString += characters[Math.floor(Math.random() * characters.length)];
                }
            }
            element.innerText = displayString;
        }
    });
}

function initScrambleReveal() {
    const heroHeading = document.querySelector('[scramble-words-hero] .heading_text');
    if (heroHeading) {
        const originalText = heroHeading.innerText;
        scrambleTextMorph(heroHeading, originalText, 1.8);
    }
}

// -------------------------------------------------------------
// 4. Lenis Smooth Scroll & Horizontal Scroll Card Track
// -------------------------------------------------------------
function initLenisScroll() {
    lenis = new Lenis({
        duration: 1.2,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        orientation: "vertical",
        gestureOrientation: "vertical",
        smoothWheel: true
    });

    lenis.on('scroll', ScrollTrigger.update);

    gsap.ticker.add((time) => {
        lenis.raf(time * 1000);
    });
    gsap.ticker.lagSmoothing(0);

    // Setup Side-Scrolling Card animations
    const stickySection = document.querySelector(".horizontal-cards_sticky");
    const trackElement = document.querySelector(".horizontal-cards_track");
    const listElement = document.querySelector(".horizontal-cards_list");

    if (stickySection && trackElement && listElement) {
        const calculateScrollOffset = () => {
            return Math.max(0, listElement.scrollWidth - window.innerWidth);
        };

        // Slide list element horizontally as user scrolls sticky space vertically (CSS sticky pinning)
        gsap.to(listElement, {
            x: () => -calculateScrollOffset(),
            ease: "none",
            scrollTrigger: {
                trigger: ".horizontal-cards_height",
                start: "top top",
                end: "bottom bottom",
                scrub: true,
                invalidateOnRefresh: true
            }
        });

        // Update main top scroll bar
        gsap.to(".scroll-progress_bar", {
            width: "100%",
            ease: "none",
            scrollTrigger: {
                trigger: ".horizontal-cards_height",
                start: "top top",
                end: "bottom bottom",
                scrub: true
            }
        });
    }

    // Refresh ScrollTrigger calculations
    ScrollTrigger.refresh();
}

// -------------------------------------------------------------
// 5. Active Header Scrolled styles
// -------------------------------------------------------------
function setupHeaderScroll() {
    const navHeader = document.querySelector(".nav_wrapper");
    if (!navHeader) return;

    window.addEventListener("scroll", () => {
        if (window.scrollY > 50) {
            navHeader.querySelector(".nav_bg").classList.add("is-scrolled");
        } else {
            navHeader.querySelector(".nav_bg").classList.remove("is-scrolled");
        }
    });
}

// -------------------------------------------------------------
// 6. Mobile Hambuger Overlay menu controls
// -------------------------------------------------------------
function setupMobileMenu() {
    const toggleButton = document.querySelector('[data-nav-button="toggle"]');
    const closeOverlay = document.querySelector(".nav_overlay");
    const navStatusEl = document.querySelector("[data-nav-wrapper]");
    const navLinks = document.querySelectorAll("[data-nav-link]");

    if (!toggleButton || !navStatusEl) return;

    const toggleMenu = () => {
        const isOpen = navStatusEl.getAttribute("data-nav-status") === "open";
        navStatusEl.setAttribute("data-nav-status", isOpen ? "closed" : "open");
        if (lenis) {
            isOpen ? lenis.start() : lenis.stop();
        }
    };

    const closeMenu = () => {
        navStatusEl.setAttribute("data-nav-status", "closed");
        if (lenis) lenis.start();
    };

    toggleButton.addEventListener("click", toggleMenu);
    if (closeOverlay) closeOverlay.addEventListener("click", closeMenu);
    navLinks.forEach(link => link.addEventListener("click", closeMenu));
}

// -------------------------------------------------------------
// 7. Accordions Pane collapsible toggle actions
// -------------------------------------------------------------
function setupAccordions() {
    const accordions = document.querySelectorAll('[data-accordion="component"]');
    
    accordions.forEach(acc => {
        const toggleBtn = acc.querySelector('[data-accordion="toggle"]');
        if (!toggleBtn) return;

        toggleBtn.addEventListener("click", () => {
            const isOpen = acc.getAttribute("data-accordion-state") === "open";
            
            // Close other sibling accordions
            const siblings = acc.parentElement.querySelectorAll('[data-accordion="component"]');
            siblings.forEach(sib => {
                sib.setAttribute("data-accordion-state", "close");
            });

            acc.setAttribute("data-accordion-state", isOpen ? "close" : "open");
            
            // Refresh scrolltrigger alignments
            setTimeout(() => ScrollTrigger.refresh(), 500);
        });
    });
}

// -------------------------------------------------------------
// 8. Sliders bindings & range conversions
// -------------------------------------------------------------
function toggleTabularDrawer() {
    const drawer = document.querySelector(".accordion-drawer");
    if (!drawer) return;
    const currentState = drawer.getAttribute("data-accordion-state");
    const isClosed = currentState === "close";
    drawer.setAttribute("data-accordion-state", isClosed ? "open" : "close");
    
    appendConsoleLog("system", `Parameter Control Panel: Drawer ${isClosed ? 'expanded' : 'collapsed'}.`);
}

function setupSliders() {
    const sliderDefs = [
        { id: "tab-dur", key: "dur", displayId: "val-tag-dur", format: (v) => `${parseFloat(v).toFixed(4)}s` },
        { id: "tab-rate", key: "rate", displayId: "val-tag-rate", format: (v) => `${parseFloat(v).toLocaleString(undefined, { maximumFractionDigits: 0 })} pkts/s` },
        { id: "tab-sbytes", key: "sbytes", displayId: "val-tag-sbytes", format: (v) => formatBytesVal(v) },
        { id: "tab-dbytes", key: "dbytes", displayId: "val-tag-dbytes", format: (v) => formatBytesVal(v) },
        { id: "tab-sload", key: "sload", displayId: "val-tag-sload", format: (v) => formatBitrateVal(v) },
        { id: "tab-dload", key: "dload", displayId: "val-tag-dload", format: (v) => formatBitrateVal(v) }
    ];

    sliderDefs.forEach(def => {
        const sliderElement = document.getElementById(def.id);
        const outputSpan = document.getElementById(def.displayId);

        if (sliderElement && outputSpan) {
            sliderElement.addEventListener("input", (e) => {
                const val = e.target.value;
                outputSpan.innerText = def.format(val);
                currentTabularPayload[def.key] = parseFloat(val);
            });
        }
    });

    // Dropdown updates
    const protoSelect = document.getElementById("tab-proto");
    const serviceSelect = document.getElementById("tab-service");
    const stateSelect = document.getElementById("tab-state");

    if (protoSelect) protoSelect.addEventListener("change", (e) => currentTabularPayload.proto = e.target.value);
    if (serviceSelect) serviceSelect.addEventListener("change", (e) => currentTabularPayload.service = e.target.value);
    if (stateSelect) stateSelect.addEventListener("change", (e) => currentTabularPayload.state = e.target.value);
}

function formatBytesVal(bytes) {
    bytes = parseInt(bytes);
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1024).toFixed(1)} KB`;
}

function formatBitrateVal(bps) {
    bps = parseFloat(bps);
    if (bps < 1000) return `${bps.toFixed(0)} bps`;
    if (bps < 1000000) return `${(bps / 1000).toFixed(1)} Kbps`;
    return `${(bps / 1000000).toFixed(1)} Mbps`;
}

// -------------------------------------------------------------
// 9. Form Staged Scenarios Preset Loaders
// -------------------------------------------------------------
function loadTabularPreset(type) {
    const data = PRESETS.tabular[type];
    if (!data) return;

    currentTabularPayload = { ...data.features };

    // Updates text info
    const infoCard = document.getElementById("tab-preset-info");
    if (infoCard) infoCard.innerText = data.desc;

    // Load form field values
    document.getElementById("tab-dur").value = currentTabularPayload.dur;
    document.getElementById("tab-rate").value = currentTabularPayload.rate;
    document.getElementById("tab-sbytes").value = currentTabularPayload.sbytes;
    document.getElementById("tab-dbytes").value = currentTabularPayload.dbytes;
    document.getElementById("tab-sttl").value = currentTabularPayload.sttl;
    document.getElementById("tab-dttl").value = currentTabularPayload.dttl;
    document.getElementById("tab-sload").value = currentTabularPayload.sload;
    document.getElementById("tab-dload").value = currentTabularPayload.dload;

    document.getElementById("tab-proto").value = currentTabularPayload.proto;
    document.getElementById("tab-service").value = currentTabularPayload.service;
    document.getElementById("tab-state").value = currentTabularPayload.state;

    // Set Slider Text Output elements
    document.getElementById("val-tag-dur").innerText = `${currentTabularPayload.dur.toFixed(4)}s`;
    document.getElementById("val-tag-rate").innerText = `${currentTabularPayload.rate.toLocaleString()} pkts/s`;
    document.getElementById("val-tag-sbytes").innerText = formatBytesVal(currentTabularPayload.sbytes);
    document.getElementById("val-tag-dbytes").innerText = formatBytesVal(currentTabularPayload.dbytes);
    document.getElementById("val-tag-sload").innerText = formatBitrateVal(currentTabularPayload.sload);
    document.getElementById("val-tag-dload").innerText = formatBitrateVal(currentTabularPayload.dload);

    // Sync button active classes
    const flowCard = document.getElementById("flow-classifier");
    if (flowCard) {
        flowCard.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
        const activeBtn = type === "normal" ? flowCard.querySelectorAll(".btn-preset")[0] : flowCard.querySelectorAll(".btn-preset")[1];
        if (activeBtn) activeBtn.classList.add("active");
    }

    appendConsoleLog("info", `flow_classifier.py: Loaded network preset parameters: ${type.toUpperCase()}`);
}

function loadNewsPreset(type) {
    const data = PRESETS.news[type];
    if (!data) return;

    document.getElementById("news-title").value = data.title;
    document.getElementById("news-text").value = data.text;

    const newsCard = document.getElementById("news-verifier");
    if (newsCard) {
        newsCard.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
        const activeBtn = type === "real" ? newsCard.querySelectorAll(".btn-preset")[0] : newsCard.querySelectorAll(".btn-preset")[1];
        if (activeBtn) activeBtn.classList.add("active");
    }

    appendConsoleLog("info", `news_verifier.py: Staged fake news narrative preset: ${type.toUpperCase()}`);
}

function loadPhishingPreset(type) {
    const data = PRESETS.phishing[type];
    if (!data) return;

    document.getElementById("phishing-text").value = data.text;

    const emailCard = document.getElementById("email-auditor");
    if (emailCard) {
        emailCard.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
        const activeBtn = type === "safe" ? emailCard.querySelectorAll(".btn-preset")[0] : emailCard.querySelectorAll(".btn-preset")[1];
        if (activeBtn) activeBtn.classList.add("active");
    }

    appendConsoleLog("info", `email_auditor.py: Staged text phishing preset: ${type.toUpperCase()}`);
}

// -------------------------------------------------------------
// 10. Circular Gauge Progress Renders
// -------------------------------------------------------------
function animateRingGauge(circleId, textId, rawProbability, isThreat) {
    const circle = document.getElementById(circleId);
    const textOutput = document.getElementById(textId);
    if (!circle || !textOutput) return;

    // Circumference = 263.8
    const circ = 263.8;
    circle.style.strokeDasharray = `${circ} ${circ}`;

    if (isThreat) {
        circle.style.stroke = "var(--color-threat)";
    } else {
        circle.style.stroke = "var(--color-safe)";
    }

    const offset = circ - (rawProbability * circ);
    circle.style.strokeDashoffset = offset;

    // Numerical readout percent animation
    textOutput.innerText = `${(rawProbability * 100).toFixed(0)}%`;
}

// -------------------------------------------------------------
// 11. Backend API integrations
// -------------------------------------------------------------
async function checkServerHealth() {
    const devicePill = document.getElementById("device-pill");
    const pingLatency = document.getElementById("ping-latency");
    const greenIndicator = document.getElementById("health-indicator-dot");

    const startTime = performance.now();
    try {
        const response = await fetch("/health");
        const elapsed = (performance.now() - startTime).toFixed(0);

        if (response.ok) {
            const data = await response.json();
            if (devicePill) devicePill.innerText = data.device.toUpperCase();
            if (pingLatency) pingLatency.innerText = `${elapsed}ms`;
            if (greenIndicator) {
                greenIndicator.className = "status-indicator-dot dot-green";
            }
        } else {
            throw new Error("HTTP state critical");
        }
    } catch (err) {
        if (devicePill) devicePill.innerText = "OFFLINE";
        if (pingLatency) pingLatency.innerText = "--";
        if (greenIndicator) {
            greenIndicator.className = "status-indicator-dot dot-red";
        }
    }
}

function updateLoadingState(buttonId, originalText, isLoading) {
    const button = document.getElementById(buttonId);
    if (!button) return;

    if (isLoading) {
        button.disabled = true;
        button.innerHTML = `<span class="spinner"></span>Evaluating...`;
    } else {
        button.disabled = false;
        button.innerText = originalText;
    }
}

async function handleTabularSubmit(event) {
    event.preventDefault();

    const placeholder = document.getElementById("placeholder-network");
    const reportBox = document.getElementById("result-network");
    const errorBox = document.getElementById("err-network");

    placeholder.classList.add("hidden");
    reportBox.classList.add("hidden");
    errorBox.classList.add("hidden");

    updateLoadingState("btn-tab-submit", "EXECUTE NEURAL FLOW SCAN", true);
    appendConsoleLog("system", "[POST] Submitting flow telemetry vector to /api/predict/tabular...");

    try {
        const response = await fetch("/api/predict/tabular", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: currentTabularPayload })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Prediction request failure.");

        const isAnomaly = data.prediction === 1;
        const confidenceVal = (data.confidence * 100).toFixed(1);

        animateRingGauge("circle-network", "val-network", data.raw_probability, isAnomaly);

        const statusBadge = document.getElementById("badge-network");
        statusBadge.innerText = `${data.label} (${confidenceVal}% CONFIDENCE)`;
        statusBadge.className = isAnomaly ? "status-badge anomaly-result" : "status-badge normal-result";

        const explanation = document.getElementById("explanation-network");
        explanation.innerText = isAnomaly ? EXPLANATIONS.network.anomaly : EXPLANATIONS.network.normal;
        explanation.className = isAnomaly ? "result-details anomaly-expl" : "result-details normal-expl";

        reportBox.classList.remove("hidden");
        appendConsoleLog(isAnomaly ? "warn" : "ok", `[RESULT] flow_classifier.py: ${data.label.toUpperCase()} (Confidence: ${confidenceVal}%)`);
    } catch (err) {
        errorBox.innerText = `Scan Failed: ${err.message}`;
        errorBox.classList.remove("hidden");
        placeholder.classList.remove("hidden");
        appendConsoleLog("warn", `[ERROR] flow_classifier: Query connection rejected: ${err.message}`);
    } finally {
        updateLoadingState("btn-tab-submit", "EXECUTE NEURAL FLOW SCAN", false);
    }
}

async function handleNewsSubmit(event) {
    event.preventDefault();

    const placeholder = document.getElementById("placeholder-news");
    const reportBox = document.getElementById("result-news");
    const errorBox = document.getElementById("err-news");

    placeholder.classList.add("hidden");
    reportBox.classList.add("hidden");
    errorBox.classList.add("hidden");

    const headline = document.getElementById("news-title").value;
    const bodyText = document.getElementById("news-text").value;

    updateLoadingState("btn-news-submit", "AUDIT NARRATIVE INTEGRITY", true);
    appendConsoleLog("system", "[POST] Submitting headline text patterns to /api/predict/news...");

    try {
        const response = await fetch("/api/predict/news", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: headline, text: bodyText })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Narrative classification error.");

        const isFake = data.prediction === 1;
        const confidenceVal = (data.confidence * 100).toFixed(1);

        animateRingGauge("circle-news", "val-news", data.raw_probability, isFake);

        const statusBadge = document.getElementById("badge-news");
        statusBadge.innerText = `${data.label} (${confidenceVal}% CONFIDENCE)`;
        statusBadge.className = isFake ? "status-badge anomaly-result" : "status-badge normal-result";

        const explanation = document.getElementById("explanation-news");
        explanation.innerText = isFake ? EXPLANATIONS.news.anomaly : EXPLANATIONS.news.normal;
        explanation.className = isFake ? "result-details anomaly-expl" : "result-details normal-expl";

        reportBox.classList.remove("hidden");
        appendConsoleLog(isFake ? "warn" : "ok", `[RESULT] news_verifier.py: ${data.label.toUpperCase()} (Confidence: ${confidenceVal}%)`);
    } catch (err) {
        errorBox.innerText = `Audit Failed: ${err.message}`;
        errorBox.classList.remove("hidden");
        placeholder.classList.remove("hidden");
        appendConsoleLog("warn", `[ERROR] news_verifier: Classification endpoint error: ${err.message}`);
    } finally {
        updateLoadingState("btn-news-submit", "AUDIT NARRATIVE INTEGRITY", false);
    }
}

async function handlePhishingSubmit(event) {
    event.preventDefault();

    const placeholder = document.getElementById("placeholder-phishing");
    const reportBox = document.getElementById("result-phishing");
    const errorBox = document.getElementById("err-phishing");

    placeholder.classList.add("hidden");
    reportBox.classList.add("hidden");
    errorBox.classList.add("hidden");

    const textInput = document.getElementById("phishing-text").value;

    updateLoadingState("btn-phishing-submit", "AUDIT EMAIL SCENARIO", true);
    appendConsoleLog("system", "[POST] Submitting email body sequence to /api/predict/phishing...");

    try {
        const response = await fetch("/api/predict/phishing", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: textInput })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Email parser prediction error.");

        const isPhish = data.prediction === 1;
        const confidenceVal = (data.confidence * 100).toFixed(1);

        animateRingGauge("circle-phishing", "val-phishing", data.raw_probability, isPhish);

        const statusBadge = document.getElementById("badge-phishing");
        statusBadge.innerText = `${data.label} (${confidenceVal}% CONFIDENCE)`;
        statusBadge.className = isPhish ? "status-badge anomaly-result" : "status-badge normal-result";

        const explanation = document.getElementById("explanation-phishing");
        explanation.innerText = isPhish ? EXPLANATIONS.phishing.anomaly : EXPLANATIONS.phishing.normal;
        explanation.className = isPhish ? "result-details anomaly-expl" : "result-details normal-expl";

        reportBox.classList.remove("hidden");
        appendConsoleLog(isPhish ? "warn" : "ok", `[RESULT] email_auditor.py: ${data.label.toUpperCase()} (Confidence: ${confidenceVal}%)`);
    } catch (err) {
        errorBox.innerText = `Audit Failed: ${err.message}`;
        errorBox.classList.remove("hidden");
        placeholder.classList.remove("hidden");
        appendConsoleLog("warn", `[ERROR] email_auditor: Auditor lookup failed: ${err.message}`);
    } finally {
        updateLoadingState("btn-phishing-submit", "AUDIT EMAIL SCENARIO", false);
    }
}

// Drag & Drop Media Scanner handlers
function triggerFileInput() {
    const fileElement = document.getElementById("deepfake-file");
    if (fileElement) fileElement.click();
}

function handleFileSelect(event) {
    const filesList = event.target.files;
    if (filesList.length > 0) {
        stagePortraitFile(filesList[0]);
    }
}

function setupDragAndDrop() {
    const zone = document.getElementById("drop-zone");
    if (!zone) return;

    ["dragenter", "dragover"].forEach(name => {
        zone.addEventListener(name, (e) => {
            e.preventDefault();
            zone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(name => {
        zone.addEventListener(name, (e) => {
            e.preventDefault();
            zone.classList.remove("dragover");
        });
    });

    zone.addEventListener("drop", (e) => {
        const droppedFiles = e.dataTransfer.files;
        if (droppedFiles.length > 0) {
            stagePortraitFile(droppedFiles[0]);
        }
    });
}

function stagePortraitFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["png", "jpg", "jpeg"].includes(ext)) {
        appendConsoleLog("warn", `[MEDIA_SCAN] Stage rejected. Invalid format: ${file.name}`);
        alert("File format rejected. Only JPG, JPEG, and PNG images are supported.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        appendConsoleLog("warn", `[MEDIA_SCAN] Stage rejected. Size exceeds 10MB limit: ${file.name}`);
        alert("File size rejected. Maximum file limit is 10MB.");
        return;
    }

    currentUploadedFile = file;

    const fileReader = new FileReader();
    fileReader.onload = (e) => {
        const overlayUI = document.getElementById("upload-elements-ui");
        const previewWrap = document.getElementById("preview-box");
        const previewImage = document.getElementById("preview-img");
        const submitBtn = document.getElementById("btn-deepfake-submit");

        if (overlayUI && previewWrap && previewImage) {
            previewImage.src = e.target.result;
            overlayUI.classList.add("hidden");
            previewWrap.classList.remove("hidden");
            if (submitBtn) submitBtn.disabled = false;

            appendConsoleLog("ok", `media_scanner.py: Staged face image: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
        }
    };
    fileReader.readAsDataURL(file);
}

async function loadDeepfakeSample(type) {
    const presetPath = `/samples/${type}.jpg`;
    currentUploadedFile = null;

    const overlayUI = document.getElementById("upload-elements-ui");
    const previewWrap = document.getElementById("preview-box");
    const previewImage = document.getElementById("preview-img");
    const submitBtn = document.getElementById("btn-deepfake-submit");

    updateLoadingState("btn-deepfake-submit", "EXECUTE SPATIAL MEDIA SCAN", true);
    appendConsoleLog("info", `media_scanner.py: Loading preset media from static path: ${presetPath}...`);

    try {
        const response = await fetch(presetPath);
        if (!response.ok) throw new Error("Preset image load failure.");

        const blob = await response.blob();
        const fileObj = new File([blob], `${type}.jpg`, { type: "image/jpeg" });
        currentUploadedFile = fileObj;

        if (overlayUI && previewWrap && previewImage) {
            previewImage.src = presetPath;
            overlayUI.classList.add("hidden");
            previewWrap.classList.remove("hidden");
            if (submitBtn) submitBtn.disabled = false;
        }

        const mediaCard = document.getElementById("media-scanner");
        if (mediaCard) {
            mediaCard.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
            const activeBtn = type === "real" ? mediaCard.querySelectorAll(".btn-preset")[0] : mediaCard.querySelectorAll(".btn-preset")[1];
            if (activeBtn) activeBtn.classList.add("active");
        }

        appendConsoleLog("ok", `media_scanner.py: Staged spatial face preset: ${type.toUpperCase()}`);
    } catch (err) {
        appendConsoleLog("warn", `[ERROR] media_scanner: Preset face load failure: ${err.message}`);
        alert("Failed to fetch preset face. Make sure FastAPI is online.");
    } finally {
        updateLoadingState("btn-deepfake-submit", "EXECUTE SPATIAL MEDIA SCAN", false);
    }
}

async function handleDeepfakeSubmit(event) {
    event.preventDefault();

    const placeholder = document.getElementById("placeholder-deepfake");
    const reportBox = document.getElementById("result-deepfake");
    const errorBox = document.getElementById("err-deepfake");

    placeholder.classList.add("hidden");
    reportBox.classList.add("hidden");
    errorBox.classList.add("hidden");

    if (!currentUploadedFile) {
        errorBox.innerText = "Execution failed: No image file selected.";
        errorBox.classList.remove("hidden");
        placeholder.classList.remove("hidden");
        return;
    }

    updateLoadingState("btn-deepfake-submit", "EXECUTE SPATIAL MEDIA SCAN", true);
    appendConsoleLog("system", "[POST] Submitting image binary to /api/predict/deepfake (ResNet TTA layers active)...");

    try {
        const dataBody = new FormData();
        dataBody.append("file", currentUploadedFile);

        const response = await fetch("/api/predict/deepfake", {
            method: "POST",
            body: dataBody
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Image classification logic failure.");

        const isFake = data.prediction === 1;
        const confidenceVal = (data.confidence * 100).toFixed(1);

        animateRingGauge("circle-deepfake", "val-deepfake", data.raw_probability, isFake);

        const statusBadge = document.getElementById("badge-deepfake");
        statusBadge.innerText = `${data.label} (${confidenceVal}% CONFIDENCE)`;
        statusBadge.className = isFake ? "status-badge anomaly-result" : "status-badge normal-result";

        const explanation = document.getElementById("explanation-deepfake");
        explanation.innerText = isFake ? EXPLANATIONS.deepfake.anomaly : EXPLANATIONS.deepfake.normal;
        explanation.className = isFake ? "result-details anomaly-expl" : "result-details normal-expl";

        reportBox.classList.remove("hidden");
        appendConsoleLog(isFake ? "warn" : "ok", `[RESULT] media_scanner.py: ${data.label.toUpperCase()} (Confidence: ${confidenceVal}%)`);
    } catch (err) {
        errorBox.innerText = `Scan Failed: ${err.message}`;
        errorBox.classList.remove("hidden");
        placeholder.classList.remove("hidden");
        appendConsoleLog("warn", `[ERROR] media_scanner: Spatial matrix analysis rejected: ${err.message}`);
    } finally {
        updateLoadingState("btn-deepfake-submit", "EXECUTE SPATIAL MEDIA SCAN", false);
    }
}

// -------------------------------------------------------------
// 12. Monospace Logger Feed Feed Console Handlers
// -------------------------------------------------------------
const devLogsMock = [
    { type: "info", text: "Staging data preprocessing configurations: numeric_scaler=RobustScaler, categoricals=OneHotEncoder." },
    { type: "system", text: "Allocating PyTorch weights: tabular_deep (256 dense units, GELU layers)." },
    { type: "ok", text: "Bi-LSTM news classification network mapped to CUDA:0 device context successfully." },
    { type: "info", text: "Loaded stopword normalization mappings from NLTK packages corpus." },
    { type: "ok", text: "Successfully deployed ResNet-18 facial verification weights (Test-Time Augmentation active)." },
    { type: "system", text: "FastAPI server running and monitoring port 8088." },
    { type: "info", text: "Heartbeat check: health logs polling ok. Memory status: 1.4GB GPU, 320MB RAM." },
    { type: "ok", text: "Packet telemetry scaler align checks: output feature vector dimension = 169." }
];

function setupLoggerConsole() {
    const box = document.getElementById("scrolling-terminal-logs");
    if (!box) return;

    // Load initial logs
    devLogsMock.forEach(log => appendConsoleLog(log.type, log.text));

    // Dynamic log generator feed
    setInterval(() => {
        const item = devLogsMock[Math.floor(Math.random() * devLogsMock.length)];
        const time = new Date().toLocaleTimeString();
        appendConsoleLog(item.type, `[${time}] ${item.text}`);
    }, 7000);
}

function appendConsoleLog(type, text) {
    const box = document.getElementById("scrolling-terminal-logs");
    if (!box) return;

    const line = document.createElement("div");
    line.className = `log-line ${type}`;
    line.innerText = text;
    box.appendChild(line);

    // Keep console output at bottom
    box.scrollTop = box.scrollHeight;
}

// -------------------------------------------------------------
// 13. Telemetry Table Tooltip Behavior (Lucien style)
// -------------------------------------------------------------
function initTooltipBehavior() {
    const desktopBreakpoint = 992;
    const tooltipSelector = '.tooltip';

    // Clean old listeners by cloning elements
    document.querySelectorAll(tooltipSelector).forEach((item) => {
        const clone = item.cloneNode(true);
        item.parentNode.replaceChild(clone, item);
    });

    const tooltips = document.querySelectorAll(tooltipSelector);

    // Desktop click toggling behavior
    tooltips.forEach((item) => {
        item.addEventListener('click', (e) => {
            if (window.innerWidth < desktopBreakpoint) return;
            e.stopPropagation();

            const parent = item.parentElement;
            const current = item.getAttribute('data-tooltip-status');
            const newState = current === 'open' ? 'closed' : 'open';

            // Close siblings first
            if (parent) {
                parent.querySelectorAll(tooltipSelector).forEach((sibling) => {
                    sibling.setAttribute('data-tooltip-status', 'closed');
                });
            }

            item.setAttribute('data-tooltip-status', newState);
        });
    });

    // Close on clicking anywhere else on document
    document.addEventListener('click', () => {
        document.querySelectorAll(tooltipSelector).forEach((tooltip) => {
            tooltip.setAttribute('data-tooltip-status', 'closed');
        });
    });
}

// -------------------------------------------------------------
// 14. How It Works Timeline Scroll Steps Tracker (GSAP ScrollTrigger)
// -------------------------------------------------------------
function initHowItWorksTimeline() {
    const steps = document.querySelectorAll(".section-how_text-wrapper");
    const panels = document.querySelectorAll(".visualizer-panel");

    steps.forEach((step) => {
        const stepNum = step.getAttribute("data-step");

        ScrollTrigger.create({
            trigger: step,
            start: "top 60%",
            end: "bottom 40%",
            onEnter: () => {
                step.classList.add("active");
                panels.forEach(p => p.classList.remove("active"));
                const targetPanel = document.getElementById(`visual-panel-${stepNum}`);
                if (targetPanel) targetPanel.classList.add("active");
            },
            onLeave: () => {
                step.classList.remove("active");
            },
            onEnterBack: () => {
                step.classList.add("active");
                panels.forEach(p => p.classList.remove("active"));
                const targetPanel = document.getElementById(`visual-panel-${stepNum}`);
                if (targetPanel) targetPanel.classList.add("active");
            },
            onLeaveBack: () => {
                step.classList.remove("active");
            }
        });
    });

    // Dynamic text decrypter text cycling for Panel 2
    const decryptEl = document.getElementById("matrix-decrypt-text");
    if (decryptEl) {
        const phrases = [
            "DECODING TEXT...",
            "ANALYZING SEMANTICS...",
            "VERIFYING WRITER...",
            "STYLE PROFILE: GENUINE",
            "CLASSIFIER VERDICT: SAFE // 98%"
        ];
        let pIdx = 0;
        setInterval(() => {
            pIdx = (pIdx + 1) % phrases.length;
            decryptEl.innerText = phrases[pIdx];
        }, 2500);
    }
}

// -------------------------------------------------------------
// 15. Hero 3D Neural Constellation WebGL Canvas (Three.js)
// -------------------------------------------------------------
function initHero3DCanvas() {
    const canvas = document.getElementById("hero-3d-canvas");
    if (!canvas || typeof THREE === "undefined") return;

    const container = canvas.parentElement;

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const camera = new THREE.PerspectiveCamera(65, 1, 0.1, 2000);
    camera.position.z = 380;

    const scene = new THREE.Scene();
    const masterGroup = new THREE.Group();
    scene.add(masterGroup);

    function resize() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }
    resize();
    window.addEventListener("resize", resize);

    // ---- LAYER 1: Background star field ----
    const starCount = 500;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
        starPositions[i * 3]     = (Math.random() - 0.5) * 1400;
        starPositions[i * 3 + 1] = (Math.random() - 0.5) * 900;
        starPositions[i * 3 + 2] = (Math.random() - 0.5) * 600 - 300;
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
    scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({
        color: 0x8899cc, size: 1.2, transparent: true, opacity: 0.45,
        blending: THREE.AdditiveBlending, depthWrite: false
    })));

    // ---- LAYER 2: Multi-color neural network nodes (expanded spherical distribution) ----
    const nodeCount = 200;
    const nodePositions = new Float32Array(nodeCount * 3);
    const nodeColors    = new Float32Array(nodeCount * 3);
    const nodeVelocities = [];

    const colorPalette = [
        new THREE.Color(0x4f7fff),
        new THREE.Color(0x7c3aed),
        new THREE.Color(0x06b6d4),
        new THREE.Color(0x3b82f6),
        new THREE.Color(0xa78bfa)
    ];

    for (let i = 0; i < nodeCount; i++) {
        // Expanded spherical distribution to cover the full screen
        const u     = Math.random();
        const v     = Math.random();
        const theta = 2 * Math.PI * u;
        const phi   = Math.acos(2 * v - 1);
        const r     = 120 + Math.random() * 480; // Expanded radius [120, 600]

        nodePositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        nodePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        nodePositions[i * 3 + 2] = r * Math.cos(phi);

        nodeVelocities.push(new THREE.Vector3(
            (Math.random() - 0.5) * 0.08,
            (Math.random() - 0.5) * 0.08,
            (Math.random() - 0.5) * 0.065
        ));

        const c = colorPalette[Math.floor(Math.random() * colorPalette.length)];
        nodeColors[i * 3]     = c.r;
        nodeColors[i * 3 + 1] = c.g;
        nodeColors[i * 3 + 2] = c.b;
    }

    const nodeGeo = new THREE.BufferGeometry();
    nodeGeo.setAttribute("position", new THREE.BufferAttribute(nodePositions, 3));
    nodeGeo.setAttribute("color",    new THREE.BufferAttribute(nodeColors, 3));

    // High-quality 32px glow sprite
    function makeGlowSprite() {
        const c = document.createElement("canvas");
        c.width = c.height = 32;
        const ctx = c.getContext("2d");
        const g = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
        g.addColorStop(0,   "rgba(255,255,255,1)");
        g.addColorStop(0.18,"rgba(160,200,255,0.9)");
        g.addColorStop(0.55,"rgba(80,110,255,0.3)");
        g.addColorStop(1,   "rgba(0,0,0,0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, 32, 32);
        return new THREE.CanvasTexture(c);
    }

    const nodeMat = new THREE.PointsMaterial({
        size: 6,
        map: makeGlowSprite(),
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true
    });
    masterGroup.add(new THREE.Points(nodeGeo, nodeMat));

    // ---- LAYER 3: Connection lines ----
    const maxConnDist    = 130;
    const linePosArray   = new Float32Array(nodeCount * nodeCount * 6);
    const lineGeo        = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.BufferAttribute(linePosArray, 3));
    const lineMat = new THREE.LineBasicMaterial({
        color: 0x3d6fff,
        transparent: true,
        opacity: 0.16,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
    masterGroup.add(new THREE.LineSegments(lineGeo, lineMat));

    // ---- LAYER 4: Central wireframe icosahedron (security shield) ----
    const coreGeo  = new THREE.IcosahedronGeometry(25, 1);
    const wireGeo  = new THREE.WireframeGeometry(coreGeo);
    const wireMat  = new THREE.LineBasicMaterial({
        color: 0x4f7fff, transparent: true, opacity: 0.22,
        blending: THREE.AdditiveBlending
    });
    const coreWire = new THREE.LineSegments(wireGeo, wireMat);
    masterGroup.add(coreWire);

    // Inner glowing sphere
    const innerMesh = new THREE.Mesh(
        new THREE.SphereGeometry(14, 16, 16),
        new THREE.MeshBasicMaterial({
            color: 0x7c3aed, transparent: true, opacity: 0.12,
            blending: THREE.AdditiveBlending
        })
    );
    masterGroup.add(innerMesh);

    // ---- Mouse parallax ----
    let mouseX = 0, mouseY = 0, smoothX = 0, smoothY = 0;
    window.addEventListener("mousemove", (e) => {
        mouseX = (e.clientX / window.innerWidth  - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    let time = 0;

    function animate() {
        requestAnimationFrame(animate);
        time += 0.005;

        // Slow auto-rotation
        masterGroup.rotation.y += 0.0007;
        masterGroup.rotation.x += 0.00025;

        // Smooth mouse parallax
        smoothX += (mouseX * 0.25 - smoothX) * 0.04;
        smoothY += (-mouseY * 0.18 - smoothY) * 0.04;
        masterGroup.rotation.y += smoothX * 0.008;
        masterGroup.rotation.x += smoothY * 0.008;

        // Pulsing core
        coreWire.rotation.y = time * 0.6;
        coreWire.rotation.x = time * 0.35;
        wireMat.opacity  = 0.15 + Math.sin(time * 2.2) * 0.08;
        innerMesh.material.opacity = 0.08 + Math.sin(time * 3.1) * 0.04;

        // Move nodes with soft spherical boundary (wide sphere limits)
        const posArr = nodeGeo.getAttribute("position").array;
        const maxR   = 600;
        for (let i = 0; i < nodeCount; i++) {
            posArr[i * 3]     += nodeVelocities[i].x;
            posArr[i * 3 + 1] += nodeVelocities[i].y;
            posArr[i * 3 + 2] += nodeVelocities[i].z;

            const dist = Math.sqrt(
                posArr[i*3]**2 + posArr[i*3+1]**2 + posArr[i*3+2]**2
            );
            if (dist > maxR) {
                nodeVelocities[i].x -= posArr[i*3]     * 0.0001;
                nodeVelocities[i].y -= posArr[i*3+1]   * 0.0001;
                nodeVelocities[i].z -= posArr[i*3+2]   * 0.0001;
            }
        }
        nodeGeo.getAttribute("position").needsUpdate = true;

        // Update connections
        let lIdx = 0, connCount = 0;
        for (let i = 0; i < nodeCount; i++) {
            for (let j = i + 1; j < nodeCount; j++) {
                const dx = posArr[i*3]   - posArr[j*3];
                const dy = posArr[i*3+1] - posArr[j*3+1];
                const dz = posArr[i*3+2] - posArr[j*3+2];
                if (Math.sqrt(dx*dx + dy*dy + dz*dz) < maxConnDist) {
                    linePosArray[lIdx++] = posArr[i*3];   linePosArray[lIdx++] = posArr[i*3+1]; linePosArray[lIdx++] = posArr[i*3+2];
                    linePosArray[lIdx++] = posArr[j*3];   linePosArray[lIdx++] = posArr[j*3+1]; linePosArray[lIdx++] = posArr[j*3+2];
                    connCount++;
                }
            }
        }
        lineGeo.getAttribute("position").needsUpdate = true;
        lineGeo.setDrawRange(0, connCount * 2);

        renderer.render(scene, camera);
    }
    animate();
}


// -------------------------------------------------------------
// 16. Single-Pane Enforcement Terminal Tabs toggles logic
// -------------------------------------------------------------
function setupTerminalTabs() {
    const tabButtons = document.querySelectorAll(".terminal-nav-item");
    const tabPanes = document.querySelectorAll(".terminal-pane");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");

            // Toggle active classes on tab buttons
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Toggle active classes on workspace panes
            tabPanes.forEach(pane => {
                if (pane.id === targetId) {
                    pane.classList.add("active");
                } else {
                    pane.classList.remove("active");
                }
            });

            appendConsoleLog("system", `Console Workspace: Switched module view to ${targetId}.py`);

            // Refresh ScrollTrigger as element bounds shifts page height
            setTimeout(() => {
                if (typeof ScrollTrigger !== "undefined") {
                    ScrollTrigger.refresh();
                }
            }, 100);
        });
    });
}

// -------------------------------------------------------------
// 17. Scroll Reveal — IntersectionObserver
// -------------------------------------------------------------
function setupScrollReveal() {
    const revealEls = document.querySelectorAll("[data-reveal]");
    const staggerEls = document.querySelectorAll("[data-reveal-stagger]");

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("is-visible");
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });

    revealEls.forEach(el => observer.observe(el));
    staggerEls.forEach(el => observer.observe(el));
}

// -------------------------------------------------------------
// 18. Demo Terminal — animated log lines
// -------------------------------------------------------------
function initDemoTerminal() {
    const terminal = document.getElementById("demo-terminal-body");
    if (!terminal) return;

    const logs = [
        { type: "ok",     text: "[10:24:01] flow_classifier.py  → MLP model loaded on CUDA:0   | Params: 420K" },
        { type: "system", text: "[10:24:02] Scaler aligned       → RobustScaler fitted on UNSW-NB15" },
        { type: "ok",     text: "[10:24:03] news_verifier.py     → Bi-LSTM loaded  | vocab: 32K tokens" },
        { type: "system", text: "[10:24:04] email_auditor.py     → LSTM ready      | stopwords: NLTK" },
        { type: "ok",     text: "[10:24:05] media_scanner.py     → ResNet-18 ready | TTA: horizontal flip" },
        { type: "info",   text: "[10:24:06] Heartbeat: 1.4 GB GPU / 310 MB RAM | Port 8088 active" },
        { type: "warn",   text: "[10:24:09] THREAT: flow_classifier → SYN FLOOD DDoS  confidence: 97.2%" },
        { type: "ok",     text: "[10:24:12] SAFE:   email_auditor   → Internal sync    confidence: 94.8%" },
        { type: "warn",   text: "[10:24:14] THREAT: media_scanner   → DEEPFAKE GAN     confidence: 91.5%" },
        { type: "ok",     text: "[10:24:17] SAFE:   news_verifier   → Reuters baseline confidence: 89.1%" },
        { type: "info",   text: "[10:24:20] Heartbeat: 1.4 GB GPU / 314 MB RAM | 4 classifiers active" },
        { type: "warn",   text: "[10:24:23] THREAT: email_auditor   → PHISHING URL     confidence: 96.4%" },
    ];

    let idx = 0;

    function addLine() {
        const log = logs[idx % logs.length];
        idx++;
        const div = document.createElement("div");
        div.className = `demo-log-line ${log.type}`;
        div.textContent = log.text;
        terminal.appendChild(div);

        // Trim to max 8 visible lines
        while (terminal.children.length > 8) {
            terminal.removeChild(terminal.firstChild);
        }
        terminal.scrollTop = terminal.scrollHeight;

        setTimeout(addLine, 2200 + Math.random() * 1000);
    }

    // Start after a short delay
    setTimeout(addLine, 800);
}

// -------------------------------------------------------------
// 19. Demo Counters — count-up animation on stat-pill numbers
// -------------------------------------------------------------
function initDemoCounters() {
    const counters = document.querySelectorAll(".stat-num[data-count]");
    if (!counters.length) return;

    const obs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseInt(el.getAttribute("data-count"), 10);
            obs.unobserve(el);

            if (target === 0) return; // already 0, no animation needed

            const duration = 1400;
            const start    = performance.now();
            function tick(now) {
                const elapsed  = now - start;
                const progress = Math.min(elapsed / duration, 1);
                // Ease-out cubic
                const eased = 1 - Math.pow(1 - progress, 3);
                el.textContent = Math.round(eased * target);
                if (progress < 1) requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
        });
    }, { threshold: 0.5 });

    counters.forEach(el => obs.observe(el));
}

