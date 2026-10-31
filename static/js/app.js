/* AI Face Mask Detection System - frontend controller */

(function () {
  "use strict";

  var els = {
    pillBackend: document.getElementById("pill-backend"),
    pillModel: document.getElementById("pill-model"),
    modelBanner: document.getElementById("model-banner"),
    modeSelect: document.getElementById("mode-select"),
    btnStart: document.getElementById("btn-start"),
    btnStop: document.getElementById("btn-stop"),
    btnReset: document.getElementById("btn-reset"),
    stageOverlay: document.getElementById("stage-overlay"),
    stageOverlayText: document.getElementById("stage-overlay-text"),
    video: document.getElementById("video"),
    canvas: document.getElementById("canvas"),
    stream: document.getElementById("stream"),
    liveBadge: document.getElementById("live-badge"),
    statFaces: document.getElementById("stat-faces"),
    statMask: document.getElementById("stat-mask"),
    statNomask: document.getElementById("stat-nomask"),
    statLatency: document.getElementById("stat-latency"),
    faceList: document.getElementById("face-list"),
    sumFrames: document.getElementById("sum-frames"),
    sumFaces: document.getElementById("sum-faces"),
    sumMask: document.getElementById("sum-mask"),
    sumNomask: document.getElementById("sum-nomask"),
    sumRatio: document.getElementById("sum-ratio"),
    logBody: document.getElementById("log-body")
  };

  var state = {
    running: false,
    mode: "browser",
    stream: null,
    loopTimer: null,
    pollTimer: null,
    modelLoaded: false,
    frames: 0,
    faces: 0,
    mask: 0,
    noMask: 0,
    log: [],
    lastLogTime: 0,
    busy: false
  };

  var DETECT_INTERVAL_MS = 160;
  var POLL_INTERVAL_MS = 600;
  var HEALTH_INTERVAL_MS = 10000;
  var LOG_THROTTLE_MS = 1000;
  var MAX_LOG_ROWS = 25;

  /* ------------------------------------------------------------------ helpers */
  function setOverlay(text, show) {
    els.stageOverlayText.innerHTML = text;
    els.stageOverlay.hidden = !show;
  }

  function setPill(el, label, status, detail) {
    el.dataset.state = status;
    el.textContent = detail ? label + ": " + detail : label;
  }

  function showPreview(which) {
    els.video.hidden = which !== "video";
    els.canvas.hidden = which !== "video";
    els.stream.hidden = which !== "stream";
  }

  function resetSession() {
    state.frames = 0;
    state.faces = 0;
    state.mask = 0;
    state.noMask = 0;
    state.log = [];
    renderCounts({ faces: 0, with_mask: 0, without_mask: 0 }, null);
    renderFaceList([]);
    renderSummary();
    renderLog();
  }

  function renderCounts(counts, latencyMs) {
    els.statFaces.textContent = counts.faces;
    els.statMask.textContent = counts.with_mask;
    els.statNomask.textContent = counts.without_mask;
    els.statLatency.textContent =
      latencyMs === null || latencyMs === undefined
        ? "\u2014"
        : Number(latencyMs).toFixed(1) + " ms";
  }

  function renderSummary() {
    els.sumFrames.textContent = state.frames;
    els.sumFaces.textContent = state.faces;
    els.sumMask.textContent = state.mask;
    els.sumNomask.textContent = state.noMask;
    els.sumRatio.textContent =
      state.faces > 0 ? ((state.mask / state.faces) * 100).toFixed(1) + "%" : "\u2014";
  }

  function tagClass(label) {
    return label === "with_mask" ? "tag-mask" : "tag-nomask";
  }

  function barClass(label) {
    return label === "with_mask" ? "bar-mask" : "bar-nomask";
  }

  function renderFaceList(faces) {
    if (!faces.length) {
      els.faceList.innerHTML = '<li class="empty">No faces in view yet.</li>';
      return;
    }
    var html = "";
    faces.forEach(function (face) {
      var pct = Math.round(face.confidence * 100);
      html +=
        '<li class="face-item">' +
        '<span class="tag ' + tagClass(face.label) + '">' + face.display_label + "</span>" +
        '<span class="bar ' + barClass(face.label) + '"><span style="width:' + pct + '%"></span></span>' +
        '<span class="conf">' + pct + "%</span>" +
        "</li>";
    });
    els.faceList.innerHTML = html;
  }

  function renderLog() {
    if (!state.log.length) {
      els.logBody.innerHTML =
        '<tr><td colspan="3" class="empty">No detections logged this session.</td></tr>';
      return;
    }
    var html = "";
    state.log.slice(0, MAX_LOG_ROWS).forEach(function (row) {
      var cls = row.label === "with_mask" ? "c-mask" : "c-nomask";
      html +=
        "<tr><td>" + row.time + '</td><td class="' + cls + '">' + row.display_label +
        "</td><td>" + Math.round(row.confidence * 100) + "%</td></tr>";
    });
    els.logBody.innerHTML = html;
  }

  function pushLog(faces) {
    if (!faces.length) {
      return;
    }
    var now = Date.now();
    if (now - state.lastLogTime < LOG_THROTTLE_MS) {
      return;
    }
    state.lastLogTime = now;
    var time = new Date().toLocaleTimeString();
    faces.forEach(function (face) {
      state.log.unshift({
        time: time,
        label: face.label,
        display_label: face.display_label,
        confidence: face.confidence
      });
    });
    renderLog();
  }

  function accumulateCounts(counts) {
    state.frames += 1;
    state.faces += counts.faces;
    state.mask += counts.with_mask;
    state.noMask += counts.without_mask;
    renderSummary();
  }

  /* ------------------------------------------------------------------ health */
  function refreshHealth() {
    return fetch("/api/health")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        setPill(els.pillBackend, "Backend", data.status === "online" ? "ok" : "error", data.status);
        state.modelLoaded = data.model === "loaded";
        if (state.modelLoaded) {
          setPill(els.pillModel, "Model", "ok", "loaded");
          els.modelBanner.hidden = true;
        } else {
          setPill(els.pillModel, "Model", "warn", "missing");
          els.modelBanner.hidden = false;
        }
        return data;
      })
      .catch(function () {
        setPill(els.pillBackend, "Backend", "error", "offline");
        setPill(els.pillModel, "Model", "unknown", "n/a");
      });
  }

  /* --------------------------------------------------------- browser camera */
  function stopBrowserStream() {
    if (state.stream) {
      state.stream.getTracks().forEach(function (track) { track.stop(); });
      state.stream = null;
    }
    els.video.srcObject = null;
  }

  function startBrowser() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setOverlay(
        "Camera API unavailable in this browser. Try the <strong>Server webcam</strong> mode instead.",
        true
      );
      return Promise.reject(new Error("getUserMedia unavailable"));
    }
    setOverlay("Requesting camera access\u2026", true);
    return navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      })
      .then(function (stream) {
        state.stream = stream;
        els.video.srcObject = stream;
        els.video.muted = true;
        els.video.playsInline = true;
        return els.video.play().then(function () {
          setOverlay("", false);
          els.liveBadge.hidden = false;
          showPreview("video");
          scheduleBrowserLoop();
        });
      })
      .catch(function (err) {
        if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
          setOverlay(
            "<strong>Camera permission denied.</strong> Allow camera access in your browser settings and try again.",
            true
          );
        } else if (err && (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")) {
          setOverlay("<strong>No camera found.</strong> Connect a webcam or switch to server mode.", true);
        } else {
          setOverlay(
            "<strong>Camera unavailable.</strong> " + (err && err.message ? err.message : "Unknown error."),
            true
          );
        }
        throw err;
      });
  }

  function scheduleBrowserLoop() {
    if (!state.running) {
      return;
    }
    state.loopTimer = setTimeout(runBrowserDetect, DETECT_INTERVAL_MS);
  }

  function runBrowserDetect() {
    if (!state.running || state.busy) {
      scheduleBrowserLoop();
      return;
    }
    var video = els.video;
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      scheduleBrowserLoop();
      return;
    }

    var canvas = els.canvas;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    var dataUrl = canvas.toDataURL("image/jpeg", 0.8);

    state.busy = true;
    fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, status: res.status, data: data };
        });
      })
      .then(function (result) {
        state.busy = false;
        if (result.status === 503) {
          setPill(els.pillModel, "Model", "warn", "missing");
          els.modelBanner.hidden = false;
          stopDetection();
          setOverlay("<strong>Model missing.</strong> Train a model, then reload this page.", true);
          return;
        }
        if (!result.ok || result.data.status !== "success") {
          scheduleBrowserLoop();
          return;
        }
        var faces = result.data.faces || [];
        drawDetections(faces);
        renderCounts(result.data.counts, result.data.latency_ms);
        renderFaceList(faces);
        accumulateCounts(result.data.counts);
        pushLog(faces);
        scheduleBrowserLoop();
      })
      .catch(function () {
        state.busy = false;
        scheduleBrowserLoop();
      });
  }

  function drawDetections(faces) {
    var canvas = els.canvas;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(els.video, 0, 0, canvas.width, canvas.height);
    var lineWidth = Math.max(2, Math.round(canvas.width / 480));
    var fontSize = Math.max(13, Math.round(canvas.width / 70));
    ctx.font = "600 " + fontSize + "px system-ui, sans-serif";
    faces.forEach(function (face) {
      var x = face.bbox[0], y = face.bbox[1], w = face.bbox[2], h = face.bbox[3];
      var color = face.label === "with_mask" ? "#3fb950" : "#f85149";
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      var label = face.display_label + " " + Math.round(face.confidence * 100) + "%";
      var pad = 6;
      var textW = ctx.measureText(label).width;
      var boxH = fontSize + pad * 2;
      var boxY = Math.max(y - boxH - 4, 0);
      ctx.fillStyle = color;
      ctx.fillRect(x, boxY, textW + pad * 2, boxH);
      ctx.fillStyle = "#0b0f14";
      ctx.fillText(label, x + pad, boxY + boxH - pad - 2);
    });
  }

  /* ---------------------------------------------------------- server camera */
  function startServer() {
    setOverlay("Starting server webcam\u2026", true);
    return fetch("/api/camera/start", { method: "POST" })
      .then(function (res) {
        return res.json().then(function (data) { return { ok: res.ok, data: data }; });
      })
      .then(function (result) {
        if (!result.ok) {
          setOverlay(
            "<strong>Camera unavailable.</strong> " +
              (result.data.message || "Could not open the webcam."),
            true
          );
          throw new Error(result.data.message || "camera start failed");
        }
        els.stream.src = "/video_feed?t=" + Date.now();
        setOverlay("", false);
        els.liveBadge.hidden = false;
        showPreview("stream");
        state.pollTimer = setInterval(pollDetections, POLL_INTERVAL_MS);
      });
  }

  function pollDetections() {
    fetch("/api/detections")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!data.camera_active && state.running) {
          stopDetection();
          setOverlay(
            "<strong>Camera stopped.</strong> " + (data.camera_error || "Restart detection to continue."),
            true
          );
          return;
        }
        renderCounts(data.counts, data.latency_ms);
        renderFaceList(data.faces || []);
        if (data.counts && data.counts.faces > 0) {
          accumulateCounts(data.counts);
          pushLog(data.faces || []);
        } else if (data.camera_active) {
          state.frames += 1;
          renderSummary();
        }
      })
      .catch(function () { /* transient poll failures are fine */ });
  }

  function stopServer() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    els.stream.removeAttribute("src");
    fetch("/api/camera/stop", { method: "POST" }).catch(function () { /* ignore */ });
  }

  /* ----------------------------------------------------------- start/stop */
  function startDetection() {
    if (state.running) {
      return;
    }
    state.running = true;
    state.mode = els.modeSelect.value;
    els.modeSelect.disabled = true;
    els.btnStart.disabled = true;
    els.btnStop.disabled = false;
    resetSession();

    refreshHealth().then(function (health) {
      if (!health || health.model !== "loaded") {
        stopDetection();
        setOverlay(
          "<strong>Model missing.</strong> Train one with <code>python -m training.train --data-dir dataset</code> and reload.",
          true
        );
        return;
      }
      var starter = state.mode === "server" ? startServer() : startBrowser();
      starter.catch(function () { stopDetection(); });
    });
  }

  function stopDetection() {
    state.running = false;
    state.busy = false;
    if (state.loopTimer) {
      clearTimeout(state.loopTimer);
      state.loopTimer = null;
    }
    els.liveBadge.hidden = true;
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
    els.modeSelect.disabled = false;

    if (state.mode === "server") {
      stopServer();
    } else {
      stopBrowserStream();
      var ctx = els.canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
      }
    }
    showPreview("none");
    if (els.modelBanner.hidden) {
      setOverlay("Detection stopped. Press <strong>Start detection</strong> to resume.", true);
    } else {
      setOverlay("Press <strong>Start detection</strong> to open the camera.", true);
    }
  }

  function resetSessionRemote() {
    resetSession();
    fetch("/api/events/reset", { method: "POST" }).catch(function () { /* ignore */ });
  }

  /* ------------------------------------------------------------------ wiring */
  els.btnStart.addEventListener("click", startDetection);
  els.btnStop.addEventListener("click", stopDetection);
  els.btnReset.addEventListener("click", resetSessionRemote);
  els.modeSelect.addEventListener("change", function () {
    if (state.running) {
      stopDetection();
    }
  });
  window.addEventListener("beforeunload", function () {
    state.running = false;
    stopBrowserStream();
    if (state.mode === "server") {
      navigator.sendBeacon("/api/camera/stop", "");
    }
  });

  showPreview("none");
  refreshHealth();
  setInterval(refreshHealth, HEALTH_INTERVAL_MS);
})();
