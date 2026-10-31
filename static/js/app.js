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
    previewBadge: document.getElementById("preview-badge"),
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
    lastFaceHtml: null,
    lastLogHtml: null,
    busy: false,
    previewOnly: false
  };

  var DETECT_INTERVAL_MS = 160;
  var POLL_INTERVAL_MS = 600;
  var HEALTH_INTERVAL_MS = 10000;

  /* One shared confidence format: model confidence -> "87.3%". */
  function formatPct(confidence) {
    return (Number(confidence) * 100).toFixed(1) + "%";
  }

  function barWidth(confidence) {
    var pct = Number(confidence) * 100;
    if (!isFinite(pct)) {
      return 0;
    }
    return Math.max(0, Math.min(100, Math.round(pct)));
  }

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
    renderCounts({ faces: 0, with_mask: 0, without_mask: 0 }, null);
    renderFaceList([]);
    renderSession(null);
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

  /* Session-wide statistics, straight from the server's tracker:
     frames analysed (each frame once), distinct face detections counted,
     and mask ratio = mask detections / total face detections (null -> "—"). */
  function renderSession(session) {
    var s = session || { frames: 0, faces: 0, mask: 0, no_mask: 0, ratio: null, events: [] };
    els.sumFrames.textContent = s.frames || 0;
    els.sumFaces.textContent = s.faces || 0;
    els.sumMask.textContent = s.mask || 0;
    els.sumNomask.textContent = s.no_mask || 0;
    els.sumRatio.textContent =
      s.ratio === null || s.ratio === undefined ? "\u2014" : (s.ratio * 100).toFixed(1) + "%";
    renderLog(s.events || []);
  }

  function tagClass(label) {
    return label === "with_mask" ? "tag-mask" : "tag-nomask";
  }

  function barClass(label) {
    return label === "with_mask" ? "bar-mask" : "bar-nomask";
  }

  function renderFaceList(faces) {
    if (!faces.length) {
      if (state.lastFaceHtml !== null) {
        state.lastFaceHtml = null;
        els.faceList.innerHTML = '<li class="empty">No faces in view yet.</li>';
      }
      return;
    }
    var html = "";
    faces.forEach(function (face) {
      html +=
        '<li class="face-item">' +
        '<span class="tag ' + tagClass(face.label) + '">' + face.display_label + "</span>" +
        '<span class="bar ' + barClass(face.label) + '"><span style="width:' +
        barWidth(face.confidence) + '%"></span></span>' +
        '<span class="conf">' + formatPct(face.confidence) + "</span>" +
        "</li>";
    });
    if (html === state.lastFaceHtml) {
      return; // identical content: skip DOM rewrite to avoid flicker
    }
    state.lastFaceHtml = html;
    els.faceList.innerHTML = html;
  }

  function renderLog(events) {
    var html;
    if (!events.length) {
      html = '<tr><td colspan="3" class="empty">No detections logged this session.</td></tr>';
    } else {
      html = "";
      events.forEach(function (row) {
        var cls = row.label === "with_mask" ? "c-mask" : "c-nomask";
        html +=
          '<tr><td class="c-time">' + row.time + '</td><td class="' + cls + '">' +
          row.display_label + '</td><td class="c-conf">' + formatPct(row.confidence) + "</td></tr>";
      });
    }
    if (html === state.lastLogHtml) {
      return;
    }
    state.lastLogHtml = html;
    els.logBody.innerHTML = html;
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
        // If the browser ends the stream (permission revoked, device pulled),
        // stop cleanly instead of leaving a dead loop behind.
        stream.getTracks().forEach(function (track) {
          track.addEventListener("ended", function () {
            if (!state.running) {
              return;
            }
            stopDetection();
            setOverlay(
              "<strong>Camera stream ended.</strong> Permission was revoked or the device was disconnected.",
              true
            );
          });
        });
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
    if (!state.running) {
      return;
    }
    if (state.previewOnly) {
      // Camera preview works without a model; just keep the loop alive.
      scheduleBrowserLoop();
      return;
    }
    if (state.busy) {
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
        if (!state.running) {
          return; // stopped while this frame was in flight: do not count it
        }
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
        renderSession(result.data.session);
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

      var label = face.display_label + " " + formatPct(face.confidence);
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
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
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
        if (!state.running) {
          return; // stopped while this poll was in flight
        }
        if (!data.camera_active) {
          stopDetection();
          setOverlay(
            "<strong>Camera stopped.</strong> " + (data.camera_error || "Restart detection to continue."),
            true
          );
          return;
        }
        if (data.model_loaded === false) {
          state.previewOnly = true;
          els.previewBadge.hidden = false;
        }
        renderCounts(data.counts, data.latency_ms);
        renderFaceList(data.faces || []);
        renderSession(data.session);
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
    resetSessionRemote(); // clear local widgets AND the server-side session

    refreshHealth().then(function (health) {
      state.previewOnly = !health || health.model !== "loaded";
      if (state.previewOnly) {
        els.previewBadge.hidden = false;
        els.stageNote.textContent =
          "Model not trained yet - showing camera preview only, no classifications. " +
          "Train one with: python -m training.train --data-dir dataset, then reload this page.";
      } else {
        els.previewBadge.hidden = true;
        els.stageNote.textContent =
          "Frames are analysed in your browser (or by the OpenCV server camera) and classified one face at a time.";
      }
      var starter = state.mode === "server" ? startServer() : startBrowser();
      starter.catch(function () { stopDetection(); });
    });
  }

  function stopDetection() {
    if (!state.running) {
      return; // already stopped (double click, auto-stop, mode change)
    }
    state.running = false;
    state.busy = false;
    if (state.loopTimer) {
      clearTimeout(state.loopTimer);
      state.loopTimer = null;
    }
    els.liveBadge.hidden = true;
    els.previewBadge.hidden = true;
    state.previewOnly = false;
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
    els.modeSelect.disabled = false;
    els.stageNote.textContent =
      "Frames are analysed in your browser (or by the OpenCV server camera) and classified one face at a time.";

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
