"""
api/dashboard.py
================
통합 졸음 감지 대시보드 HTML 생성.

섹션:
  1. IR 카메라 영상 스트림
  2. 카메라 지표 카드 (EAR, PERCLOS, MAR, Pitch)
  3. EEG 지표 카드 (Alpha/Beta, Relative Theta, Blink Rate, 신호 품질)
  4. 최종 융합 판정 보드 (NORMAL / CAUTION / WARNING / DROWSY)

JS 폴링: /metrics/all 단일 엔드포인트를 50ms 주기로 호출하여
카메라·EEG·융합 결과를 동시 갱신한다.
"""


def build_dashboard_html(video_feed_url: str = "/video_feed") -> str:
    html = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>졸음 감지 통합 대시보드</title>
  <style>
    :root {
      --bg: #0a0a0a; --surface: #161616; --surface2: #1f1f1f;
      --border: #2a2a2a; --border-active: #3a3a3a;
      --green: #4caf50; --orange: #ff9800; --yellow: #ffeb3b;
      --red-mild: #ff5722; --red: #ff4d4d;
      --text: #e8e8e8; --muted: #666; --font: 'Segoe UI', system-ui, sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg); color: var(--text); font-family: var(--font);
      display: flex; flex-direction: column; align-items: center;
      padding: 28px 20px; gap: 28px; min-height: 100vh;
    }

    /* ── 헤더 ── */
    header { text-align: center; }
    header h1 { font-weight: 300; font-size: 24px; letter-spacing: 2px; color: #fff; }
    header p  { font-size: 12px; color: var(--muted); margin-top: 4px; letter-spacing: 1px; }

    /* ── 영상 ── */
    .video-wrap {
      border: 2px solid var(--border); border-radius: 14px;
      overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,0.8);
    }

    /* ── 섹션 제목 ── */
    .section-label {
      width: 100%; max-width: 920px;
      font-size: 10px; color: var(--muted); letter-spacing: 2px;
      text-transform: uppercase; padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
    }

    /* ── 메트릭 그리드 ── */
    .grid {
      display: grid; gap: 14px; width: 100%; max-width: 920px;
    }
    .grid-4 { grid-template-columns: repeat(4, 1fr); }
    .grid-4-eeg { grid-template-columns: repeat(4, 1fr); }

    .card {
      background: var(--surface); border-radius: 10px; padding: 18px 14px;
      text-align: center; border-top: 3px solid var(--green);
      transition: border-top-color 0.4s, box-shadow 0.4s;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .card--eeg { border-top-color: #7c4dff; }
    .card__label {
      font-size: 10px; color: var(--muted); text-transform: uppercase;
      letter-spacing: 1.5px; margin-bottom: 10px;
    }
    .card__value { font-size: 28px; font-weight: 700; line-height: 1; }
    .card__sub   { font-size: 11px; color: var(--muted); margin-top: 8px; }
    .card__bar-bg {
      height: 4px; background: var(--border); border-radius: 2px; margin-top: 10px; overflow: hidden;
    }
    .card__bar { height: 100%; border-radius: 2px; width: 0%; transition: width 0.4s, background 0.4s; }

    /* ── EEG 연결 상태 배지 ── */
    .eeg-badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 3px 10px; border-radius: 20px; font-size: 11px;
      background: var(--surface2); border: 1px solid var(--border);
    }
    .eeg-badge__dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
    .eeg-badge__dot--on  { background: var(--green); box-shadow: 0 0 6px var(--green); }
    .eeg-badge__dot--off { background: var(--red); }

    /* ── 최종 판정 보드 ── */
    .verdict-board {
      width: 100%; max-width: 920px;
      background: var(--surface); border-radius: 14px;
      padding: 28px 24px; text-align: center;
      border: 2px solid var(--border);
      transition: border-color 0.4s, box-shadow 0.4s;
    }
    .verdict-board__level {
      font-size: 44px; font-weight: 800; letter-spacing: 4px;
      transition: color 0.4s;
    }
    .verdict-board__score {
      margin-top: 10px; font-size: 14px; color: var(--muted);
      display: flex; justify-content: center; gap: 24px; flex-wrap: wrap;
    }
    .verdict-board__score span { white-space: nowrap; }

    /* 판정 바 */
    .score-bar-wrap {
      width: 100%; max-width: 500px; margin: 14px auto 0;
    }
    .score-bar-bg {
      height: 8px; border-radius: 4px; overflow: hidden;
      background: linear-gradient(to right, var(--green), var(--orange), var(--red-mild), var(--red));
    }
    .score-needle-wrap { position: relative; height: 14px; margin-top: 4px; }
    .score-needle {
      position: absolute; top: 0; transform: translateX(-50%);
      width: 2px; height: 10px; background: #fff; border-radius: 1px;
      transition: left 0.4s;
    }
    .score-needle-label {
      position: absolute; top: 12px; transform: translateX(-50%);
      font-size: 10px; color: var(--muted); transition: left 0.4s;
    }

    /* ── 색상 상태 클래스 ── */
    .ok      { color: var(--green); }
    .caution { color: var(--orange); }
    .warn    { color: var(--red-mild); animation: pulse-warn 1.2s ease-in-out infinite; }
    .drowsy  { color: var(--red);     animation: pulse-warn 0.8s ease-in-out infinite; }

    @keyframes pulse-warn {
      0%, 100% { text-shadow: none; }
      50%       { text-shadow: 0 0 22px currentColor; }
    }
  </style>
</head>
<body>

  <header>
    <h1>DROWSINESS MONITOR</h1>
    <p>카메라 + EEG 다중모달 실시간 졸음 감지 시스템</p>
  </header>

  <div class="video-wrap">
    <img src="__VIDEO_FEED_URL__" width="640" height="480" alt="IR 카메라 스트림">
  </div>

  <!-- 카메라 + 캘리브레이션 상태 바 -->
  <div class="section-label" style="display:flex; justify-content:space-between; align-items:center;">
    <span>Camera Metrics</span>
    <div style="display:flex; align-items:center; gap:12px;">
      <button id="btn-recalib" style="background:var(--surface2); border:1px solid var(--border); color:var(--text); padding:4px 10px; border-radius:4px; font-size:11px; cursor:pointer; transition:0.3s;">🔄 재보정(Recalibrate)</button>
      <span class="eeg-badge">
        <span class="eeg-badge__dot" id="cam-dot"></span>
        <span id="cam-status-text">카메라 대기 중</span>
      </span>
    </div>
  </div>
  <!-- 캘리브레이션 진행 바 -->
  <div id="calib-wrap" style="width:100%; max-width:920px; display:none;">
    <div style="font-size:11px; color:var(--muted); margin-bottom:6px; letter-spacing:1px;">
      🔧 EAR 캘리브레이션 진행 중 — 완료 전까지 졸음 점수가 계산되지 않습니다
    </div>
    <div style="height:4px; background:var(--border); border-radius:2px; overflow:hidden;">
      <div id="calib-bar" style="height:100%; background:var(--orange); width:0%; transition:width 0.5s;"></div>
    </div>
  </div>
  <div class="grid grid-4">
    <div class="card" id="card-ear">
      <div class="card__label">EAR — 눈 개방비</div>
      <div class="card__value" id="v-ear">—</div>
      <div class="card__sub">임계값: <span id="v-thresh">—</span></div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-ear"></div></div>
    </div>
    <div class="card" id="card-perclos">
      <div class="card__label">PERCLOS</div>
      <div class="card__value" id="v-perclos">—</div>
      <div class="card__sub">60초 창 · 위험 ≥70%</div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-perclos"></div></div>
    </div>
    <div class="card" id="card-mar">
      <div class="card__label">MAR — 하품</div>
      <div class="card__value" id="v-mar">—</div>
      <div class="card__sub">임계값: 0.600</div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-mar"></div></div>
    </div>
    <div class="card" id="card-pitch">
      <div class="card__label">Head Pitch</div>
      <div class="card__value" id="v-pitch">—</div>
      <div class="card__sub">임계값: 25.0°</div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-pitch"></div></div>
    </div>
  </div>

  <!-- EEG 지표 -->
  <div class="section-label" style="display:flex; justify-content:space-between; align-items:center;">
    <span>EEG Metrics (Muse2)</span>
    <span class="eeg-badge">
      <span class="eeg-badge__dot" id="eeg-dot"></span>
      <span id="eeg-status-text">연결 대기 중</span>
    </span>
  </div>
  <div class="grid grid-4-eeg">
    <div class="card card--eeg" id="card-ab">
      <div class="card__label">Alpha / Beta 비율</div>
      <div class="card__value" id="v-ab">—</div>
      <div class="card__sub">정상 ≈1.5 · 위험 ≥4.0</div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-ab"></div></div>
    </div>
    <div class="card card--eeg" id="card-theta">
      <div class="card__label">Relative Theta</div>
      <div class="card__value" id="v-theta">—</div>
      <div class="card__sub">정상 ≈0.12 · 위험 ≥0.40</div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-theta"></div></div>
    </div>
    <div class="card card--eeg" id="card-blink">
      <div class="card__label">Blink Rate</div>
      <div class="card__value" id="v-blink">—</div>
      <div class="card__sub">정상 12–20 회/분</div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-blink"></div></div>
    </div>
    <div class="card card--eeg" id="card-sq">
      <div class="card__label">Signal Quality</div>
      <div class="card__value" id="v-sq">—</div>
      <div class="card__sub">0.0(불량) – 1.0(양호)</div>
      <div class="card__bar-bg"><div class="card__bar" id="bar-sq"></div></div>
    </div>
  </div>

  <!-- 최종 판정 -->
  <div class="section-label" style="display:flex; justify-content:space-between; align-items:center;">
    <span>Final Verdict</span>
    <span style="font-size:10px; color:var(--muted);">점수 0 = 정상(졸음 없음) · 점수 1 = 최대 졸음</span>
  </div>
  <div class="verdict-board" id="verdict-board">
    <div class="verdict-board__level ok" id="v-level">WAITING...</div>
    <div class="score-bar-wrap">
      <div class="score-bar-bg"></div>
      <div class="score-needle-wrap">
        <div class="score-needle" id="v-needle" style="left:0%"></div>
        <div class="score-needle-label" id="v-needle-label" style="left:0%">0.00</div>
      </div>
    </div>
    <div class="verdict-board__score" id="v-score-detail">
      <span>카메라: <b id="v-cam-score">—</b></span>
      <span>EEG: <b id="v-eeg-score">—</b></span>
      <span>최종: <b id="v-final-score">—</b></span>
      <span>신뢰도: <b id="v-conf">—</b></span>
    </div>
  </div>
  <script>
    const $ = id => document.getElementById(id);

    function colorForScore(s) {
      if (s >= 0.65) return 'var(--red)';
      if (s >= 0.45) return 'var(--red-mild)';
      if (s >= 0.25) return 'var(--orange)';
      return 'var(--green)';
    }

    function setBar(barId, pct, color) {
      const el = $(barId);
      el.style.width  = Math.min(pct * 100, 100) + '%';
      el.style.background = color;
    }

    function levelClass(level) {
      return { NORMAL:'ok', CAUTION:'caution', WARNING:'warn', DROWSY:'drowsy' }[level] || 'ok';
    }

    /* ── 카메라 활성 상태 렌더 ── */
    function renderCameraStatus(cameraActive, cam) {
      const dot  = $('cam-dot');
      const text = $('cam-status-text');
      const calibWrap = $('calib-wrap');
      const calibBar  = $('calib-bar');

      if (!cameraActive) {
        dot.className  = 'eeg-badge__dot eeg-badge__dot--off';
        text.textContent = '카메라 미연결 / 오류';
        calibWrap.style.display = 'none';
        return;
      }

      // 카메라 활성 상태 세분화
      if (cam.status === 'No Face') {
        dot.className  = 'eeg-badge__dot';
        dot.style.background = 'var(--orange)';
        text.textContent = '얼굴 미검출 — 카메라 앞에 앉아주세요';
        calibWrap.style.display = 'none';
      } else if (!cam.is_calibrated) {
        dot.className  = 'eeg-badge__dot';
        dot.style.background = 'var(--orange)';
        text.textContent = cam.status;   // "Calibrating... Ns"

        // 캘리브레이션 진행 바 표시
        calibWrap.style.display = 'block';
        const match = cam.status.match(/(\d+)s/);
        const remaining = match ? parseInt(match[1]) : 0;
        const pct = ((5 - remaining) / 5) * 100;
        calibBar.style.width = pct + '%';
      } else {
        dot.className  = 'eeg-badge__dot eeg-badge__dot--on';
        dot.style.background = '';
        text.textContent = '활성 · 캘리브레이션 완료';
        calibWrap.style.display = 'none';
      }
    }

    /* ── 카메라 지표 렌더 ── */
    function renderCamera(c, cameraActive) {
      renderCameraStatus(cameraActive, c);

      $('v-ear').textContent    = c.ear.toFixed(3);
      $('v-thresh').textContent = c.threshold.toFixed(3);
      $('v-perclos').textContent = c.perclos.toFixed(1) + '%';
      $('v-mar').textContent    = c.mar.toFixed(3);
      $('v-pitch').textContent  = c.pitch.toFixed(1) + '°';

      const earPct = c.threshold > 0 ? Math.max(0, (c.threshold - c.ear) / c.threshold) : 0;
      setBar('bar-ear', earPct, earPct > 0.5 ? 'var(--red)' : 'var(--green)');
      setBar('bar-perclos', c.perclos / 100, colorForScore(c.perclos / 100));
      const marPct = c.mar > 0.6 ? (c.mar - 0.6) / 0.4 : 0;
      setBar('bar-mar', marPct, marPct > 0.5 ? 'var(--red)' : 'var(--green)');
      const pitchPct = c.pitch > 25 ? (c.pitch - 25) / 25 : 0;
      setBar('bar-pitch', pitchPct, pitchPct > 0.5 ? 'var(--red)' : 'var(--green)');
    }

    /* ── EEG 지표 렌더 ── */
    function renderEEG(e) {
      const connected = e.is_connected;
      $('eeg-dot').className  = 'eeg-badge__dot ' + (connected ? 'eeg-badge__dot--on' : 'eeg-badge__dot--off');
      $('eeg-status-text').textContent = connected ? `연결됨 (품질: ${(e.signal_quality*100).toFixed(0)}%)` : 'Muse2 미연결 (스텁 모드)';

      $('v-ab').textContent    = e.alpha_beta_ratio.toFixed(2);
      $('v-theta').textContent = e.relative_theta.toFixed(3);
      $('v-blink').textContent = e.blink_rate.toFixed(1);
      $('v-sq').textContent    = e.signal_quality.toFixed(2);

      const abPct = Math.min((e.alpha_beta_ratio - 1.5) / 2.5, 1);
      setBar('bar-ab',    Math.max(abPct, 0),   colorForScore(Math.max(abPct, 0)));
      const thetaPct = Math.min((e.relative_theta - 0.12) / 0.28, 1);
      setBar('bar-theta', Math.max(thetaPct, 0), colorForScore(Math.max(thetaPct, 0)));
      const blinkDev = Math.abs(e.blink_rate - 15) / 15;
      setBar('bar-blink', Math.min(blinkDev, 1), blinkDev > 0.5 ? 'var(--orange)' : 'var(--green)');
      setBar('bar-sq',    e.signal_quality, e.signal_quality < 0.5 ? 'var(--red)' : 'var(--green)');
    }

    /* ── 최종 판정 렌더 ── */
    function renderFusion(f) {
      const levelEl = $('v-level');
      levelEl.textContent = f.level;
      levelEl.className   = 'verdict-board__level ' + levelClass(f.level);

      const boardEl = $('verdict-board');
      const color = { NORMAL:'var(--border)', CAUTION:'var(--orange)', WARNING:'var(--red-mild)', DROWSY:'var(--red)' }[f.level];
      boardEl.style.borderColor = color;
      boardEl.style.boxShadow   = f.level === 'NORMAL' ? 'none' : `0 0 30px ${color}44`;

      // 점수 바 바늘
      const pct = (f.final_score * 100).toFixed(1);
      $('v-needle').style.left       = pct + '%';
      $('v-needle-label').style.left = pct + '%';
      $('v-needle-label').textContent = f.final_score.toFixed(2);

      $('v-cam-score').textContent   = f.camera_score ? f.camera_score.total.toFixed(3) : '—';
      $('v-eeg-score').textContent   = f.eeg_score    ? f.eeg_score.total.toFixed(3)    : '(미연결)';
      $('v-final-score').textContent = f.final_score.toFixed(3);
      $('v-conf').textContent        = (f.confidence * 100).toFixed(0) + '%';
    }

    /* ── 폴링 (50ms = 20fps UI 갱신) ── */
    setInterval(() => {
      fetch('/metrics/all')
        .then(r => r.json())
        .then(d => {
          renderCamera(d.camera, d.camera_active);
          renderEEG(d.eeg);
          renderFusion(d.fusion);
        })
        .catch(err => console.error('metrics 수신 실패:', err));
    }, 50);

    /* ── 캘리브레이션 재설정 ── */
    $('btn-recalib').addEventListener('click', () => {
      fetch('/metrics/camera/reset_calibration', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          console.log('재보정 요청됨:', d);
          $('btn-recalib').style.background = 'var(--green)';
          setTimeout(() => { $('btn-recalib').style.background = 'var(--surface2)'; }, 500);
        })
        .catch(err => console.error('재보정 요청 실패:', err));
    });
  </script>
</body>
</html>"""
    return html.replace("__VIDEO_FEED_URL__", video_feed_url)
