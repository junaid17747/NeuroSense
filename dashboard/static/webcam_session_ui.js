// Render aggregate measurements only. Values never cross the Streamlit bridge.
export function renderBehaviorSession(select, session, now, ready) {
  const stats = session.snapshot(now), active = ["running", "paused", "quality_paused"].includes(stats.state);
  const format = (number, unit = "", digits = 1) => Number.isFinite(number) ? `${number.toFixed(digits)}${unit}` : "—";
  select("pause-monitoring").disabled = stats.state !== "running";
  select("resume-monitoring").disabled = !["paused", "quality_paused"].includes(stats.state) || !ready;
  select("stop-monitoring").disabled = !active;
  select("tiredness-before").disabled = active;
  select("tiredness-after").disabled = stats.state !== "stopped";
  if (select("session-status").textContent !== session.reason) select("session-status").textContent = session.reason;
  select("blink-count").textContent = format(stats.blinkCount, "", 0);
  select("blink-rate").textContent = format(stats.blinkRate);
  select("eye-closure").textContent = stats.state === "running" ? format(stats.closureMs === null && session.measurements?.eyeState === "open" ? 0 : stats.closureMs, " ms", 0) : "—";
  select("last-closure").textContent = `Last closure: ${format(stats.lastClosureMs, " ms", 0)}`;
  select("closed-percent").textContent = format(stats.closedPercent, "%");
  select("head-pose").textContent = stats.pose ? [stats.pose.yaw, stats.pose.pitch, stats.pose.roll].map(n => format(n, "°")).join(" / ") : "—";
  select("head-movement").textContent = format(stats.motion, "°/s");
  select("tracking-confidence").textContent = stats.confidence;
  select("measurement-quality").textContent = format(stats.coverage, "%");
  select("valid-time").textContent = `Valid observed time: ${format((stats.validMs || 0) / 1000, " s")}`;
  select("rating-summary").textContent = session.state === "idle" ? "Self-reports are separate from camera measurements."
    : `Self-reported tiredness — Before: ${session.before ?? "not provided"}; After: ${session.after ?? "not provided"}. Separate from camera measurements.`;

  const points = session.timeline;
  const timeLabel = time => new Date(time).toLocaleTimeString([], { hour12: false });
  if (!points.length) {
    select("closed-timeline").setAttribute("d", ""); select("motion-timeline").setAttribute("d", "");
    select("timeline-start").textContent = "—"; select("timeline-end").textContent = "—";
    select("timeline-summary").textContent = "Start a session to see a timeline. Gaps represent unavailable measurements.";
    return;
  }
  const first = points[0].time, last = points.at(-1).time, range = Math.max(1000, last - first);
  const path = (field, max, bottom) => {
    let move = true, result = "";
    for (const point of points) {
      if (point.gap || !Number.isFinite(point[field])) { move = true; continue; }
      const x = 8 + (point.time - first) / range * 584;
      const y = bottom - Math.min(max, Math.max(0, point[field])) / max * 60;
      result += `${move ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)} `; move = false;
    }
    return result;
  };
  select("closed-timeline").setAttribute("d", path("closedPercent", 100, 92));
  select("motion-timeline").setAttribute("d", path("motion", 60, 188));
  select("timeline-start").textContent = timeLabel(first); select("timeline-end").textContent = timeLabel(last);
  select("timeline-summary").textContent = `${timeLabel(last)} · ${stats.blinkCount} blinks observed. Paused or invalid intervals are gaps. Summary cards retain valid session totals.`;
}
