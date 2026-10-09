import assert from "node:assert/strict";
import test from "node:test";
import { loadBehavior, face, frontalMatrix, rotation } from "./webcam_test_helpers.mjs";

const { BEHAVIOR, eyeAspectRatios, EyeCalibration, BlinkMeasurements, BehavioralSession, behavioralQuality, estimateHeadPose, tirednessRating } = await loadBehavior();
const baseline = { left: .3, right: .3 }, open = { left: .3, right: .3 }, closed = { left: .1, right: .1 };
const sample = (eyes = open, pose = { yaw: 0, pitch: 0, roll: 0 }) => ({ eyes, pose, valid: true });

test("EAR uses both eye contours and correct pixel aspect on portrait/landscape cameras", () => {
  for (const [width, height] of [[640,480], [360,640], [1920,1080]]) {
    const eyes = eyeAspectRatios(face({ width, height, leftEAR: .3, rightEAR: .25 }), width, height);
    assert(Math.abs(eyes.left - .3) < 1e-9); assert(Math.abs(eyes.right - .25) < 1e-9);
  }
});

test("missing, out-of-frame, nonfinite and unresolved eye landmarks fail closed", () => {
  assert.equal(eyeAspectRatios([],640,480),null);
  for (const x of [NaN, -1, 2]) {
    const landmarks = face(); landmarks[33].x = x;
    assert.equal(eyeAspectRatios(landmarks,640,480),null);
  }
  assert.equal(eyeAspectRatios(face({ fill: .01 }),640,480),null);
});

test("eye baseline is individually calibrated, robust to occasional blinks and rejects closed-eye setup", () => {
  const calibration = new EyeCalibration();
  assert.equal(calibration.baseline(), null);
  for (let time = 0; time <= 3000; time += 50) calibration.add(time % 500 === 0 ? closed : { left: .28, right: .33 }, time);
  assert.deepEqual(calibration.baseline(), { left: .28, right: .33 });
  calibration.reset();
  for (let time = 0; time <= 3000; time += 50) calibration.add(closed,time);
  assert.equal(calibration.baseline(),null);
});

test("complete bilateral blink counts once, with interpolated closure duration", () => {
  const measurements = new BlinkMeasurements(baseline);
  measurements.update(open,0); measurements.update(open,50);
  measurements.update(closed,100); measurements.update(closed,150);
  measurements.update(open,200); measurements.update(open,250);
  assert.equal(measurements.count,1);
  assert.equal(measurements.lastClosureMs,100);
  assert.equal(measurements.closedMs,100);
  assert.equal(measurements.snapshot(250).closedPercent,40);
});

test("brief noise, threshold jitter, initial closed eyes and long closure do not inflate blinks", () => {
  const noisy = new BlinkMeasurements(baseline);
  noisy.update(open,0); noisy.update(closed,40); noisy.update(open,80);
  assert.equal(noisy.count,0);
  const jitter = new BlinkMeasurements(baseline);
  jitter.update(open,0);
  for (let t = 50; t <= 500; t += 50) jitter.update({left:.22,right:.22},t);
  assert.equal(jitter.count,0);
  const initiallyClosed = new BlinkMeasurements(baseline);
  initiallyClosed.update(closed,0); initiallyClosed.update(closed,50); initiallyClosed.update(open,100);
  assert.equal(initiallyClosed.count,0);
  const long = new BlinkMeasurements(baseline); long.update(open,0);
  for (let t = 50; t <= 1000; t += 50) long.update(closed,t);
  long.update(open,1050);
  assert.equal(long.count,0); assert.equal(long.lastClosureMs,1000); assert(long.closedMs > 0);
});

test("blinks do not span missing frames, duplicate timestamps, pauses or invalid eyes", () => {
  for (const interrupt of [m => m.interrupt(), m => m.update(null,110), m => m.update(open,500), m => m.update(closed,100)]) {
    const m = new BlinkMeasurements(baseline); m.update(open,0); m.update(closed,50); m.update(closed,100);
    interrupt(m); m.update(open,550);
    assert.equal(m.count,0);
  }
});

test("rate uses valid observed time and remains unavailable for short samples", () => {
  const m = new BlinkMeasurements(baseline);
  for (let t = 0; t <= 10000; t += 50) m.update(t === 100 || t === 150 ? closed : open,t);
  assert.equal(m.snapshot(10000).blinkRate,6);
  m.interrupt(); m.update(open,60000); m.update(open,60050);
  assert.equal(m.validMs,10050);
  assert.equal(new BlinkMeasurements(baseline).snapshot(0).blinkRate,null);
});

test("tracking loss, multiple faces, lighting, pose and eye inconsistency invalidate measurements", () => {
  const bounds = {left:.3,right:.7,top:.2,bottom:.8};
  for (const code of ["no_face","multiple_faces","too_dark","too_bright","rotation","too_far","invalid"]) {
    assert.equal(behavioralQuality({code,message:code},open,baseline,true,bounds).valid,false);
  }
  assert.equal(behavioralQuality({code:"aligned"},open,baseline,false,bounds).valid,false);
  assert.equal(behavioralQuality({code:"aligned"},{left:.1,right:.3},baseline,true,bounds).valid,false);
  assert.equal(behavioralQuality({code:"left"},open,baseline,true,bounds).valid,true);
});

test("head pose reports camera-relative rotations and movement does not bridge pauses", () => {
  const pose = estimateHeadPose(frontalMatrix);
  for (const angle of Object.values(pose)) assert(Math.abs(angle) < 1e-8);
  assert(Math.abs(estimateHeadPose(rotation(10,"y")).yaw - 10) < 1e-8);
  const session = new BehavioralSession(); session.start(0,1000,baseline);
  session.sample(sample(open,{yaw:0,pitch:0,roll:0}),0);
  session.sample(sample(open,{yaw:1,pitch:0,roll:0}),100);
  assert.equal(session.snapshot(100).motion,10);
  session.pause(100); session.resume(10000,baseline); session.sample(sample(open,{yaw:15,pitch:0,roll:0}),10000);
  assert.equal(session.snapshot(10000).motion,null);
});

test("session lifecycle is explicit, paused data is excluded and Stop is final", () => {
  const session = new BehavioralSession(); session.sample(sample(),0);
  assert.equal(session.measurements,null);
  assert.equal(session.start(0,1000,null),false);
  assert.equal(session.start(0,1000,baseline,"4"),true);
  assert.equal(session.start(1,1001,baseline),false);
  session.sample(sample(),0); session.sample(sample(),100);
  session.pause(100); session.sample(sample(closed),200);
  assert.equal(session.snapshot(200).validMs,100);
  assert.equal(session.resume(2000,null),false);
  assert.equal(session.resume(2000,baseline),true);
  session.sample(sample(),2000); session.sample(sample(),2100);
  assert.equal(session.snapshot(2100).validMs,200);
  session.stop(2100); session.sample(sample(),2200);
  assert.equal(session.resume(2200,baseline),false);
  assert.equal(session.snapshot(2200).validMs,200);
});

test("bad measurements automatically pause, clear instantaneous values and require Resume", () => {
  const session = new BehavioralSession(); session.start(0,1000,baseline);
  session.sample(sample(),0); session.sample(sample(),50);
  session.sample({valid:false,reason:"Face lost"},100);
  assert.equal(session.state,"quality_paused");
  assert.equal(session.snapshot(100).pose,null);
  assert.equal(session.snapshot(100).confidence,"Unavailable");
  session.sample(sample(),200);
  assert.equal(session.state,"quality_paused");
  assert.equal(session.timeline.at(-1).gap,true);
});

test("ratings are optional, bounded, separate, and only accepted after Stop", () => {
  for (const bad of ["",null,undefined,11,-1,2.5,"unknown"]) assert.equal(tirednessRating(bad),null);
  const session = new BehavioralSession(); session.start(0,1000,baseline,"0");
  session.setAfter("9"); assert.equal(session.after,null);
  session.stop(50); session.setAfter("9");
  assert.equal(session.before,0); assert.equal(session.after,9); assert.equal(session.measurements.count,0);
});

test("new sessions, new instances and cleanup isolate all metrics and ratings", () => {
  const first = new BehavioralSession(), second = new BehavioralSession();
  first.start(0,1000,baseline,"5"); first.sample(sample(),0); first.sample(sample(),100); first.stop(100); first.setAfter("8");
  assert.equal(second.measurements,null); assert.equal(second.before,null); assert.deepEqual(second.timeline,[]);
  first.start(200,1200,baseline);
  assert.equal(first.before,null); assert.equal(first.after,null); assert.equal(first.measurements.validMs,0); assert.deepEqual(first.timeline,[]);
  first.clear(); assert.equal(first.measurements,null); assert.equal(first.before,null); assert.equal(first.state,"idle");
});

test("timeline keeps only bounded aggregate values with wall-clock timestamps", () => {
  const session = new BehavioralSession(); session.start(0,1700000000000,baseline);
  for (let t = 0; t <= 400000; t += 100) session.sample(sample(),t);
  assert.equal(session.timeline.length,BEHAVIOR.timelineLimit);
  assert.equal(session.timeline.at(-1).time,1700000400000);
  for (const point of session.timeline) assert.deepEqual(Object.keys(point).sort(),["blinks","closedPercent","gap","motion","pitch","roll","time","yaw"].sort());
});

test("invalid numeric samples never poison totals or create a blink", () => {
  for (const eyes of [{left:NaN,right:.3},{left:-.1,right:-.1},{left:2,right:2}]) {
    const session = new BehavioralSession(); session.start(0,1000,baseline);
    session.sample(sample(),0); session.sample(sample(),50);
    session.sample(sample(eyes),100);
    assert.equal(session.state,"quality_paused");
    assert.equal(session.snapshot(100).blinkCount,0);
    assert(Number.isFinite(session.snapshot(100).coverage));
  }
  const session = new BehavioralSession(); session.start(0,1000,baseline);
  session.sample(sample(),0); session.sample(sample(),50); session.sample(sample(),NaN);
  assert.equal(session.state,"quality_paused");
  assert.equal(session.snapshot(100).validMs,50);
  assert(Number.isFinite(session.runningMs));
});
