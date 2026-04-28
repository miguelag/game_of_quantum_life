import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════
//  BRANCHING AUTOMATON
//  A deterministic cellular automaton whose wave interference dynamics
//  naturally generate branching parallel worlds — no randomness required.
//
//  Physics summary:
//    • Each cell holds a complex amplitude (real + imaginary parts).
//    • The update rule is the 2D discrete wave equation — purely linear,
//      allowing for exact superposition.
//    • Nonlinearity (NL_ALPHA) and Quantum Foam (FOAM_STRENGTH) amplify 
//      infinitesimal differences between worlds into macroscopic divergence.
//    • When interference creates "tension" (low amplitude but high rate of change),
//      the simulation explicitly branches the world into two outcomes.
//    • Signed Divergence: We measure the aggregate difference of the real
//      wave components to visualize how "different" a world is from World 0.
//    • Worlds that converge back toward each other are merged. Coherence
//      restored — the branch was temporary.
// ═══════════════════════════════════════════════════════════════════════════

const W = 100, H = 100, N = W * H;
const CANVAS_SZ = 360;
const TREE_W = 460, TREE_H = 360;

// Identity threshold: worlds must be closer than this to be considered "nearly identical."
// Merges (especially cap enforcement) will only occur if a pair meets this criteria.
const WORLD_DIFF_MIN = 0.005;

// Number of timesteps to display in the branching tree graph.
const TREE_WINDOW = 600;

// Nonlinear self-interaction coefficient.
// Breaks the exact linearity of the wave equation, causing small
// differences between worlds to amplify over time.
// Negative = defocusing (stable).
const NL_ALPHA = -0.010;

// Wave propagation: r² = (c·Δt/Δx)² with c=0.632. Stability limit for 2D is r²≤0.5.
// This is within the stable range and gives visually pleasing propagation speed.
const WAVE_R2 = 0.40;

// Carrier spatial frequency for drawn wave packets (~16-cell wavelength)
const WAVE_K = Math.PI / 8;

// Phase shift per timestep for directed packet: ω = k·c = WAVE_K·√WAVE_R2
const WAVE_PS = WAVE_K * Math.sqrt(WAVE_R2);

const MAX_WORLDS = 128;  // hard cap on simultaneously active worlds
const SIGMA = 7;         // Gaussian envelope half-width for drawn packets (cells)
const BRANCH_COOLDOWN = 2; // Reduced to allow more reactive branching
const MERGE_GRACE_PERIOD = 100; // Worlds must "mature" before they can be merged

// Quantum Foam Strength: controls how strongly differences between worlds
// trigger new perturbations. This creates a deterministic "butterfly effect"
// where divergence leads to further noise, which is then amplified by NL_ALPHA.
const FOAM_STRENGTH = 0.005;

// Default threshold for the quantum foam to trigger a perturbation.
const FOAM_THRESHOLD_DEFAULT = 0.01;

const _enableCsvLogging = true; // Set to false to disable CSV logging
let _csvLog = [];

// Module-level scratch buffers — allocated once to avoid GC pressure in the hot loop
const _nr = new Float32Array(N);
const _ni = new Float32Array(N);

let _uid = 0; // monotone world-ID counter (no randomness — genealogy is deterministic)

// ─── World management ────────────────────────────────────────────────────────

function mkWorld(parentId = null, t = 0, weight = 1.0, birthDist = 0) {
  const world = {
    id: _uid++, parentId,
    birthTime: t, deathTime: null, mergedInto: null, weight,
    birthDist, // Absolute distance from origin at the moment of birth
    re:    new Float32Array(N),
    im:    new Float32Array(N),
    preRe: new Float32Array(N),
    preIm: new Float32Array(N),
    distHistory: [birthDist], // Absolute displacement from t=0 state
  };
  return world;
}

function cloneWorld(src, t, weight, birthDist) {
  const d = mkWorld(src.id, t, weight, birthDist);
  d.re.set(src.re); d.im.set(src.im);
  d.preRe.set(src.preRe); d.preIm.set(src.preIm);
  return d;
}

// ─── Wave physics ────────────────────────────────────────────────────────────

/**
 * Advance one world by one discrete wave equation step.
 *
 *   next[i] = 2·cur[i] − prev[i] + r²·( L + R + U + D − 4·cur[i] )
 *
 * WHY INTERFERENCE EMERGES: This equation is perfectly LINEAR — superposition
 * holds exactly. Two wave packets passing through the same cell simply add their
 * amplitudes without any interaction term. When they are out of phase, their sum
 * approaches zero even though both contributions continue to vary rapidly. This
 * is the physics of destructive interference, and it creates the tension condition
 * where the automaton's branching pressure lives.
 *
 * Applied identically to the real and imaginary parts. Both components propagate
 * at the same speed c = √(WAVE_R2), preserving their phase relationship.
 *
 * Boundary: Dirichlet (zero amplitude outside the grid). Waves are absorbed at edges.
 */
function stepWorld(w) {
  const { re, im, preRe, preIm } = w;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const Lr = c > 0   ? re[i-1] : 0,  Rr = c < W-1 ? re[i+1] : 0;
      const Ur = r > 0   ? re[i-W] : 0,  Dr = r < H-1 ? re[i+W] : 0;
      const Li = c > 0   ? im[i-1] : 0,  Ri = c < W-1 ? im[i+1] : 0;
      const Ui = r > 0   ? im[i-W] : 0,  Di = r < H-1 ? im[i+W] : 0;

      const magSq = re[i] * re[i] + im[i] * im[i];
      _nr[i] = 2*re[i] - preRe[i] + WAVE_R2*(Lr+Rr+Ur+Dr - 4*re[i]) + NL_ALPHA * magSq * re[i];
      _ni[i] = 2*im[i] - preIm[i] + WAVE_R2*(Li+Ri+Ui+Di - 4*im[i]) + NL_ALPHA * magSq * im[i];
    }
  }
  w.preRe.set(re); w.preIm.set(im);
  w.re.set(_nr);   w.im.set(_ni);
}

/**
 * L2 distance between two amplitude fields.
 */
function calcL2(re1, im1, re2, im2) {
  let s = 0;
  for (let i = 0; i < N; i++) {
    const dr = re1[i] - re2[i], di = im1[i] - im2[i];
    s += dr * dr + di * di;
  }
  return Math.sqrt(s);
}

/**
 * Scan a world for cells in "interference tension."
 *
 * A cell is in tension when simultaneously:
 *   |amplitude| < T    — the resultant is nearly zero (waves canceling)
 *   |Δamplitude| > R   — but the rate of change is high (waves still competing)
 *
 * WHY THIS IDENTIFIES GENUINE SUPERPOSITION: A merely quiet cell has both low
 * amplitude AND low rate of change — it's just empty. A tense cell has low
 * amplitude but high rate of change: the cancellation is dynamic, maintained by
 * two competing wave components that are actively pushing in opposite directions.
 * This is the automaton's analog of a quantum cell that "doesn't know" its
 * outcome — the branching pressure lives here.
 *
 * Returns the count of tense cells and the index + score of the most extreme one.
 */
function scanTension(w, T, R) {
  const { re, im, preRe, preIm } = w;
  let count = 0, best = 0, bIdx = -1;
  for (let i = 0; i < N; i++) {
    const mag  = Math.hypot(re[i], im[i]);
    const rate = Math.hypot(re[i]-preRe[i], im[i]-preIm[i]);
    if (mag < T && rate > R) {
      count++;
      // Score: higher for deeper cancellation at higher rate
      const score = rate * (1 - mag / T);
      if (score > best) { best = score; bIdx = i; }
    }
  }
  return { count, bIdx, best };
}

/**
 * L2 norm of amplitude difference between two worlds.
 * Uses early exit once sum exceeds M² to keep the check cheap.
 */
function l2Diff(a, b, M2) {
  let s = 0;
  for (let i = 0; i < N; i++) {
    const dr = a.re[i]-b.re[i], di = a.im[i]-b.im[i];
    s += dr*dr + di*di;
    if (s > M2) return Infinity;
  }
  return Math.sqrt(s);
}

/**
 * Merge world b into world a.
 *
 * WHY MERGING IS PHYSICALLY MEANINGFUL: Two worlds with indistinguishable
 * amplitude fields are the same world — they are not two realities but one.
 * Their weights (shares of computational/probability resources) add together.
 * This represents the reversal of decoherence: the branching was temporary,
 * and the two outcomes have converged back to a single observable state.
 */
function doMerge(sim, a, b, t) {
  a.weight += b.weight;
  b.deathTime = t; b.mergedInto = a.id;

  if (_enableCsvLogging) {
    _csvLog.push({
      time: t,
      event_type: "merge",
      world_id: b.id,
      parent_id: b.parentId,
      weight: b.weight,
      distance_from_origin: b.distHistory.at(-1) || 0,
      merged_into_id: a.id,
    });
  }
}

/**
 * Advance the full simulation by one timestep.
 *
 * Order of operations:
 *  1. Advance every active world by one wave step.
 *  2. Compute GLOBAL tension (weighted sum across all worlds).
 *  3. Branch at the single most-tense cell (if threshold exceeded and room remains).
 *  4. Merge any pair of worlds that has converged below threshold M.
 *
 * WHY GLOBAL TENSION: Tension is a property of the full wave dynamics, not of
 * any single branch. Each world's tense cells contribute to the global metric
 * weighted by that world's weight — its Born-rule share of "reality." A world
 * with weight 0.25 contributes one quarter as much tension as a full world.
 * This mirrors quantum mechanics: observables are expectation values weighted by |ψ|².
 */
function advance(sim, T, R, M, I, F) {
  const t = sim.time;
  const active = [...sim.worlds.values()].filter(w => !w.deathTime);
  if (!active.length) { sim.time++; return; }

  // Step 1: Advance all active worlds by the wave equation
  for (const w of active) stepWorld(w);

  // Step 1.5: Quantum Foam (Deterministic Interaction)
  // Instead of a global mean field, we now iterate through parent-child pairs.
  // Any difference between a world and its immediate parent triggers a 
  // deterministic perturbation that amplifies that specific genealogical drift.
  if (active.length > 1) {
    for (const w of active) {
      if (w.parentId === null) continue;
      const p = sim.worlds.get(w.parentId);
      if (!p || p.deathTime) continue; // Only interact with active parents

      let localFoamIntensity = 0;
      for (let i = 0; i < N; i++) {
        const dr = w.re[i] - p.re[i], di = w.im[i] - p.im[i];
        const dist = Math.hypot(dr, di);
        if (dist > F) {
          const f = Math.sin(i * 1.5 + t * 0.7) * FOAM_STRENGTH;
          w.re[i] += dr * f; w.im[i] += di * f;
          p.re[i] -= dr * f; // Mutual repulsion drives the lineage apart
          localFoamIntensity += Math.hypot(dr * f, di * f);
        }
      }
      if (localFoamIntensity > 0.001) {
        w.foamEventHistory.push({ t, mag: localFoamIntensity / N });
      }
    }
  }

  // Step 1.7: Update distance history and select world 0 as reference
  for (const w of active) {
    let d;
    if (w.id === 0) {
      d = 0; // World 0 always has L2 of 0 relative to itself
    } else {
      const world0 = sim.worlds.get(0);
      if (world0) {
        d = calcL2(w.re, w.im, world0.re, world0.im);
      } else { d = 0; } // Fallback if world 0 is somehow missing
    }
    w.distHistory.push(d);
  }

  // Step 2: compute global tension + locate the most-tense cell
  let rawTension = 0, topScore = 0, topWorld = null, topIdx = -1;
  for (const w of active) {
    const { count, bIdx, best } = scanTension(w, T, R);
    rawTension += w.weight * count;
    if (best * w.weight > topScore) {
      topScore = best * w.weight; topWorld = w; topIdx = bIdx;
    }
  }
  // Normalize by grid size so the tension metric is scale-independent
  const globalTension = rawTension / N;

  // Step 3: branch at the most-tense cell
  //
  // WHY DETERMINISTIC: No random number generator is consulted anywhere here.
  // The branching direction is determined by the rate-of-change vector of the
  // tense cell — the very wave state that created the tension. Child A is nudged
  // in the +direction of that vector; Child B is nudged in the -direction.
  // Both children are equally real continuations of the wave dynamics.
  // We are not choosing between outcomes — we are acknowledging that both outcomes
  // were already encoded in the superposition, and making that duality explicit.
  const sinceLastBranch = t - (sim.lastBranchTime ?? -999);
  // Lower threshold (0.05) to encourage significantly more branching
  if (topIdx >= 0 && topScore > 0.05 && active.length < MAX_WORLDS && sinceLastBranch >= BRANCH_COOLDOWN) {
    const p = topWorld;
    const r0 = Math.floor(topIdx / W), c0 = topIdx % W;
    const dRe = p.re[topIdx] - p.preRe[topIdx], dIm = p.im[topIdx] - p.preIm[topIdx];
    const rate = Math.hypot(dRe, dIm);
    // Fixed seed nudge: the continuous "foam" process will take this delta
    // and amplify it spatially across the grid in subsequent steps.
    const eps = 0.001;
    const nx = rate > 1e-12 ? dRe/rate : 1, ny = rate > 1e-12 ? dIm/rate : 0;

    const currentDist = p.distHistory.at(-1) || 0;
    const child = cloneWorld(p, t+1, p.weight * 0.5, currentDist);

    // Parent stays alive as outcome A, child becomes outcome B
    p.weight *= 0.5;
    p.re[topIdx] += nx * eps; p.im[topIdx] += ny * eps;
    child.re[topIdx] -= nx * eps; child.im[topIdx] -= ny * eps;

    sim.worlds.set(child.id, child);
    sim.lastBranchTime = t;

    // Enforce world cap: if over limit, merge the two lightest worlds to make room.
    // UPDATED: Now performs a global search for the most similar pair of worlds.
    // Only merges the higher ID into the lower ID if they are closer than threshold I.
    const na = [...sim.worlds.values()].filter(w => !w.deathTime);
    if (na.length > MAX_WORLDS) {
      let bestD = I, bestPair = null;
      for (let i = 0; i < na.length; i++) {
        for (let j = i + 1; j < na.length; j++) {
          const d = l2Diff(na[i], na[j], bestD * bestD);
          if (d < bestD) {
            bestD = d;
            bestPair = [na[i], na[j]];
          }
        }
      }
      if (bestPair) {
        const [w1, w2] = bestPair[0].id < bestPair[1].id ? [bestPair[0], bestPair[1]] : [bestPair[1], bestPair[0]];
        doMerge(sim, w1, w2, t+1);
      }
    }
  }

  // Step 4: check for convergent merging (check every 3 steps for performance)
  if (t % 3 === 0) {
    const na = [...sim.worlds.values()].filter(w => !w.deathTime);
    const M2 = M * M;
    outer: for (let i = 0; i < na.length; i++) { // Iterate through all active worlds
      for (let j = i+1; j < na.length; j++) { // Compare each world with every other world
        // Ignore worlds that were recently born to give them time to diverge
        if (t - na[i].birthTime < MERGE_GRACE_PERIOD || t - na[j].birthTime < MERGE_GRACE_PERIOD) continue;
        if (l2Diff(na[i], na[j], M2) < M) {
          // Merge the higher ID world into the lower ID world
          const w1 = na[i], w2 = na[j];
          if (w1.id < w2.id) {
            doMerge(sim, w1, w2, t+1);
          } else {
            doMerge(sim, w2, w1, t+1);
          }
          break outer;
        }
      }
    }
  }

  sim.time++;
}

// ─── Rendering ───────────────────────────────────────────────────────────────

function hsv2rgb(h, s, v) {
  const c = v*s, x = c*(1 - Math.abs((h/60)%2 - 1)), m = v-c;
  let r,g,b;
  if      (h < 60)  [r,g,b] = [c,x,0];
  else if (h < 120) [r,g,b] = [x,c,0];
  else if (h < 180) [r,g,b] = [0,c,x];
  else if (h < 240) [r,g,b] = [0,x,c];
  else if (h < 300) [r,g,b] = [x,0,c];
  else              [r,g,b] = [c,0,x];
  return [(r+m)*255, (g+m)*255, (b+m)*255];
}

/**
 * Render the amplitude field of one world as an HSV heatmap.
 * Phase → hue (full circular spectrum, continuous at ±π)
 * Magnitude → brightness (value in HSV)
 * Tense cells → bright flash (magnitude + white boost)
 */
function paintField(canvas, world, T, R, parentWorld = null) {
  if (!canvas || !world) return;
  const ctx = canvas.getContext('2d');
  const cw = canvas.width, ch = canvas.height;
  const img = ctx.createImageData(cw, ch);
  const d = img.data;
  const { re, im, preRe, preIm } = world;

  let maxM = 0.0001;
  // Magnitude scaling factor
  for (let i = 0; i < N; i++) maxM = Math.max(maxM, Math.hypot(re[i], im[i]));

  for (let py = 0; py < ch; py++) {
    const row = Math.floor(py * H / ch);
    for (let px = 0; px < cw; px++) {
      const col = Math.floor(px * W / cw);
      const i = row * W + col;
      const rv = re[i], iv = im[i];
      const mag  = Math.hypot(rv, iv);
      const rate = Math.hypot(rv-preRe[i], iv-preIm[i]);
      const tense = mag < T && rate > R;

      const hue = ((Math.atan2(iv, rv) + Math.PI) / (2*Math.PI)) * 360;
      const val = tense ? Math.min(mag/maxM*0.5 + 0.55, 1) : Math.min(mag/maxM, 1);
      let [rp,gp,bp] = hsv2rgb(hue, 0.95, val);
      if (tense) { rp=Math.min(rp+100,255); gp=Math.min(gp+100,255); bp=Math.min(bp+100,255); }

      // Visual mark for foam activity: highlight cells where we differ significantly from parent
      if (parentWorld) {
        const dMag = Math.hypot(re[i] - parentWorld.re[i], im[i] - parentWorld.im[i]);
        if (dMag > 0.02) {
          rp = Math.min(rp + 40, 255);
          bp = Math.min(bp + 60, 255);
        }
      }

      const p = (py * cw + px) * 4;
      d[p] = rp; d[p+1] = gp; d[p+2] = bp; d[p+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * Render the branching tree diagram.
 * Color encodes weight via golden-angle hue and brightness:
 *   heavy worlds (high weight) → bright warm tones
 *   light worlds (low weight)  → dim cool tones
 *   selected world             → glowing highlight
 *
 * Branch events (forks) are marked with dots.
 * Merge/death events are marked with × symbols.
 *
 * UPDATED: Y axis now represents L2 Divergence from Birth. 
 * Every branch starts at Y=0 (bottom) when born and curves upward as it
 * evolves away from the state of its common ancestor.
 */
function paintTree(canvas, sim, selId) {
  if (!canvas || !sim) return;
  const ctx = canvas.getContext('2d');
  const cw = canvas.width, ch = canvas.height;
  ctx.fillStyle = '#030310'; ctx.fillRect(0, 0, cw, ch);

  const { worlds, time } = sim;
  if (time < 1) return;

  const [ML, MR, MT, MB] = [50, 14, 20, 30];
  const PW = cw-ML-MR, PH = ch-MT-MB;

  const t0 = Math.max(0, time - TREE_WINDOW);
  const tRange = TREE_WINDOW;

  const allW = [...worlds.values()];
  // Calculate dynamic max divergence for Y scaling
  let maxD = 0.001;
  for (const w of allW) {
    const birth = w.birthTime, death = w.deathTime ?? time;
    if (death < t0 || birth > time) continue;
    const startIdx = Math.max(0, t0 - birth), endIdx = Math.min(w.distHistory.length - 1, time - birth);
    for (let i = startIdx; i <= endIdx; i++) if (w.distHistory[i] > maxD) maxD = w.distHistory[i];
  }
  const tMax = maxD * 1.1;

  const tx = t  => ML + (t-t0)/tRange*PW;
  const ty = tv => MT + PH * (1 - tv/tMax);

  // World lines
  const maxWt = Math.max(...allW.map(w=>w.weight), 0.001);

  for (const w of allW) {
    const birth = w.birthTime, death = w.deathTime ?? time;
    if (death < t0 || birth > time) continue;
    const s = Math.max(birth, t0), e = Math.min(death, time);
    if (s >= e) continue;

    // Deterministic color from world ID: golden angle hue spacing
    // avoids any two sibling worlds having similar hues
    const hue = (w.id * 137.508) % 360;
    const brt  = 22 + (w.weight/maxWt)*50;
    const alpha = w.deathTime ? 0.22 : (w.id===selId ? 1.0 : 0.72);
    const lwd  = w.id===selId ? 2.8 : (w.weight>0.2 ? 1.5 : 0.75);

    // Per-world y-jitter for visual separation
    const getJy = (id) => ((id*23+7)%42-21)/21 * (PH*0.038);
    const w_jy = getJy(w.id);
    const p_jy = w.parentId !== null ? getJy(w.parentId) : w_jy;

    ctx.save();
    ctx.strokeStyle = `hsla(${hue},90%,${brt}%,${alpha})`;
    ctx.lineWidth = lwd;
    if (w.id === selId) { ctx.shadowBlur=12; ctx.shadowColor=`hsl(${hue},100%,70%)`; }

    ctx.beginPath();
    for (let t=s; t<=e; t++) {
      const hIdx = Math.min(t - birth, w.distHistory.length - 1);
      const dist = w.distHistory[hIdx] ?? 0;
      // Blend jitter over first 10 steps to connect branch to parent line smoothly
      const blend = Math.min((t - birth) / 10, 1.0);
      const currentJy = p_jy * (1 - blend) + w_jy * blend;
      const x=tx(t), y=ty(dist) + currentJy;
      t===s ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke(); ctx.restore();

    // Quantum Foam markers on the line graph
    ctx.save();
    ctx.fillStyle = `hsla(${hue}, 100%, 80%, 0.4)`;
    for (const foam of w.foamEventHistory) {
      if (foam.t >= s && foam.t <= e) {
        const hIdx = Math.min(foam.t - birth, w.distHistory.length - 1);
        const dist = w.distHistory[hIdx] ?? 0;
        const blend = Math.min((foam.t - birth) / 10, 1.0);
        const currentJy = p_jy * (1 - blend) + w_jy * blend;
        ctx.beginPath();
        ctx.arc(tx(foam.t), ty(dist) + currentJy, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();

    // Birth marker: glowing dot at branch event
    if (w.parentId !== null && birth>=t0 && birth<=time) {
      ctx.fillStyle = `hsla(${hue},90%,72%,0.9)`;
      ctx.beginPath();
      ctx.arc(tx(birth), ty(w.birthDist) + p_jy, 3.5, 0, 2*Math.PI);
      ctx.fill();
    }

    // Death/merge marker: × symbol
    if (w.deathTime != null && w.deathTime>=t0 && w.deathTime<=time) {
      const hIdx = Math.min(w.deathTime - birth, w.distHistory.length - 1);
      const dist = w.distHistory[hIdx] ?? 0;
      // Match jitter at death point
      const blend = Math.min((w.deathTime - birth) / 10, 1.0);
      const currentJy = p_jy * (1 - blend) + w_jy * blend;
      const dx=tx(w.deathTime), dy=ty(dist) + currentJy;
      ctx.save(); ctx.strokeStyle='rgba(255,195,70,0.55)'; ctx.lineWidth=1.3;
      ctx.beginPath();
      ctx.moveTo(dx-3.5,dy-3.5); ctx.lineTo(dx+3.5,dy+3.5);
      ctx.moveTo(dx+3.5,dy-3.5); ctx.lineTo(dx-3.5,dy+3.5);
      ctx.stroke(); ctx.restore();
    }
  }

  // Grid lines
  ctx.save(); ctx.strokeStyle='rgba(255,255,255,0.07)'; ctx.lineWidth=0.5;
  for (let f=0.5; f<=1; f+=0.5) {
    const y = ty(tMax*f);
    ctx.beginPath(); ctx.moveTo(ML,y); ctx.lineTo(ML+PW,y); ctx.stroke();
    const yInv = ty(-tMax*f);
    ctx.beginPath(); ctx.moveTo(ML,yInv); ctx.lineTo(ML+PW,yInv); ctx.stroke();
  }
  // Center line (zero)
  ctx.strokeStyle='rgba(255,255,255,0.15)';
  ctx.beginPath(); ctx.moveTo(ML, ty(0)); ctx.lineTo(ML+PW, ty(0)); ctx.stroke();
  ctx.restore();

  // Axes
  ctx.save(); ctx.strokeStyle='rgba(255,255,255,0.13)'; ctx.lineWidth=0.5;
  ctx.beginPath(); ctx.moveTo(ML,MT); ctx.lineTo(ML,MT+PH+2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ML-2,MT+PH); ctx.lineTo(ML+PW,MT+PH); ctx.stroke();
  ctx.restore();

  // Labels
  const nActive = allW.filter(w=>!w.deathTime).length;
  ctx.fillStyle='rgba(110,145,210,0.5)'; ctx.font='9px "Courier New"';
  ctx.textAlign='left';  ctx.fillText('DIVERGENCE', ML+4, MT+13);
  ctx.textAlign='right'; ctx.fillText(tMax.toFixed(4), ML-3, MT+13);
  ctx.fillText('0', ML-3, MT+PH);
  ctx.textAlign='center';
  ctx.fillText(`t = ${time}  ·  ${nActive} world${nActive!==1?'s':''}`, ML+PW/2, MT+PH+20);

  // Phase legend (top right)
  ctx.save();
  const lx=cw-MR-56, ly=MT+2, lr=18;
  ctx.font='8px "Courier New"';
  ctx.fillStyle='rgba(80,100,160,0.4)';
  ctx.fillText('● birth  × merge', lx+30, ly+9);
  ctx.restore();
}

// ─── Wave packet placement ────────────────────────────────────────────────────

/**
 * Place a directed Gaussian wave packet centered at (col, row) into all active worlds.
 *
 * The packet propagates in the direction given by `ph` (angle in radians).
 * The carrier wave is a complex plane wave: e^{i(k_x·x + k_y·y)}.
 * The previous frame is set to the carrier shifted by one step's phase advance (WAVE_PS),
 * which gives the wave packet a definite propagation direction.
 *
 * Drawing applies to ALL active worlds, not just the selected one —
 * the user is setting an initial physical condition shared across all branches.
 */
function placeWaveInto(sim, col, row, ph, amp) {
  const active = [...sim.worlds.values()].filter(w => !w.deathTime);
  const cosPh = Math.cos(ph), sinPh = Math.sin(ph);
  for (const w of active) {
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const dc = c-col, dr = r-row;
        const g = amp * Math.exp(-(dc*dc+dr*dr)/(2*SIGMA*SIGMA));
        if (g < 0.002) continue;
        const carrier = WAVE_K*(dc*cosPh + dr*sinPh);
        const i = r*W+c;
        w.re[i]    += g * Math.cos(carrier);
        w.im[i]    += g * Math.sin(carrier);
        // Previous frame: carrier at t−1 (shifted by one step's dispersion phase)
        w.preRe[i] += g * Math.cos(carrier + WAVE_PS);
        w.preIm[i] += g * Math.sin(carrier + WAVE_PS);
      }
    }
  }
}

// ─── CSV Download ────────────────────────────────────────────────────────────
function downloadCsv() {
  if (!_enableCsvLogging || _csvLog.length === 0) {
    console.warn("CSV logging is not enabled or log is empty.");
    return;
  }

  const header = Object.keys(_csvLog[0]).join(',');
  const rows = _csvLog.map(row => Object.values(row).map(value => {
    if (value === null) return ''; // Handle nulls for CSV
    if (typeof value === 'string' && value.includes(',')) return `"${value}"`; // Quote strings with commas
    return value;
  }).join(','));

  const csvContent = [header, ...rows].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `simulation_log_${Date.now()}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ─── React Component ─────────────────────────────────────────────────────────

function Btn({ children, active, danger, onClick, small }) {
  return (
    <button onClick={onClick} style={{
      background: active ? '#111c35' : '#07070e',
      border: `1px solid ${danger ? '#581818' : active ? '#2d4490' : '#151e30'}`,
      color: active ? '#6a8ed4' : danger ? '#b04545' : '#3d5278',
      padding: small ? '3px 8px' : '5px 13px',
      cursor: 'pointer', fontSize: small ? '9px' : '10px',
      fontFamily: '"Courier New",monospace', letterSpacing: '1.2px', borderRadius: '2px',
      transition: 'all 0.1s',
    }}>
      {children}
    </button>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt, wide }) {
  return (
    <label style={{ display:'flex', alignItems:'center', gap:'5px', cursor:'pointer' }}>
      <span style={{ color:'#344060', fontSize:'10px', letterSpacing:'1px', minWidth:'20px', textAlign:'right' }}>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: wide ? '90px' : '68px', accentColor:'#3356a8', cursor:'pointer' }} />
      <span style={{ color:'#4a6090', fontSize:'10px', fontFamily:'monospace', minWidth:'38px' }}>
        {fmt ? fmt(value) : value.toFixed(3)}
      </span>
    </label>
  );
}

export default function BranchingAutomaton() {
  const fieldRef   = useRef(null);
  const treeRef    = useRef(null);
  const simRef     = useRef(null);
  const rafRef     = useRef(null);
  const drawingRef = useRef(false);

  // Physics param refs — allows RAF loop to always read current values
  // without the stale-closure problem of useEffect dependencies
  const Tr  = useRef(0.15),  Rr  = useRef(0.010), Mr  = useRef(WORLD_DIFF_MIN), Ir = useRef(WORLD_DIFF_MIN), Fr = useRef(FOAM_THRESHOLD_DEFAULT);
  const spr = useRef(3),     phr = useRef(0),      amr = useRef(2.0);
  const sr  = useRef(0); // selected world ID

  const [running,  setRunning]  = useState(false);
  const [drawMode, setDrawMode] = useState(true);
  const [selId,   _setSelId]    = useState(0);
  const [T,       _setT]        = useState(0.15);
  const [R,       _setR]        = useState(0.010);
  const [M,       _setM]        = useState(WORLD_DIFF_MIN);
  const [I,       _setI]        = useState(WORLD_DIFF_MIN);
  const [F,       _setF]        = useState(FOAM_THRESHOLD_DEFAULT);
  const [speed,   _setSpeed]    = useState(3);
  const [phase,   _setPhase]    = useState(0);
  const [ampl,    _setAmpl]     = useState(2.0);
  const [info,     setInfo]     = useState({ t:0, nw:1, tension:'0.000000' });
  const [wList,    setWList]    = useState([{ id:0, weight:'1.0000', birth:0 }]);

  // Setters that update both React state and the corresponding ref
  const setSelId = v => { sr.current=v;  _setSelId(v); };
  const setT     = v => { Tr.current=v;  _setT(v); };
  const setR     = v => { Rr.current=v;  _setR(v); };
  const setM     = v => { Mr.current=v;  _setM(v); };
  const setI     = v => { Ir.current=v;  _setI(v); };
  const setF     = v => { Fr.current=v;  _setF(v); };
  const setSpeed = v => { spr.current=v; _setSpeed(v); };
  const setPhase = v => { phr.current=v; _setPhase(v); };
  const setAmpl  = v => { amr.current=v; _setAmpl(v); };

  // ── Render (stable reference — all data from refs, setters are stable) ──────
  const doRender = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;

    // Auto-switch selected world if it has died
    let w = sim.worlds.get(sr.current);
    if (!w || w.deathTime) {
      const first = [...sim.worlds.values()].find(x => !x.deathTime);
      if (!first) return;
      sr.current = first.id; _setSelId(first.id); w = first;
    }

    paintField(fieldRef.current, w, Tr.current, Rr.current);
    paintTree(treeRef.current, sim, sr.current);

    const active = [...sim.worlds.values()].filter(x => !x.deathTime);
    setInfo({
      t: sim.time, nw: active.length
    });
    setWList(active.map(x => ({
      id: x.id,
      weight: x.weight.toFixed(4),
      birth: x.birthTime,
    })));
  }, []); // stable: all data comes from refs or stable React setters

  // ── Reset ────────────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setRunning(false);
    _uid = 0;
    const root = mkWorld();
    if (_enableCsvLogging) {
      _csvLog = []; // Clear log on reset
    }
    simRef.current = { 
      worlds: new Map([[root.id, root]]), time:0
    };
    setSelId(root.id);
    requestAnimationFrame(doRender);
  }, [doRender]);

  useEffect(() => { reset(); }, []);

  // ── Animation loop ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!running) { if (rafRef.current) cancelAnimationFrame(rafRef.current); return; }
    const loop = () => {
      const s = spr.current;
      for (let i=0; i<s; i++) advance(simRef.current, Tr.current, Rr.current, Mr.current, Ir.current, Fr.current);
      doRender();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [running, doRender]);

  function stepOnce() {
    advance(simRef.current, Tr.current, Rr.current, Mr.current, Ir.current, Fr.current);
    doRender();
  }

  // ── Demo preset: two opposing wave packets ────────────────────────────────────
  function placeDemo() {
    const sim = simRef.current;
    if (!sim) return;
    // Opposing wave packets positioned for center-grid collision
    placeWaveInto(sim, W * 0.25, H * 0.5, 0,       2.5); // rightward
    placeWaveInto(sim, W * 0.75, H * 0.5, Math.PI, 2.5); // leftward
    doRender();
  }

  // ── Drawing ───────────────────────────────────────────────────────────────────
  function getCell(e) {
    const rect = fieldRef.current.getBoundingClientRect();
    return {
      col: Math.max(0, Math.min(W-1, Math.floor((e.clientX-rect.left) / (rect.width/W)))),
      row: Math.max(0, Math.min(H-1, Math.floor((e.clientY-rect.top)  / (rect.height/H)))),
    };
  }

  function handleMouseDown(e) {
    if (!drawMode) return;
    drawingRef.current = true;
    const { col, row } = getCell(e);
    placeWaveInto(simRef.current, col, row, phr.current, amr.current);
    doRender();
  }
  function handleMouseMove(e) {
    if (!drawingRef.current || !drawMode) return;
    const { col, row } = getCell(e);
    placeWaveInto(simRef.current, col, row, phr.current, amr.current);
    doRender();
  }
  function handleMouseUp() { drawingRef.current = false; }

  // ── Layout ───────────────────────────────────────────────────────────────────
  return (
    <div style={{
      background: '#040410', color: '#ccd6f6',
      fontFamily: '"Courier New", Courier, monospace',
      minHeight: '100vh', padding: '14px 16px', boxSizing: 'border-box',
      userSelect: 'none',
    }}>
      {/* Header */}
      <div style={{ borderBottom:'1px solid #0d1525', paddingBottom:'8px', marginBottom:'10px' }}>
        <span style={{ fontSize:'13px', letterSpacing:'5px', color:'#64ffda', fontWeight:'bold' }}>
          BRANCHING AUTOMATON
        </span>
        <span style={{ marginLeft:'16px', fontSize:'9px', color:'#8892b0', letterSpacing:'3px' }}>
          DETERMINISTIC WAVE INTERFERENCE  →  EMERGENT PARALLEL WORLDS
        </span>
      </div>

      {/* Controls */}
      <div style={{ display:'flex', gap:'7px', flexWrap:'wrap', marginBottom:'8px', alignItems:'center' }}>
        <Btn active={drawMode} onClick={()=>setDrawMode(d=>!d)}>
          {drawMode ? '✎ DRAW' : '◎ WATCH'}
        </Btn>
        <Btn onClick={stepOnce}>▶ STEP</Btn>
        <Btn active={running} onClick={()=>setRunning(r=>!r)}>
          {running ? '⏸ PAUSE' : '▶▶ RUN'}
        </Btn>
        <Btn danger onClick={reset}>↺ RESET</Btn>
        <Btn onClick={placeDemo}>★ DEMO</Btn>

        {_enableCsvLogging && <Btn onClick={downloadCsv}>↓ DOWNLOAD CSV</Btn>}
        <div style={{ width:1, height:18, background:'#0d1525', margin:'0 3px' }}/>

        <Slider label="T"   value={T}     min={0.02} max={0.8}  step={0.01}  onChange={setT}
          fmt={v=>v.toFixed(2)} />
        <Slider label="R"   value={R}     min={0.003} max={0.12} step={0.002} onChange={setR}
          fmt={v=>v.toFixed(3)} />
        <Slider label="M"   value={M}     min={0.05}  max={20}  step={0.05}  onChange={setM}
          fmt={v=>v.toFixed(2)} />
        <Slider label="I"   value={I}     min={0.001} max={1.0} step={0.001} onChange={setI}
          fmt={v=>v.toFixed(3)} />
        <Slider label="F"   value={F}     min={0.000} max={0.2} step={0.001} onChange={setF}
          fmt={v=>v.toFixed(3)} />
        <Slider label="SPD" value={speed} min={1}     max={12}  step={1}     onChange={setSpeed}
          fmt={v=>v+'×'} />

        {drawMode && <>
          <div style={{ width:1, height:18, background:'#0d1525', margin:'0 3px' }}/>
          <Slider label="φ"  value={phase} min={0} max={2*Math.PI} step={0.05}
            onChange={setPhase} fmt={v=>(v*180/Math.PI).toFixed(0)+'°'} wide />
          <Slider label="A"  value={ampl}  min={0.2} max={4} step={0.1}
            onChange={setAmpl} fmt={v=>v.toFixed(1)} />
        </>}
      </div>

      {/* Status */}
      <div style={{ display:'flex', gap:'20px', marginBottom:'8px', fontSize:'10px', color:'#a8b2d1' }}>
        <span>t = <b style={{color:'#ccd6f6'}}>{info.t}</b></span>
        <span>active = <b style={{color:'#ccd6f6'}}>{info.nw}</b></span>
        <span style={{color:'#8892b0', fontSize:'9px'}}>
          PARAMS: T={T.toFixed(2)} · R={R.toFixed(3)} · NL_α={NL_ALPHA.toFixed(3)} · Foam={FOAM_STRENGTH.toFixed(3)}
        </span>
      </div>

      {/* Main panels */}
      <div style={{ display:'flex', gap:'14px', alignItems:'flex-start' }}>

        {/* Left: Field canvas */}
        <div>
          <div style={{ fontSize:'9px', color:'#a8b2d1', letterSpacing:'2px', marginBottom:'3px' }}>
            AMPLITUDE FIELD  ·  hue=phase  ·  brightness=magnitude  ·  white flash=tension cell
          </div>
          <canvas
            ref={fieldRef}
            width={CANVAS_SZ} height={CANVAS_SZ}
            style={{
              display:'block',
              cursor: drawMode ? 'crosshair' : 'default',
              border:'1px solid #0d1525', borderRadius:'1px',
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />

          {/* World selector */}
          <div style={{ marginTop:'8px', fontSize:'9px', color:'#a8b2d1', letterSpacing:'2px', marginBottom:'3px' }}>
            WORLD DIRECTORY  <span style={{letterSpacing:0, color:'#8892b0'}}>(click to view)</span>
          </div>
          <div style={{ maxHeight:'102px', overflowY:'auto', border:'1px solid #151e30', borderRadius:'1px' }}>
            {wList.map(w => (
              <div
                key={w.id}
                onClick={() => { setSelId(w.id); doRender(); }}
                style={{
                  padding:'2px 8px', cursor:'pointer', fontSize:'10px',
                  display:'flex', justifyContent:'space-between', gap:'10px',
                  background: w.id===selId ? '#111c35' : 'transparent',
                  color:      w.id===selId ? '#64ffda' : '#8892b0',
                  borderLeft: `2px solid ${w.id===selId ? '#64ffda' : 'transparent'}`,
                }}>
                <span>W{w.id}</span>
                <span>wt={w.weight}</span>
                <span>t₀={w.birth}</span>
              </div>
            ))}
          </div>

          {/* Phase legend strip */}
          <div style={{ marginTop:'8px', display:'flex', alignItems:'center', gap:'6px' }}>
            <span style={{ fontSize:'9px', color:'#182030', letterSpacing:'1px' }}>PHASE</span>
            <div style={{
              width:'120px', height:'8px', borderRadius:'1px',
              background:'linear-gradient(to right,hsl(0,90%,45%),hsl(60,90%,45%),hsl(120,90%,35%),hsl(180,90%,40%),hsl(240,90%,45%),hsl(300,90%,40%),hsl(360,90%,45%))',
            }}/>
            <span style={{ fontSize:'9px', color:'#182030' }}>0°→360°</span>
          </div>
        </div>

        {/* Right: Branching tree */}
        <div>
          <div style={{ fontSize:'9px', color:'#a8b2d1', letterSpacing:'2px', marginBottom:'3px' }}>
            DIVERGENCE HISTORY  ·  y=absolute real-part displacement from world 0
          </div>
          <canvas
            ref={treeRef}
            width={TREE_W} height={TREE_H}
            style={{ display:'block', border:'1px solid #0d1525', borderRadius:'1px' }}
          />
          <div style={{ marginTop:'6px', fontSize:'9px', color:'#8892b0', maxWidth:`${TREE_W}px`, lineHeight:'1.75' }}>
            ● = world fork · × = world convergence · Dots on lines indicate Foam events.<br/>
            Each path represents a deterministic parallel reality. The Y-axis shows the 
            total spatial difference from World 0. When paths merge, it indicates 
            decoherence has been reversed and the outcomes have become 
            indistinguishable.
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div style={{
        marginTop:'12px', borderTop:'1px solid #151e30', paddingTop:'9px',
        fontSize:'9px', color:'#a8b2d1', maxWidth:'870px', lineHeight:'1.9',
      }}>
        <span style={{color:'#64ffda', letterSpacing:'1px'}}>SIMULATION PHYSICS: </span>
        This is a deterministic multiverse. <b>T (Tension)</b> and <b>R (Rate)</b> detect zones 
        of destructive interference where the wave field is "undecided." In these zones, the world 
        is split into two outcomes. <b>Foam (F)</b> triggers a feedback loop where worlds push 
        away from their immediate ancestors, amplified by the <b>NL_α</b> nonlinearity. 
        <b>M (Merge)</b> allows worlds that have physically converged back to the same state to 
        reunite, representing the restoration of coherence.
      </div>
    </div>
  );
}
