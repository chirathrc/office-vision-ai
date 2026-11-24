export type OfficeActivity =
  | "ON_PHONE"
  | "WORKING_AT_PC"
  | "READING"
  | "ON_BREAK"
  | "IDLE";

export interface RawDetection {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}

export interface PersonActivity {
  personIndex: number;
  label: string;
  activity: OfficeActivity;
  confidence: number;
  bbox: [number, number, number, number];
  nearbyObjects: string[];
}

export interface FrameAnalysis {
  people: PersonActivity[];
  objects: RawDetection[];
  counts: {
    total: number;
    working: number;
    onPhone: number;
    readingOrBreak: number;
    idle: number;
  };
}

export const ACTIVITY_META: Record<
  OfficeActivity,
  { label: string; color: string; badgeBg: string }
> = {
  ON_PHONE: {
    label: "ON PHONE",
    color: "#ef4444",
    badgeBg: "bg-red-500/20 text-red-300 border-red-500/40",
  },
  WORKING_AT_PC: {
    label: "WORKING AT PC",
    color: "#22c55e",
    badgeBg: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  },
  READING: {
    label: "READING / DOCS",
    color: "#38bdf8",
    badgeBg: "bg-sky-500/20 text-sky-300 border-sky-500/40",
  },
  ON_BREAK: {
    label: "BREAK / DRINK",
    color: "#f59e0b",
    badgeBg: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  },
  IDLE: {
    label: "IDLE / STANDING",
    color: "#94a3b8",
    badgeBg: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  },
};

const WORK_OBJECTS = new Set(["laptop", "keyboard", "mouse", "tv"]);
const BREAK_OBJECTS = new Set(["cup", "bottle", "bowl", "wine glass"]);
const TRACKED_OBJECTS = new Set([
  "cell phone",
  "laptop",
  "keyboard",
  "mouse",
  "tv",
  "book",
  "cup",
  "bottle",
]);

function boxesOverlapOrNear(
  personBox: [number, number, number, number],
  objBox: [number, number, number, number],
  marginRatio: number
): boolean {
  const [px, py, pw, ph] = personBox;
  const [ox, oy, ow, oh] = objBox;

  const mx = pw * marginRatio;
  const my = ph * marginRatio;

  const pLeft = px - mx;
  const pRight = px + pw + mx;
  const pTop = py - my;
  const pBottom = py + ph + my;

  const oRight = ox + ow;
  const oBottom = oy + oh;

  return pLeft < oRight && pRight > ox && pTop < oBottom && pBottom > oy;
}

export function classifyOfficeFrame(
  detections: RawDetection[],
  minScore = 0.4
): FrameAnalysis {
  const filtered = detections.filter((d) => d.score >= minScore);

  const persons = filtered
    .filter((d) => d.class === "person")
    .sort((a, b) => a.bbox[0] - b.bbox[0]); // Stable left-to-right indexing

  const objects = filtered.filter((d) => TRACKED_OBJECTS.has(d.class));

  const people: PersonActivity[] = persons.map((person, idx) => {
    const nearby: string[] = [];

    for (const obj of objects) {
      // Tighter proximity for cell phone, wider desk proximity for laptop/monitor
      const margin = obj.class === "cell phone" ? 0.15 : 0.35;
      if (boxesOverlapOrNear(person.bbox, obj.bbox, margin)) {
        if (!nearby.includes(obj.class)) {
          nearby.push(obj.class);
        }
      }
    }

    let activity: OfficeActivity = "IDLE";
    if (nearby.includes("cell phone")) {
      activity = "ON_PHONE";
    } else if (nearby.some((o) => WORK_OBJECTS.has(o))) {
      activity = "WORKING_AT_PC";
    } else if (nearby.includes("book")) {
      activity = "READING";
    } else if (nearby.some((o) => BREAK_OBJECTS.has(o))) {
      activity = "ON_BREAK";
    }

    return {
      personIndex: idx + 1,
      label: `Person #${idx + 1}`,
      activity,
      confidence: person.score,
      bbox: person.bbox,
      nearbyObjects: nearby,
    };
  });

  const counts = {
    total: people.length,
    working: people.filter((p) => p.activity === "WORKING_AT_PC").length,
    onPhone: people.filter((p) => p.activity === "ON_PHONE").length,
    readingOrBreak: people.filter(
      (p) => p.activity === "READING" || p.activity === "ON_BREAK"
    ).length,
    idle: people.filter((p) => p.activity === "IDLE").length,
  };

  return { people, objects, counts };
}

export function drawDetectionsOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  analysis: FrameAnalysis
) {
  ctx.clearRect(0, 0, width, height);

  // Draw detected office objects subtly
  for (const obj of analysis.objects) {
    const [x, y, w, h] = obj.bbox;
    const isPhone = obj.class === "cell phone";
    ctx.strokeStyle = isPhone ? "#f87171" : "#38bdf8";
    ctx.lineWidth = isPhone ? 2.5 : 1.5;
    ctx.setLineDash(isPhone ? [] : [5, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    const tag = `${obj.class.toUpperCase()} ${Math.round(obj.score * 100)}%`;
    ctx.font = "600 11px ui-monospace, SFMono-Regular, monospace";
    const tw = ctx.measureText(tag).width + 10;
    ctx.fillStyle = isPhone ? "rgba(220,38,38,0.88)" : "rgba(15,23,42,0.85)";
    ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(tag, x + 5, Math.max(12, y - 5));
  }

  // Draw each person with their classified activity HUD
  for (const person of analysis.people) {
    const [x, y, w, h] = person.bbox;
    const meta = ACTIVITY_META[person.activity];

    ctx.strokeStyle = meta.color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    const title = `${person.label} · ${meta.label} (${Math.round(
      person.confidence * 100
    )}%)`;
    ctx.font = "700 13px system-ui, sans-serif";
    const tw = ctx.measureText(title).width + 14;
    const labelY = Math.max(0, y - 24);
    ctx.fillStyle = meta.color;
    ctx.fillRect(x, labelY, tw, 24);
    ctx.fillStyle = "#0f172a";
    ctx.fillText(title, x + 7, labelY + 16);
  }
}
