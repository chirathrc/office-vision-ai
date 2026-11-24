import fs from "node:fs";
import path from "node:path";
import jpeg from "jpeg-js";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as knnClassifier from "@tensorflow-models/knn-classifier";

const DATASET_DIR = path.resolve("training/dataset");
const OUTPUT_MODEL_PATH = path.resolve("public/pretrained-office-model.json");

const CLASS_QUERIES = {
  "Working at PC": [
    "person working laptop",
    "typing on laptop computer",
    "man working computer office",
    "woman working laptop desk",
    "software developer computer",
    "office worker desktop monitor",
    "student studying laptop",
    "programming laptop keyboard",
  ],
  "Using Phone": [
    "person using smartphone",
    "woman looking at smartphone",
    "man holding mobile phone",
    "texting on cell phone",
    "person talking on mobile phone",
    "hand holding smartphone screen",
    "reading smartphone hand",
    "calling on cellphone",
  ],
  "Reading a Book": [
    "person reading a book",
    "woman reading open book",
    "man reading book library",
    "student reading textbook",
    "holding open book reading",
    "reading paper book desk",
    "person reading magazine",
    "reading notebook pen",
  ],
  "On Break / Drinking": [
    "person drinking coffee cup",
    "woman drinking coffee mug",
    "man drinking water bottle",
    "drinking tea cup office",
    "person holding coffee mug",
    "drinking glass of water",
    "coffee break office",
    "sipping coffee cup",
  ],
  "Idle / Away": [
    "office desk",
    "empty chair office",
    "office interior workspace",
    "modern office room",
    "conference room empty chairs",
    "desk monitor chair",
    "coworking space empty",
    "office cubicle",
  ],
};

const TARGET_PER_CLASS = 60;
const CONCURRENCY = 6;

async function searchWikimediaJpegs(query, limit = 25) {
  const url =
    "https://commons.wikimedia.org/w/api.php?" +
    new URLSearchParams({
      action: "query",
      generator: "search",
      gsrnamespace: "6",
      gsrsearch: `${query} filetype:bitmap`,
      gsrlimit: String(limit),
      prop: "imageinfo",
      iiprop: "url|mime",
      iiurlwidth: "320",
      format: "json",
    });

  try {
    const res = await fetch(url, {
      headers: { "User-Agent": "OfficeVisionAI-Trainer/2.0 (local-training)" },
    });
    if (!res.ok) return [];
    const data = await res.json();
    const pages = Object.values(data?.query?.pages || {});
    const urls = [];
    for (const p of pages) {
      const info = p?.imageinfo?.[0];
      if (!info) continue;
      const mime = (info.mime || "").toLowerCase();
      const imgUrl = info.thumburl || info.url;
      if (
        imgUrl &&
        (mime.includes("jpeg") ||
          imgUrl.toLowerCase().endsWith(".jpg") ||
          imgUrl.toLowerCase().endsWith(".jpeg"))
      ) {
        urls.push(imgUrl);
      }
    }
    return urls;
  } catch {
    return [];
  }
}

async function downloadAndVerify(imgUrl, outPath) {
  try {
    const res = await fetch(imgUrl, {
      headers: { "User-Agent": "OfficeVisionAI-Trainer/2.0" },
    });
    if (!res.ok) return false;
    const buf = Buffer.from(await res.arrayBuffer());
    const decoded = jpeg.decode(buf, { useTArray: true, maxMemoryUsageInMB: 128 });
    if (!decoded || decoded.width < 64 || decoded.height < 64) return false;
    fs.writeFileSync(outPath, buf);
    return true;
  } catch {
    return false;
  }
}

async function downloadDataset() {
  fs.mkdirSync(DATASET_DIR, { recursive: true });
  const classFiles = {};

  for (const [label, queries] of Object.entries(CLASS_QUERIES)) {
    const safeDir = path.join(
      DATASET_DIR,
      label.replace(/[^a-z0-9]/gi, "_").toLowerCase()
    );
    fs.mkdirSync(safeDir, { recursive: true });

    // Reuse any already-downloaded verified JPEGs first
    const existing = fs
      .readdirSync(safeDir)
      .filter((f) => f.endsWith(".jpg"))
      .map((f) => path.join(safeDir, f));

    const saved = [...existing];
    if (saved.length < TARGET_PER_CLASS) {
      const seen = new Set();
      const candidateUrls = [];
      for (const q of queries) {
        const found = await searchWikimediaJpegs(q, 25);
        for (const u of found) {
          if (!seen.has(u)) {
            seen.add(u);
            candidateUrls.push(u);
          }
        }
      }

      let nextIdx = saved.length + 1;
      for (let i = 0; i < candidateUrls.length && saved.length < TARGET_PER_CLASS; i += CONCURRENCY) {
        const batch = candidateUrls.slice(i, i + CONCURRENCY);
        const results = await Promise.all(
          batch.map(async (imgUrl, offset) => {
            const filePath = path.join(
              safeDir,
              `img_${String(nextIdx + offset).padStart(3, "0")}.jpg`
            );
            const ok = await downloadAndVerify(imgUrl, filePath);
            return ok ? filePath : null;
          })
        );
        nextIdx += batch.length;
        for (const fp of results) {
          if (fp && saved.length < TARGET_PER_CLASS) {
            saved.push(fp);
          } else if (fp) {
            fs.unlinkSync(fp);
          }
        }
      }
    }

    classFiles[label] = saved.slice(0, TARGET_PER_CLASS);
    console.log(`[Dataset] ${label}: ${classFiles[label].length} verified JPEGs ready`);
  }

  return classFiles;
}

function decodeJpegToRgbTensorsWithFlip(filePath) {
  const buf = fs.readFileSync(filePath);
  const raw = jpeg.decode(buf, { useTArray: true, maxMemoryUsageInMB: 128 });
  const { width, height, data } = raw;
  const rgb = new Uint8Array(width * height * 3);
  const flipped = new Uint8Array(width * height * 3);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcIdx = (y * width + x) * 4;
      const dstIdx = (y * width + x) * 3;
      const flipIdx = (y * width + (width - 1 - x)) * 3;

      rgb[dstIdx] = data[srcIdx];
      rgb[dstIdx + 1] = data[srcIdx + 1];
      rgb[dstIdx + 2] = data[srcIdx + 2];

      flipped[flipIdx] = data[srcIdx];
      flipped[flipIdx + 1] = data[srcIdx + 1];
      flipped[flipIdx + 2] = data[srcIdx + 2];
    }
  }

  return [
    tf.tensor3d(rgb, [height, width, 3], "int32"),
    tf.tensor3d(flipped, [height, width, 3], "int32"),
  ];
}

async function main() {
  console.log("=== Step 1: Scaling Up Real Office Dataset (Target: 60 photos/class) ===");
  const classFiles = await downloadDataset();

  console.log("\n=== Step 2: Loading MobileNet v2 (alpha 1.0) ===");
  await tf.ready();
  const net = await mobilenet.load({ version: 2, alpha: 1.0 });
  const classifier = knnClassifier.create();

  const originalEmbeddings = [];
  const classes = Object.keys(CLASS_QUERIES);
  let totalEmbeddings = 0;

  console.log("\n=== Step 3: Extracting Deep Embeddings (with 2x Horizontal Mirror Augmentation) ===");
  for (const label of classes) {
    const files = classFiles[label] || [];
    let count = 0;
    for (const filePath of files) {
      try {
        const [origTensor, flipTensor] = decodeJpegToRgbTensorsWithFlip(filePath);

        const actOrig = net.infer(origTensor, true);
        const flatOrig = Array.from(actOrig.dataSync());
        originalEmbeddings.push({ label, data: flatOrig, shape: actOrig.shape });
        classifier.addExample(actOrig, label);
        actOrig.dispose();
        origTensor.dispose();

        const actFlip = net.infer(flipTensor, true);
        classifier.addExample(actFlip, label);
        actFlip.dispose();
        flipTensor.dispose();

        count += 2;
        totalEmbeddings += 2;
      } catch {
        // skip corrupted image
      }
    }
    console.log(`[Trained] ${label}: ${files.length} photos -> ${count} augmented embeddings`);
  }

  // Honest Leave-One-Out Cross-Validation on non-augmented originals (k=7)
  let correct = 0;
  for (let i = 0; i < originalEmbeddings.length; i++) {
    const evalKnn = knnClassifier.create();
    for (let j = 0; j < originalEmbeddings.length; j++) {
      if (i === j) continue;
      const t = tf.tensor(originalEmbeddings[j].data, originalEmbeddings[j].shape);
      evalKnn.addExample(t, originalEmbeddings[j].label);
      t.dispose();
    }
    const query = tf.tensor(originalEmbeddings[i].data, originalEmbeddings[i].shape);
    const pred = await evalKnn.predictClass(query, 7);
    if (pred.label === originalEmbeddings[i].label) correct++;
    query.dispose();
    evalKnn.dispose();
  }
  const accuracy =
    originalEmbeddings.length > 0
      ? ((correct / originalEmbeddings.length) * 100).toFixed(1)
      : "0.0";
  console.log(
    `\n=== Validation Result (Original Photos Only): ${correct}/${originalEmbeddings.length} (${accuracy}% LOO accuracy) ===`
  );

  // Compact serialization (round to 4 decimal places for fast browser load)
  const dataset = classifier.getClassifierDataset();
  const serializedDataset = {};
  for (const [label, tensor] of Object.entries(dataset)) {
    const rawArr = Array.from(tensor.dataSync());
    serializedDataset[label] = {
      shape: tensor.shape,
      data: rawArr.map((v) => Math.round(v * 10000) / 10000),
    };
  }

  fs.mkdirSync(path.dirname(OUTPUT_MODEL_PATH), { recursive: true });
  fs.writeFileSync(
    OUTPUT_MODEL_PATH,
    JSON.stringify({
      version: 2,
      savedAt: new Date().toISOString(),
      source: "Wikimedia Commons Large Office Dataset (2x Mirror Augmented MobileNet v2)",
      validationAccuracyPct: Number(accuracy),
      totalPhotos: originalEmbeddings.length,
      totalSamples: totalEmbeddings,
      classes,
      dataset: serializedDataset,
    })
  );

  const sizeMb = (fs.statSync(OUTPUT_MODEL_PATH).size / (1024 * 1024)).toFixed(2);
  console.log(`\nSaved v2 large pre-trained model (${totalEmbeddings} embeddings, ${sizeMb} MB) to: ${OUTPUT_MODEL_PATH}`);
}

main().catch((err) => {
  console.error("Training failed:", err);
  process.exit(1);
});
