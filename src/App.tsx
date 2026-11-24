import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import {
  Camera,
  CameraOff,
  Upload,
  Download,
  Sparkles,
  Brain,
  Activity,
  Plus,
  Trash2,
  Save,
  FolderUp,
  CheckCircle2,
} from "lucide-react";
import {
  ACTIVITY_META,
  classifyOfficeFrame,
  drawDetectionsOverlay,
  FrameAnalysis,
  RawDetection,
} from "./lib/activityClassifier";
import {
  CustomActivityModel,
  CustomPrediction,
  SerializedCustomModel,
} from "./lib/customActivityModel";
import { analyzeFrameWithGemini } from "./lib/geminiVision";

interface ActivityLogItem {
  id: string;
  time: string;
  source: "CUSTOM AI" | "OBJECT AI";
  label: string;
  confidence: number;
}

const DEFAULT_CLASSES = [
  "Working at PC",
  "Using Phone",
  "Reading a Book",
  "On Break / Drinking",
  "Idle / Away",
];

function colorForClass(label: string): string {
  const l = label.toLowerCase();
  if (l.includes("phone")) return "#ef4444";
  if (l.includes("work") || l.includes("pc") || l.includes("laptop")) return "#22c55e";
  if (l.includes("book") || l.includes("read")) return "#38bdf8";
  if (l.includes("break") || l.includes("drink") || l.includes("coffee")) return "#f59e0b";
  if (l.includes("idle") || l.includes("sleep") || l.includes("away")) return "#94a3b8";
  return "#a855f7";
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cocoRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const customModelRef = useRef<CustomActivityModel>(new CustomActivityModel());
  const rafRef = useRef<number | null>(null);
  const lastLoggedLabelRef = useRef<string>("");

  const [modelStatus, setModelStatus] = useState<"loading" | "ready" | "error">("loading");
  const [cameraActive, setCameraActive] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [showObjectBoxes, setShowObjectBoxes] = useState(true);
  const [fps, setFps] = useState(0);

  // Custom Trainable Model State
  const [classes, setClasses] = useState<string[]>(DEFAULT_CLASSES);
  const [newClassName, setNewClassName] = useState("");
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [recordingClass, setRecordingClass] = useState<string | null>(null);
  const [customPred, setCustomPred] = useState<CustomPrediction | null>(null);

  // COCO-SSD Object State
  const [analysis, setAnalysis] = useState<FrameAnalysis>({
    people: [],
    objects: [],
    counts: { total: 0, working: 0, onPhone: 0, readingOrBreak: 0, idle: 0 },
  });

  const [logs, setLogs] = useState<ActivityLogItem[]>([]);
  const [toast, setToast] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  // Optional Gemini Vision
  const [geminiKey, setGeminiKey] = useState("");
  const [geminiLoading, setGeminiLoading] = useState(false);
  const [geminiResult, setGeminiResult] = useState("");

  const notify = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(""), 4000);
  };

  const loadPretrainedFromPublic = async (silent = false) => {
    try {
      const res = await fetch("/pretrained-office-model.json");
      if (!res.ok) return false;
      const parsed = (await res.json()) as SerializedCustomModel;
      const loadedClasses = customModelRef.current.deserialize(parsed);
      setClasses(loadedClasses);
      setClassCounts({ ...customModelRef.current.getClassCounts() });
      if (!silent) {
        notify(`Loaded pre-trained Office AI model (${customModelRef.current.getTotalExamples()} real photos)!`);
      }
      return true;
    } catch {
      return false;
    }
  };

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await tf.ready();
        // 1. Load local MobileNet v2 (< 0.5s from localhost!) + pre-trained dataset immediately
        await customModelRef.current.load();
        const savedClasses = customModelRef.current.loadFromLocalStorage();
        if (mounted && savedClasses && savedClasses.length > 0 && customModelRef.current.getTotalExamples() > 0) {
          setClasses(savedClasses);
          setClassCounts({ ...customModelRef.current.getClassCounts() });
          notify("Restored your saved Custom AI Model!");
        } else if (mounted) {
          await loadPretrainedFromPublic(false);
        }
        if (mounted) {
          setModelStatus("ready");
        }
        // 2. Load COCO-SSD object boxes non-blocking in the background
        cocoSsd
          .load({ base: "lite_mobilenet_v2" })
          .then((coco) => {
            if (mounted) cocoRef.current = coco;
          })
          .catch(() => {
            // Custom AI works 100% independently even if COCO-SSD CDN is slow
          });
      } catch (e) {
        console.error(e);
        if (mounted) {
          setModelStatus("error");
          setErrorMsg("Failed to load local MobileNet v2 model.");
        }
      }
    })();

    refreshCameraList();
    return () => {
      mounted = false;
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshCameraList = async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all.filter((d) => d.kind === "videoinput");
      setDevices(cams);
      if (!selectedDeviceId && cams.length > 0) {
        setSelectedDeviceId(cams[0].deviceId);
      }
    } catch {
      // ignore
    }
  };

  const logActivityTransition = (
    source: "CUSTOM AI" | "OBJECT AI",
    label: string,
    confidence: number
  ) => {
    if (confidence < 0.65 || lastLoggedLabelRef.current === label) return;
    lastLoggedLabelRef.current = label;
    setLogs((prev) =>
      [
        {
          id: `${Date.now()}`,
          time: new Date().toLocaleTimeString(),
          source,
          label,
          confidence,
        },
        ...prev,
      ].slice(0, 60)
    );
  };

  const drawCustomHudBanner = (
    ctx: CanvasRenderingContext2D,
    pred: CustomPrediction | null
  ) => {
    if (!pred) return;
    const color = colorForClass(pred.topLabel);
    const text = `MY CUSTOM AI: ${pred.topLabel.toUpperCase()} (${Math.round(
      pred.topConfidence * 100
    )}%)`;
    ctx.font = "700 18px system-ui, sans-serif";
    const tw = ctx.measureText(text).width + 26;
    ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
    ctx.fillRect(14, 14, tw, 40);
    ctx.fillStyle = color;
    ctx.fillRect(14, 14, 6, 40);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, 28, 40);
  };

  const detectLoop = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(detectLoop);
      return;
    }

    const t0 = performance.now();
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Run Object Detection if enabled & loaded
    if (showObjectBoxes && cocoRef.current) {
      const preds = (await cocoRef.current.detect(video)) as RawDetection[];
      const result = classifyOfficeFrame(preds, 0.35);
      setAnalysis(result);
      if (ctx) {
        drawDetectionsOverlay(ctx, canvas.width, canvas.height, result);
      }
      if (
        customModelRef.current.getTotalExamples() === 0 &&
        result.people.length > 0
      ) {
        const p = result.people[0];
        logActivityTransition(
          "OBJECT AI",
          ACTIVITY_META[p.activity].label,
          p.confidence
        );
      }
    }

    // 2. Run User's Custom Trained AI Model (Primary!)
    if (
      customModelRef.current.isLoaded() &&
      customModelRef.current.getTotalExamples() > 0
    ) {
      const pred = await customModelRef.current.predict(video, 10);
      setCustomPred(pred);
      if (ctx && pred) {
        drawCustomHudBanner(ctx, pred);
      }
      if (pred) {
        logActivityTransition("CUSTOM AI", pred.topLabel, pred.topConfidence);
      }
    }

    const dt = performance.now() - t0;
    setFps(dt > 0 ? Math.min(60, Math.round(1000 / dt)) : 0);
    rafRef.current = requestAnimationFrame(detectLoop);
  };

  const startCamera = async (overrideDeviceId?: string) => {
    setErrorMsg("");
    setUploadedImageUrl(null);
    stopCamera();

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMsg("Camera API unavailable. Open http://localhost:5173 in Chrome or Edge.");
      return;
    }

    const deviceId = overrideDeviceId ?? selectedDeviceId;
    const tries: MediaStreamConstraints[] = [
      deviceId
        ? { video: { deviceId: { exact: deviceId } }, audio: false }
        : { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: true, audio: false },
    ];

    let stream: MediaStream | null = null;
    let lastErr: unknown = null;
    for (const c of tries) {
      try {
        stream = await navigator.mediaDevices.getUserMedia(c);
        break;
      } catch (e) {
        lastErr = e;
      }
    }

    if (!stream) {
      const e = lastErr as { name?: string; message?: string };
      setErrorMsg(`[${e?.name || "CameraError"}] ${e?.message || "Could not open camera."}`);
      return;
    }

    await refreshCameraList();
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch {
        // ignore
      }
      setCameraActive(true);
      rafRef.current = requestAnimationFrame(detectLoop);
    }
  };

  const stopCamera = () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const video = videoRef.current;
    if (video && video.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    setCameraActive(false);
    setFps(0);
  };

  const getActiveSource = (): HTMLVideoElement | HTMLImageElement | null => {
    if (cameraActive && videoRef.current && videoRef.current.readyState >= 2) {
      return videoRef.current;
    }
    if (uploadedImageUrl && imageRef.current) {
      return imageRef.current;
    }
    return null;
  };

  const trainSingleFrame = (label: string) => {
    const src = getActiveSource();
    if (!src) {
      setErrorMsg("Start the camera or upload a photo first to train your model.");
      return;
    }
    if (!customModelRef.current.isLoaded()) {
      setErrorMsg("AI feature extractor is still loading—wait 2 seconds.");
      return;
    }
    setErrorMsg("");
    customModelRef.current.addExample(src, label);
    setClassCounts({ ...customModelRef.current.getClassCounts() });
  };

  const trainBurstFrames = async (label: string, count = 15) => {
    const src = getActiveSource();
    if (!src) {
      setErrorMsg("Start the camera first, hold the pose/object, and click Burst +15.");
      return;
    }
    if (!customModelRef.current.isLoaded()) {
      setErrorMsg("AI feature extractor is still loading—wait 2 seconds.");
      return;
    }
    setErrorMsg("");
    setRecordingClass(label);
    for (let i = 0; i < count; i++) {
      const current = getActiveSource();
      if (!current) break;
      customModelRef.current.addExample(current, label);
      setClassCounts({ ...customModelRef.current.getClassCounts() });
      await new Promise((r) => setTimeout(r, 90));
    }
    setRecordingClass(null);
    customModelRef.current.saveToLocalStorage(classes);
    notify(`Trained +${count} frames for "${label}" and auto-saved model!`);
  };

  const handleAddCustomClass = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newClassName.trim();
    if (!trimmed) return;
    if (!classes.includes(trimmed)) {
      setClasses((prev) => [...prev, trimmed]);
    }
    setNewClassName("");
    notify(`Added custom class "${trimmed}" — click Burst +15 while showing it to camera!`);
  };

  const handleClearClass = (label: string) => {
    customModelRef.current.clearClass(label);
    setClassCounts({ ...customModelRef.current.getClassCounts() });
    customModelRef.current.saveToLocalStorage(classes);
  };

  const handleSaveModel = () => {
    customModelRef.current.saveToLocalStorage(classes);
    notify("Custom AI model saved in browser storage!");
  };

  const handleExportModel = () => {
    const data = customModelRef.current.serialize(classes);
    const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `my-office-ai-model-${Date.now()}.json`;
    a.click();
  };

  const handleImportModel = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as SerializedCustomModel;
      const loadedClasses = customModelRef.current.deserialize(parsed);
      setClasses(loadedClasses);
      setClassCounts({ ...customModelRef.current.getClassCounts() });
      customModelRef.current.saveToLocalStorage(loadedClasses);
      notify(`Imported custom AI model with ${loadedClasses.length} classes!`);
    } catch {
      setErrorMsg("Invalid model JSON file.");
    }
  };

  const trainFromUploadedPhotos = async (
    label: string,
    files: FileList | null
  ) => {
    if (!files || files.length === 0) return;
    if (!customModelRef.current.isLoaded()) {
      setErrorMsg("Local MobileNet v2 is still loading—wait 1 second and retry.");
      return;
    }
    setErrorMsg("");
    let added = 0;
    for (const file of Array.from(files)) {
      const url = URL.createObjectURL(file);
      try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const el = new Image();
          el.onload = () => resolve(el);
          el.onerror = reject;
          el.src = url;
        });
        customModelRef.current.addExample(img, label);
        added++;
      } catch {
        // skip unreadable file
      } finally {
        URL.revokeObjectURL(url);
      }
    }
    if (added > 0) {
      setClassCounts({ ...customModelRef.current.getClassCounts() });
      customModelRef.current.saveToLocalStorage(classes);
      notify(`Trained +${added} uploaded photo${added > 1 ? "s" : ""} for "${label}"!`);
      if (uploadedImageUrl) {
        analyzeUploadedImage();
      }
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    stopCamera();
    setErrorMsg("");
    setUploadedImageUrl(URL.createObjectURL(file));
    e.target.value = "";
  };

  const handleDropPhoto = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    stopCamera();
    setErrorMsg("");
    setUploadedImageUrl(URL.createObjectURL(file));
  };

  const analyzeUploadedImage = async () => {
    const img = imageRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showObjectBoxes && cocoRef.current) {
      const preds = (await cocoRef.current.detect(img)) as RawDetection[];
      const result = classifyOfficeFrame(preds, 0.35);
      setAnalysis(result);
      if (ctx) drawDetectionsOverlay(ctx, canvas.width, canvas.height, result);
    }

    if (customModelRef.current.getTotalExamples() > 0) {
      const pred = await customModelRef.current.predict(img, 10);
      setCustomPred(pred);
      if (ctx && pred) drawCustomHudBanner(ctx, pred);
    }
  };

  const captureFrameAsJpeg = (): string | null => {
    const src = getActiveSource();
    if (!src) return null;
    const tmp = document.createElement("canvas");
    if (src instanceof HTMLVideoElement) {
      tmp.width = src.videoWidth;
      tmp.height = src.videoHeight;
    } else {
      tmp.width = src.naturalWidth;
      tmp.height = src.naturalHeight;
    }
    tmp.getContext("2d")?.drawImage(src, 0, 0);
    return tmp.toDataURL("image/jpeg", 0.85);
  };

  const runGeminiDeepAnalysis = async () => {
    if (!geminiKey.trim()) {
      setErrorMsg("Paste your Gemini API key first.");
      return;
    }
    const jpeg = captureFrameAsJpeg();
    if (!jpeg) {
      setErrorMsg("Start the camera or upload a photo first.");
      return;
    }
    setErrorMsg("");
    setGeminiLoading(true);
    try {
      setGeminiResult(await analyzeFrameWithGemini(geminiKey, jpeg));
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Gemini Vision failed");
    } finally {
      setGeminiLoading(false);
    }
  };

  const exportLogsCsv = () => {
    const header = "Time,Source,Activity,Confidence\n";
    const rows = logs
      .map((l) => `"${l.time}","${l.source}","${l.label}","${Math.round(l.confidence * 100)}%"`)
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "office-activity-log.csv";
    a.click();
  };

  const totalTrainedSamples = Object.values(classCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      <header className="h-14 border-b border-slate-800 bg-slate-900/90 px-6 flex items-center justify-between sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-indigo-400" />
          <span className="font-bold tracking-tight text-lg">OfficeVision AI</span>
          <span className="text-xs px-2 py-0.5 rounded bg-emerald-600/25 text-emerald-300 border border-emerald-500/40">
            In-App Trainable AI Model
          </span>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <span className="text-slate-400 hidden sm:inline">
            Trained Samples: <strong className="text-white">{totalTrainedSamples}</strong>
          </span>
          <span
            className={`px-2.5 py-1 rounded-full font-medium border ${
              modelStatus === "ready"
                ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
                : "bg-amber-500/15 text-amber-300 border-amber-500/40"
            }`}
          >
            {modelStatus === "ready" ? `AI Ready · ${fps} FPS` : "Loading Neural Networks..."}
          </span>
        </div>
      </header>

      {toast && (
        <div className="fixed bottom-5 right-5 z-50 flex items-center gap-2 bg-slate-900 text-white text-xs px-4 py-3 rounded-lg shadow-lg border border-emerald-500">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{toast}</span>
        </div>
      )}

      <main className="flex-1 p-5 max-w-[1480px] w-full mx-auto space-y-5">
        {errorMsg && (
          <div className="bg-red-500/15 border border-red-500/40 text-red-200 px-4 py-2.5 rounded-lg text-xs">
            {errorMsg}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left 7 cols: Live Camera Viewport & Controls */}
          <div className="lg:col-span-7 space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                {!cameraActive ? (
                  <button
                    onClick={() => startCamera()}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500"
                  >
                    <Camera className="w-4 h-4" /> Start Live Camera
                  </button>
                ) : (
                  <button
                    onClick={stopCamera}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-red-600 hover:bg-red-500"
                  >
                    <CameraOff className="w-4 h-4" /> Stop Camera
                  </button>
                )}

                {devices.length > 0 && (
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => {
                      setSelectedDeviceId(e.target.value);
                      if (cameraActive) startCamera(e.target.value);
                    }}
                    className="text-xs bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-2 max-w-[180px]"
                  >
                    {devices.map((d, i) => (
                      <option key={d.deviceId || i} value={d.deviceId}>
                        {d.label || `Camera #${i + 1}`}
                      </option>
                    ))}
                  </select>
                )}

                <label className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold bg-emerald-700/80 hover:bg-emerald-600 text-white cursor-pointer">
                  <Upload className="w-3.5 h-3.5" /> Upload Photo
                  <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
                </label>
              </div>

              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showObjectBoxes}
                  onChange={(e) => setShowObjectBoxes(e.target.checked)}
                  className="accent-indigo-500"
                />
                Object Boxes
              </label>
            </div>

            {/* Video / Canvas HUD (also supports Drag & Drop Photo) */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDropPhoto}
              className="relative bg-black rounded-xl border border-slate-800 overflow-hidden aspect-video flex items-center justify-center"
            >
              {!cameraActive && !uploadedImageUrl && (
                <div className="text-center space-y-2 p-8">
                  <Brain className="w-10 h-10 text-indigo-400 mx-auto" />
                  <p className="text-sm text-slate-300 font-medium">
                    Click <strong>Start Live Camera</strong>, click <strong>Upload Photo</strong>, or <strong>Drag &amp; Drop</strong> an office photo here to identify activities!
                  </p>
                </div>
              )}

              <video
                ref={videoRef}
                muted
                playsInline
                className={`w-full h-full object-contain ${cameraActive ? "block" : "hidden"}`}
              />

              {uploadedImageUrl && (
                <img
                  ref={imageRef}
                  src={uploadedImageUrl}
                  alt="Uploaded scene"
                  onLoad={analyzeUploadedImage}
                  className="w-full h-full object-contain"
                />
              )}

              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              />
            </div>

            {/* Live Prediction Summary Banner */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Live Prediction (Your Custom AI Model)
                </div>
                <div
                  className="text-2xl font-bold mt-0.5"
                  style={{
                    color: customPred ? colorForClass(customPred.topLabel) : "#94a3b8",
                  }}
                >
                  {customPred
                    ? `${customPred.topLabel} (${Math.round(customPred.topConfidence * 100)}%)`
                    : "Not Trained Yet — Click 'Burst +15' on at least 2 classes ->"}
                </div>
              </div>
              <div className="text-xs text-slate-400">
                People in Frame: <strong className="text-white">{analysis.counts.total}</strong>
              </div>
            </div>

            {/* Live Activity Log */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2.5">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold">Live Activity Timeline</h2>
                {logs.length > 0 && (
                  <button
                    onClick={exportLogsCsv}
                    className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300"
                  >
                    <Download className="w-3.5 h-3.5" /> Export CSV
                  </button>
                )}
              </div>
              <div className="space-y-1.5 max-h-48 overflow-auto text-xs">
                {logs.length === 0 ? (
                  <p className="text-slate-500">Activity transitions (&ge;65% confidence) log here automatically.</p>
                ) : (
                  logs.map((log) => (
                    <div
                      key={log.id}
                      className="flex items-center justify-between bg-slate-950 border border-slate-800 rounded px-3 py-1.5"
                    >
                      <div>
                        <span className="font-mono text-slate-500 mr-2">{log.time}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 mr-2">
                          {log.source}
                        </span>
                        <span className="font-semibold">{log.label}</span>
                      </div>
                      <span className="font-mono text-emerald-400">
                        {Math.round(log.confidence * 100)}%
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right 5 cols: In-App Custom AI Model Trainer + Gemini Vision */}
          <div className="lg:col-span-5 space-y-4">
            <div className="bg-slate-900 border border-indigo-500/40 rounded-xl p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-indigo-400" />
                  <div>
                    <h2 className="text-sm font-bold text-white">
                      Train Your Own AI Model (Live)
                    </h2>
                    <p className="text-[11px] text-slate-400">
                      Show each activity to your camera &amp; click <strong>Burst +15</strong>. Train at least 2 classes!
                    </p>
                  </div>
                </div>
              </div>

              {/* Add ANY Custom Class */}
              <form onSubmit={handleAddCustomClass} className="flex gap-2">
                <input
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                  placeholder="Add anything (e.g., Sleeping, Talking, Writing)..."
                  className="flex-1 text-xs bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500"
                />
                <button
                  type="submit"
                  className="inline-flex items-center gap-1 px-3 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500"
                >
                  <Plus className="w-3.5 h-3.5" /> Add Class
                </button>
              </form>

              {/* Trainable Class Cards with Live Confidence Bars */}
              <div className="space-y-2.5 max-h-[380px] overflow-auto pr-1">
                {classes.map((label) => {
                  const count = classCounts[label] || 0;
                  const score = customPred?.confidences[label] || 0;
                  const pct = Math.round(score * 100);
                  const barColor = colorForClass(label);
                  const isRec = recordingClass === label;

                  return (
                    <div
                      key={label}
                      className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <div className="text-xs font-bold text-slate-100">{label}</div>
                          <div className="text-[11px] text-slate-400">
                            {count} trained frames ·{" "}
                            <span style={{ color: barColor }} className="font-mono font-semibold">
                              {pct}%
                            </span>
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5">
                          <label
                            title="Upload 1 or more photos to train this class"
                            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-semibold rounded bg-indigo-900/60 hover:bg-indigo-800 border border-indigo-500/40 text-indigo-200 cursor-pointer"
                          >
                            <Upload className="w-3 h-3" /> Photos
                            <input
                              type="file"
                              accept="image/*"
                              multiple
                              onChange={(e) => {
                                trainFromUploadedPhotos(label, e.target.files);
                                e.target.value = "";
                              }}
                              className="hidden"
                            />
                          </label>
                          <button
                            onClick={() => trainSingleFrame(label)}
                            disabled={recordingClass !== null}
                            className="px-2 py-1 text-[11px] font-semibold rounded bg-slate-800 hover:bg-slate-700 border border-slate-700"
                          >
                            +1
                          </button>
                          <button
                            onClick={() => trainBurstFrames(label, 15)}
                            disabled={recordingClass !== null}
                            className={`px-2.5 py-1 text-[11px] font-semibold rounded ${
                              isRec
                                ? "bg-red-600 text-white animate-pulse"
                                : "bg-emerald-600 hover:bg-emerald-500 text-white"
                            }`}
                          >
                            {isRec ? "Recording..." : "Burst +15"}
                          </button>
                          {count > 0 && (
                            <button
                              onClick={() => handleClearClass(label)}
                              title="Clear class samples"
                              className="p-1 text-slate-500 hover:text-red-400"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </div>

                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full transition-all duration-150"
                          style={{ width: `${pct}%`, backgroundColor: barColor }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Save / Reset / Export / Import Model Controls */}
              <div className="grid grid-cols-4 gap-2 pt-2 border-t border-slate-800">
                <button
                  onClick={handleSaveModel}
                  className="inline-flex items-center justify-center gap-1 py-2 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700"
                >
                  <Save className="w-3.5 h-3.5" /> Save
                </button>
                <button
                  onClick={() => loadPretrainedFromPublic(false)}
                  title="Reload bundled pre-trained dataset model"
                  className="inline-flex items-center justify-center gap-1 py-2 rounded-lg text-xs font-medium bg-indigo-900/50 hover:bg-indigo-800/60 border border-indigo-500/40 text-indigo-200"
                >
                  Pre-Trained
                </button>
                <button
                  onClick={handleExportModel}
                  disabled={totalTrainedSamples === 0}
                  className="inline-flex items-center justify-center gap-1 py-2 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 disabled:opacity-40"
                >
                  <Download className="w-3.5 h-3.5" /> Export
                </button>
                <label className="inline-flex items-center justify-center gap-1 py-2 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 cursor-pointer">
                  <FolderUp className="w-3.5 h-3.5" /> Import
                  <input type="file" accept=".json" onChange={handleImportModel} className="hidden" />
                </label>
              </div>
            </div>

            {/* Optional Gemini Vision Deep Scene Analysis */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-semibold text-indigo-300">
                <Sparkles className="w-4 h-4" /> Gemini Vision Deep Scene Analysis (Optional)
              </div>
              <div className="flex gap-2">
                <input
                  type="password"
                  placeholder="Gemini API Key"
                  value={geminiKey}
                  onChange={(e) => setGeminiKey(e.target.value)}
                  className="flex-1 text-xs bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5"
                />
                <button
                  onClick={runGeminiDeepAnalysis}
                  disabled={geminiLoading}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
                >
                  {geminiLoading ? "..." : "Analyze"}
                </button>
              </div>
              {geminiResult && (
                <pre className="text-xs bg-slate-950 border border-slate-800 rounded-lg p-3 whitespace-pre-wrap text-slate-200 max-h-40 overflow-auto">
                  {geminiResult}
                </pre>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
