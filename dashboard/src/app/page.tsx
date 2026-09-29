"use client";

import React, { useState, useEffect } from "react";
import { Spotlight } from "@/components/ui/spotlight";
import { FileUpload } from "@/components/ui/file-upload";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { Check, Database, Gauge, Layers, SlidersHorizontal, Target, TriangleAlert, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

type Metrics = {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  test_samples: number;
  params?: string;
  inference?: string;
  size?: string;
  split_counts?: { train: number; val: number; test: number };
  // Present when trained over several seeds: metric values are means across seeds
  accuracy_std?: number;
  f1_score_std?: number;
  precision_std?: number;
  seeds?: { seed: number; accuracy: number }[];
  selected_seed?: number;
  confusion_matrix?: number[][];
  recall_per_class?: Record<string, number>;
  classes?: string[];
  config?: { phase1_epochs: number; phase2_epochs: number };
};

type TrainingHistory = {
  train_loss: number[];
  train_acc: number[];
  val_loss: number[];
  val_acc: number[];
};

type Prediction = {
  class: string;
  confidence: number;
  probabilities: Record<string, number>;
  error?: string;
};

type ChartPoint = Record<string, number>;

// Monochrome identities: DenseNet = white, MobileNet = gray; val lines are dashed
const M1_COLOR = "#ffffff";
const M2_COLOR = "#a3a3a3";
const SPLIT_COLORS = ["#ffffff", "#a3a3a3", "#525252"];

// "95.0 ± 1.2%" when a std is available, otherwise "95.0%"
function formatPct(m: Metrics | null, key: "accuracy" | "f1_score" | "precision") {
  if (!m) return "…";
  const std = m[`${key}_std`];
  const mean = (m[key] * 100).toFixed(1);
  return std === undefined ? `${mean}%` : `${mean} ± ${(std * 100).toFixed(1)}%`;
}

const num = (s?: string) => (s ? parseFloat(s) : NaN);
const meanPct = (m: Metrics | null) => (m ? `${(m.accuracy * 100).toFixed(1)}%` : "…");

function SectionTitle({ eyebrow, title, description }: { eyebrow: string; title: string; description?: string }) {
  return (
    <div className="mb-8 border-l-4 border-white pl-4">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-neutral-500">{eyebrow}</p>
      <h2 className="mt-1 text-3xl font-bold text-white">{title}</h2>
      {description && <p className="mt-2 max-w-2xl text-sm text-neutral-400">{description}</p>}
    </div>
  );
}

function StatTile({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail: string }) {
  return (
    <div className="border border-neutral-800 bg-black/60 p-5 backdrop-blur-sm">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-neutral-500">
        {icon}
        {label}
      </div>
      <p className="mt-3 text-3xl font-bold tabular-nums text-white">{value}</p>
      <p className="mt-1 text-xs text-neutral-400">{detail}</p>
    </div>
  );
}

function ProbabilityBar({ label, value, strong }: { label: string; value: number; strong: boolean }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className={strong ? "text-white" : "text-neutral-500"}>{label}</span>
        <span className={cn("tabular-nums", strong ? "text-white" : "text-neutral-500")}>{value.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full bg-neutral-800">
        <div
          className={cn("h-2 transition-all duration-700", strong ? "bg-white" : "bg-neutral-600")}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function ModelResult({
  name,
  tag,
  metrics,
  prediction,
  isProcessing,
}: {
  name: string;
  tag: string;
  metrics: Metrics | null;
  prediction: Prediction | null;
  isProcessing: boolean;
}) {
  return (
    <Card className="flex h-full flex-col bg-neutral-950">
      <CardHeader className="pb-3">
        <p className="font-mono text-xs uppercase tracking-wider text-neutral-500">{tag}</p>
        <CardTitle className="text-xl text-white">{name}</CardTitle>
        <CardDescription className="text-neutral-500">
          Test acc {formatPct(metrics, "accuracy")} · {metrics?.inference ?? "…"} / image
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col justify-center">
        {isProcessing ? (
          <div className="animate-pulse space-y-3">
            <div className="h-8 w-1/2 bg-neutral-800" />
            <div className="h-2 w-full bg-neutral-800" />
            <div className="h-2 w-full bg-neutral-800" />
          </div>
        ) : prediction?.error ? (
          <p className="text-sm text-neutral-400">{prediction.error}</p>
        ) : prediction ? (
          <div className="space-y-4">
            <div>
              <p className="text-xs uppercase tracking-wider text-neutral-500">Prediction</p>
              <p className="text-4xl font-bold text-white">{prediction.class}</p>
              <p className="text-sm tabular-nums text-neutral-400">{prediction.confidence.toFixed(1)}% confidence</p>
            </div>
            <div className="space-y-3">
              {Object.entries(prediction.probabilities).map(([label, value]) => (
                <ProbabilityBar key={label} label={label} value={value} strong={label === prediction.class} />
              ))}
            </div>
          </div>
        ) : (
          <p className="py-6 text-sm italic text-neutral-500">Waiting for an image…</p>
        )}
      </CardContent>
    </Card>
  );
}

function ConfusionGrid({ metrics }: { metrics: Metrics | null }) {
  const cm = metrics?.confusion_matrix;
  const classes = metrics?.classes ?? ["Salmon", "Trout"];
  if (!cm) return <div className="h-48 animate-pulse bg-neutral-900" />;
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-[2px] text-center text-sm">
      <div />
      {classes.map((c) => (
        <div key={c} className="pb-1 text-xs text-neutral-500">
          Predicted {c}
        </div>
      ))}
      {cm.map((row, r) => {
        const total = row.reduce((a, b) => a + b, 0);
        return (
          <React.Fragment key={r}>
            <div className="flex items-center pr-3 text-xs text-neutral-500">Actual {classes[r]}</div>
            {row.map((count, c) => {
              const share = count / total;
              return (
                <div
                  key={c}
                  title={`${count} of ${total} ${classes[r]} images predicted as ${classes[c]}`}
                  className={cn("py-5", share > 0.5 ? "text-black" : "text-white")}
                  style={{ backgroundColor: `rgba(255,255,255,${0.08 + share * 0.85})` }}
                >
                  <p className="text-xl font-bold tabular-nums">{count}</p>
                  <p className="text-xs tabular-nums opacity-70">{(share * 100).toFixed(0)}%</p>
                </div>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [metricsM1, setMetricsM1] = useState<Metrics | null>(null);
  const [metricsM2, setMetricsM2] = useState<Metrics | null>(null);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [chartMetric, setChartMetric] = useState<"loss" | "acc">("acc");
  const [predictionM1, setPredictionM1] = useState<Prediction | null>(null);
  const [predictionM2, setPredictionM2] = useState<Prediction | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [m1, m2, h1, h2] = await Promise.all(
          ["metrics.json", "metrics_model2.json", "training_history.json", "training_history_model2.json"].map((f) =>
            fetch(`/data/${f}`).then((r) => r.json())
          )
        );
        setMetricsM1(m1);
        setMetricsM2(m2);

        const hist1 = h1 as TrainingHistory;
        const hist2 = h2 as TrainingHistory;
        const epochs = Math.max(hist1.train_loss.length, hist2.train_loss.length);
        setChartData(
          Array.from({ length: epochs }, (_, i) => ({
            epoch: i + 1,
            loss_m1: hist1.train_loss[i],
            val_loss_m1: hist1.val_loss[i],
            loss_m2: hist2.train_loss[i],
            val_loss_m2: hist2.val_loss[i],
            acc_m1: hist1.train_acc[i] * 100,
            val_acc_m1: hist1.val_acc[i] * 100,
            acc_m2: hist2.train_acc[i] * 100,
            val_acc_m2: hist2.val_acc[i] * 100,
          }))
        );
      } catch (error) {
        console.error("Failed to fetch dashboard data", error);
      }
    }
    fetchData();
  }, []);

  const handleFileUpload = async (files: File[]) => {
    if (!files[0]) return;
    setIsProcessing(true);
    setPredictionM1(null);
    setPredictionM2(null);
    setPreviewImage(URL.createObjectURL(files[0]));

    const formData = new FormData();
    formData.append("file", files[0]);
    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await response.json();
      setPredictionM1(data.model1 ?? { error: "No result" });
      setPredictionM2(data.model2 ?? { error: "No result" });
    } catch (error) {
      console.error("Prediction Error:", error);
      setPredictionM1({ error: "Request failed" } as Prediction);
      setPredictionM2({ error: "Request failed" } as Prediction);
    } finally {
      setIsProcessing(false);
    }
  };

  const bothPredicted = predictionM1?.class && predictionM2?.class;
  const agree = bothPredicted && predictionM1.class === predictionM2.class;

  const split = metricsM1?.split_counts;
  const splitData = split
    ? [
        { name: "Train", value: split.train },
        { name: "Validation", value: split.val },
        { name: "Test", value: split.test },
      ]
    : [];
  const totalImages = split ? split.train + split.val + split.test : 0;
  const speedup = num(metricsM1?.inference) / num(metricsM2?.inference);
  const phase1 = metricsM1?.config?.phase1_epochs ?? 15;
  const nSeeds = metricsM1?.seeds?.length ?? 1;

  // Row definitions for the comparison table; `better` says which direction wins
  const rows: { label: string; m1: string; m2: string; better?: "higher" | "lower"; v1?: number; v2?: number }[] = [
    { label: "Accuracy", m1: formatPct(metricsM1, "accuracy"), m2: formatPct(metricsM2, "accuracy"), better: "higher", v1: metricsM1?.accuracy, v2: metricsM2?.accuracy },
    { label: "F1 score", m1: formatPct(metricsM1, "f1_score"), m2: formatPct(metricsM2, "f1_score"), better: "higher", v1: metricsM1?.f1_score, v2: metricsM2?.f1_score },
    { label: "Precision", m1: formatPct(metricsM1, "precision"), m2: formatPct(metricsM2, "precision"), better: "higher", v1: metricsM1?.precision, v2: metricsM2?.precision },
    { label: "Latency", m1: metricsM1?.inference ?? "…", m2: metricsM2?.inference ?? "…", better: "lower", v1: num(metricsM1?.inference), v2: num(metricsM2?.inference) },
    { label: "Size", m1: metricsM1?.size ?? "…", m2: metricsM2?.size ?? "…", better: "lower", v1: num(metricsM1?.size), v2: num(metricsM2?.size) },
    { label: "Params", m1: metricsM1?.params ?? "…", m2: metricsM2?.params ?? "…", better: "lower", v1: num(metricsM1?.params), v2: num(metricsM2?.params) },
  ];
  const winner = (r: (typeof rows)[number]) => {
    if (r.v1 === undefined || r.v2 === undefined || isNaN(r.v1) || isNaN(r.v2) || r.v1 === r.v2) return 0;
    return (r.better === "higher") === r.v1 > r.v2 ? 1 : 2;
  };

  const suffix = chartMetric === "acc" ? "%" : "";
  const lines = [
    { key: `${chartMetric}_m1`, name: "DenseNet train", color: M1_COLOR, dashed: false },
    { key: `val_${chartMetric}_m1`, name: "DenseNet val", color: M1_COLOR, dashed: true },
    { key: `${chartMetric}_m2`, name: "MobileNet train", color: M2_COLOR, dashed: false },
    { key: `val_${chartMetric}_m2`, name: "MobileNet val", color: M2_COLOR, dashed: true },
  ];

  return (
    <div className="min-h-screen w-full bg-black/[0.96] antialiased bg-grid-white/[0.02] relative overflow-hidden text-neutral-200">
      
      {/* 1. Hero Section */}
      <div className="h-[40rem] w-full flex md:items-center md:justify-center bg-black/[0.96] antialiased bg-grid-white/[0.02] relative overflow-hidden">
        <Spotlight
          className="-top-40 left-0 md:left-60 md:-top-20"
          fill="white"
        />
        <div className="p-4 max-w-7xl mx-auto relative z-10 w-full pt-20 md:pt-0">
          <h1 className="text-4xl md:text-7xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 bg-opacity-50 pb-4">
            Salmon vs. Trout <br /> AI Classification
          </h1>
          <p className="mt-4 font-normal text-base text-neutral-300 max-w-lg text-center mx-auto">
            Advanced Deep Learning Comparison: DenseNet121 vs. MobileNetV2.
            Strict monochrome design system.
          </p>
        </div>
      </div>

      <div className="relative z-10 mx-auto -mt-32 max-w-7xl space-y-24 px-4 pb-24 sm:px-6 lg:px-8">
        {/* 2. Headline numbers */}
        <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatTile icon={<Target className="h-4 w-4" />} label="DenseNet121" value={meanPct(metricsM1)} detail={`test accuracy · ±${((metricsM1?.accuracy_std ?? 0) * 100).toFixed(1)} over ${nSeeds} seeds`} />
          <StatTile icon={<Target className="h-4 w-4" />} label="MobileNetV2" value={meanPct(metricsM2)} detail={`test accuracy · ±${((metricsM2?.accuracy_std ?? 0) * 100).toFixed(1)} over ${nSeeds} seeds`} />
          <StatTile icon={<Zap className="h-4 w-4" />} label="Speed" value={isNaN(speedup) ? "…" : `${speedup.toFixed(1)}×`} detail={`MobileNetV2 faster (${metricsM2?.inference ?? "…"} vs ${metricsM1?.inference ?? "…"})`} />
          <StatTile icon={<Database className="h-4 w-4" />} label="Dataset" value={totalImages ? totalImages.toLocaleString() : "…"} detail={`images · ${split?.test ?? "…"} held out for testing`} />
        </section>

        {/* 3. Try it */}
        <section>
          <SectionTitle
            eyebrow="01 · Try it"
            title="Interactive Model Comparison"
            description="Upload one photo and both models classify it side by side."
          />
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Upload Image</CardTitle>
                <CardDescription>Drag & drop a salmon or trout photo.</CardDescription>
              </CardHeader>
              <CardContent>
                <FileUpload onChange={handleFileUpload} previewUrl={previewImage} scanning={isProcessing} />
              </CardContent>
            </Card>

            <div className="flex flex-col gap-6 lg:col-span-3">
              <div className="grid flex-1 grid-cols-1 gap-6 sm:grid-cols-2">
                <ModelResult name="DenseNet121" tag="Model 1" metrics={metricsM1} prediction={predictionM1} isProcessing={isProcessing} />
                <ModelResult name="MobileNetV2" tag="Model 2" metrics={metricsM2} prediction={predictionM2} isProcessing={isProcessing} />
              </div>
              <div
                className={cn(
                  "flex items-center gap-3 border px-5 py-4 text-sm",
                  !bothPredicted && "border-neutral-800 text-neutral-500",
                  bothPredicted && agree && "border-white/40 text-white",
                  bothPredicted && !agree && "border-neutral-600 border-dashed text-neutral-300"
                )}
              >
                {bothPredicted ? (
                  agree ? (
                    <>
                      <Check className="h-4 w-4 shrink-0" /> Both models agree: <span className="font-bold">{predictionM1.class}</span>
                    </>
                  ) : (
                    <>
                      <TriangleAlert className="h-4 w-4 shrink-0" /> The models disagree. Around 40% of test-set trout are misread as salmon, so treat
                      a &quot;Salmon&quot; answer with care.
                    </>
                  )
                ) : (
                  <>
                    <Gauge className="h-4 w-4 shrink-0" /> Verdict appears here once both models have answered.
                  </>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* 4. Results */}
        <section>
          <SectionTitle
            eyebrow="02 · Results"
            title="Performance Metrics"
            description={`Both models share one training recipe; only the backbone differs. Test-set numbers are mean ± SD over ${nSeeds} seeds.`}
          />

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
                <div className="space-y-1.5">
                  <CardTitle>Learning Curve</CardTitle>
                  <CardDescription>Selected seed of each model. Dotted line = start of full fine-tuning.</CardDescription>
                </div>
                <div className="flex border border-neutral-800 text-xs" role="tablist">
                  {(["acc", "loss"] as const).map((m) => (
                    <button
                      key={m}
                      role="tab"
                      aria-selected={chartMetric === m}
                      onClick={() => setChartMetric(m)}
                      className={cn(
                        "px-3 py-1.5 transition",
                        chartMetric === m ? "bg-white text-black" : "text-neutral-400 hover:text-white"
                      )}
                    >
                      {m === "acc" ? "Accuracy" : "Loss"}
                    </button>
                  ))}
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-[320px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                      <XAxis dataKey="epoch" stroke="#525252" tick={{ fill: "#a3a3a3", fontSize: 12 }} tickLine={false} interval={4} />
                      <YAxis
                        stroke="#525252"
                        tick={{ fill: "#a3a3a3", fontSize: 12 }}
                        tickLine={false}
                        domain={chartMetric === "acc" ? [40, 90] : ["auto", "auto"]}
                        ticks={chartMetric === "acc" ? [40, 50, 60, 70, 80, 90] : undefined}
                        tickFormatter={(v: number) => (chartMetric === "acc" ? `${v.toFixed(0)}%` : v.toFixed(2))}
                      />
                      <ReferenceLine x={phase1 + 0.5} stroke="#525252" strokeDasharray="2 4" />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#0a0a0a", borderColor: "#333", color: "#fff", fontSize: 12 }}
                        labelFormatter={(e) => `Epoch ${e}`}
                        formatter={(v) => `${Number(v).toFixed(chartMetric === "acc" ? 1 : 4)}${suffix}`}
                      />
                      {lines.map((l) => (
                        <Line
                          key={l.key}
                          type="monotone"
                          dataKey={l.key}
                          name={l.name}
                          stroke={l.color}
                          strokeWidth={2}
                          strokeDasharray={l.dashed ? "5 5" : undefined}
                          dot={false}
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 text-xs text-neutral-400">
                  {lines.map((l) => (
                    <span key={l.key} className="flex items-center gap-2">
                      <svg width="24" height="4" aria-hidden>
                        <line x1="0" y1="2" x2="24" y2="2" stroke={l.color} strokeWidth="2" strokeDasharray={l.dashed ? "5 3" : undefined} />
                      </svg>
                      {l.name}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Comparison</CardTitle>
                <CardDescription>Better value in each row is bold.</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-neutral-800 hover:bg-transparent">
                      <TableHead className="text-neutral-500">Metric</TableHead>
                      <TableHead className="text-white">DenseNet</TableHead>
                      <TableHead className="text-neutral-400">MobileNet</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rows.map((r) => {
                      const w = winner(r);
                      return (
                        <TableRow key={r.label} className="border-neutral-800 hover:bg-neutral-900/50">
                          <TableCell className="text-neutral-400">{r.label}</TableCell>
                          <TableCell className={cn("whitespace-nowrap tabular-nums", w === 1 ? "font-bold text-white" : "text-neutral-400")}>{r.m1}</TableCell>
                          <TableCell className={cn("whitespace-nowrap tabular-nums", w === 2 ? "font-bold text-white" : "text-neutral-400")}>{r.m2}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
            {[
              { name: "DenseNet121", m: metricsM1 },
              { name: "MobileNetV2", m: metricsM2 },
            ].map(({ name, m }) => (
              <Card key={name}>
                <CardHeader>
                  <CardTitle>{name}: where it goes wrong</CardTitle>
                  <CardDescription>
                    Confusion matrix on {m?.test_samples ?? "…"} test images (seed {m?.selected_seed ?? "…"}). Row % = share of each actual class.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ConfusionGrid metrics={m} />
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* 5. Pipeline */}
        <section>
          <SectionTitle
            eyebrow="03 · Method"
            title="Data & Training Pipeline"
            description="Every setting below is shared by both models, so differences come from the architecture alone."
          />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <Card>
              <CardHeader>
                <Database className="h-6 w-6 text-neutral-300" />
                <CardTitle className="pt-2">Dataset split</CardTitle>
                <CardDescription>Stratified, fixed seed. Duplicate images are kept in the same split.</CardDescription>
              </CardHeader>
              <CardContent className="flex items-center gap-6">
                <div className="h-28 w-28 shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={splitData} dataKey="value" innerRadius={32} outerRadius={52} paddingAngle={3} stroke="none">
                        {splitData.map((entry, i) => (
                          <Cell key={entry.name} fill={SPLIT_COLORS[i]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ backgroundColor: "#0a0a0a", borderColor: "#333", fontSize: 12 }} itemStyle={{ color: "#fff" }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <ul className="space-y-2 text-sm">
                  {splitData.map((d, i) => (
                    <li key={d.name} className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5" style={{ backgroundColor: SPLIT_COLORS[i] }} />
                      <span className="text-neutral-400">{d.name}</span>
                      <span className="tabular-nums text-white">{d.value}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Layers className="h-6 w-6 text-neutral-300" />
                <CardTitle className="pt-2">Preprocessing</CardTitle>
                <CardDescription>Same transforms for training, testing and this demo.</CardDescription>
              </CardHeader>
              <CardContent>
                <ol className="space-y-2 text-sm">
                  {["Resize to 224 × 224", "CLAHE contrast enhancement", "ImageNet normalization", "Train only: crop, flip, rotate, color jitter"].map((s, i) => (
                    <li key={s} className="flex gap-3">
                      <span className="font-mono text-neutral-500">{String(i + 1).padStart(2, "0")}</span>
                      <span className="text-neutral-300">{s}</span>
                    </li>
                  ))}
                </ol>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <SlidersHorizontal className="h-6 w-6 text-neutral-300" />
                <CardTitle className="pt-2">Training recipe</CardTitle>
                <CardDescription>Two phases, {nSeeds} seeds per model.</CardDescription>
              </CardHeader>
              <CardContent>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                  {[
                    ["Phase 1", `${phase1} epochs, backbone frozen`],
                    ["Phase 2", `${metricsM1?.config?.phase2_epochs ?? 15} epochs, all layers`],
                    ["Loss", "Focal (γ = 2)"],
                    ["Optimizer", "Adam, lr 1e-4 → 1e-5"],
                    ["Scheduler", "Plateau on val loss"],
                    ["Selection", "Best val accuracy"],
                  ].map(([k, v]) => (
                    <React.Fragment key={k}>
                      <dt className="text-neutral-500">{k}</dt>
                      <dd className="text-neutral-300">{v}</dd>
                    </React.Fragment>
                  ))}
                </dl>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </div>
  );
}
