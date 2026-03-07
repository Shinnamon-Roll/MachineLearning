import fs from 'fs';
import path from 'path';
import HeroSection from "@/components/hero-section";
import PerformanceChart from "@/components/performance-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

// Function to read data from JSON files
async function getData() {
  const metricsPath = path.join(process.cwd(), 'public', 'data', 'metrics.json');
  const historyPath = path.join(process.cwd(), 'public', 'data', 'training_history.json');

  let metrics = null;
  let history = null;

  try {
    if (fs.existsSync(metricsPath)) {
      metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    }
    if (fs.existsSync(historyPath)) {
      history = JSON.parse(fs.readFileSync(historyPath, 'utf8'));
    }
  } catch (error) {
    console.error("Error reading data", error);
  }

  return { metrics, history };
}

export default async function Model1Page() {
  const { metrics, history } = await getData();

  // --- Prepare Chart Data ---

  // 1. Learning Curve
  let learningCurveSeries: { name: string; data: number[] }[] = [];
  let learningCurveOptions: any = {};

  if (history) {
    learningCurveSeries = [
      { name: "Training Loss", data: history.train_loss },
      { name: "Validation Loss", data: history.val_loss },
      { name: "Training Accuracy", data: history.train_acc },
      { name: "Validation Accuracy", data: history.val_acc }
    ];
    learningCurveOptions = {
      chart: {
        id: 'learning-curve',
        toolbar: { show: false },
        zoom: { enabled: false },
      },
      colors: ['#3b82f6', '#94a3b8', '#3b82f6', '#94a3b8'], // Blue and Slate
      stroke: {
        width: [2, 2, 2, 2],
        dashArray: [0, 0, 5, 5] // Solid for Loss, Dashed for Acc
      },
      xaxis: {
        title: { text: 'Epochs' },
        categories: Array.from({ length: history.train_loss.length }, (_, i) => i + 1)
      },
      yaxis: [
        { title: { text: 'Loss' } },
        { opposite: true, title: { text: 'Accuracy' }, max: 1.0, min: 0.0 }
      ],
      legend: { position: 'top' },
      theme: {
          monochrome: { enabled: false },
          mode: 'dark'
      }
    };
  }

  // 2. Metrics Bar Chart
  const metricsSeries = [{
    name: 'Score',
    data: metrics ? [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score] : []
  }];
  
  const metricsOptions: any = {
    chart: { type: 'bar', toolbar: { show: false } },
    plotOptions: {
      bar: { borderRadius: 4, horizontal: true, barHeight: '50%' }
    },
    colors: ['#3b82f6'], // Blue
    xaxis: {
      categories: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
      max: 1.0
    },
    theme: {
        monochrome: { enabled: true, color: '#3b82f6' },
        mode: 'dark'
    }
  };


  return (
    <div className="min-h-screen bg-neutral-950 text-foreground p-4 md:p-8 font-sans transition-colors duration-300 relative overflow-hidden">
      
      {/* Background Ambience */}
      <div className="absolute inset-0 w-full h-full bg-neutral-950 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(59,130,246,0.15),rgba(255,255,255,0))] pointer-events-none" />

      {/* Back Button */}
      <div className="relative z-10 mb-6">
        <Link href="/" className="inline-flex items-center text-sm text-neutral-400 hover:text-white transition-colors">
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Dashboard
        </Link>
      </div>

      {/* Header */}
      <header className="mb-10 text-center relative z-10">
        <h1 className="text-3xl md:text-5xl font-bold mb-2 tracking-tight text-white">
          Model 1: ImprovedDenseNet121
        </h1>
        <p className="text-neutral-400 mt-2 text-lg">Research Based Architecture (Binary Classification)</p>
      </header>

      {/* Part 1: Inference */}
      <section className="mb-16 border-b border-neutral-800 pb-10 relative z-10">
        <div className="flex items-center gap-2 mb-6">
           <span className="bg-blue-600/20 text-blue-400 border border-blue-600/30 px-3 py-1 text-sm font-bold rounded-full">INTERACTIVE</span>
           <h2 className="text-2xl font-bold text-white">Try Model 1</h2>
        </div>
        <HeroSection modelId="model-1" />
      </section>

      {/* Part 2: Model Details */}
      <section className="mb-16 relative z-10">
        <div className="flex items-center gap-2 mb-6">
           <span className="bg-purple-600/20 text-purple-400 border border-purple-600/30 px-3 py-1 text-sm font-bold rounded-full">DETAILS</span>
           <h2 className="text-2xl font-bold text-white">Model Performance & Metrics</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Data Management */}
          <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md lg:col-span-2">
            <CardHeader className="border-b border-neutral-800">
              <CardTitle className="text-xl font-bold text-white">Data Management & Preprocessing</CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-neutral-300">
                <div className="space-y-2">
                    <div className="flex justify-between border-b border-neutral-800 pb-1">
                        <span className="font-semibold text-white">Dataset Source</span>
                        <span className="text-neutral-400">Local (/Image)</span>
                    </div>
                    <div className="flex justify-between border-b border-neutral-800 pb-1">
                        <span className="font-semibold text-white">Classes</span>
                        <span className="text-neutral-400">Salmon, Trout</span>
                    </div>
                </div>
                <div className="space-y-2">
                    <div className="flex justify-between border-b border-neutral-800 pb-1">
                        <span className="font-semibold text-white">Image Size</span>
                        <span className="text-neutral-400">224x224</span>
                    </div>
                    <div className="flex justify-between border-b border-neutral-800 pb-1">
                        <span className="font-semibold text-white">Batch Size</span>
                        <span className="text-neutral-400">32</span>
                    </div>
                </div>
                <div className="space-y-2">
                     <div className="flex justify-between border-b border-neutral-800 pb-1">
                        <span className="font-semibold text-white">Augmentation</span>
                        <span className="text-neutral-400">Resize, Norm</span>
                    </div>
                     <div className="flex justify-between border-b border-neutral-800 pb-1">
                        <span className="font-semibold text-white">Train/Test Split</span>
                        <span className="text-neutral-400">80/20</span>
                    </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Learning Curve */}
          <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-white">Learning Curve (Loss & Accuracy)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[350px] w-full">
                {history ? (
                   <PerformanceChart type="line" series={learningCurveSeries} options={learningCurveOptions} />
                ) : (
                    <div className="flex items-center justify-center h-full text-neutral-500">No training history available</div>
                )}
              </div>
            </CardContent>
          </Card>

           {/* Metrics */}
           <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md">
            <CardHeader>
              <CardTitle className="text-white">Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] w-full">
                {metrics ? (
                   <PerformanceChart type="bar" series={metricsSeries} options={metricsOptions} />
                ) : (
                    <div className="flex items-center justify-center h-full text-neutral-500">No metrics available</div>
                )}
              </div>
            </CardContent>
          </Card>

           {/* Confusion Matrix Placeholder (since we can't easily render the image dynamically without API, assume static image exists or omit) */}
           <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md">
            <CardHeader>
              <CardTitle className="text-white">Confusion Matrix</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center justify-center h-[300px]">
                <div className="text-center">
                    <p className="text-neutral-500 mb-4">Visual representation of True Positives vs False Positives</p>
                    {/* If you have an image, you can load it here */}
                    <div className="w-48 h-48 bg-neutral-800 rounded-lg mx-auto flex items-center justify-center">
                        <span className="text-neutral-600 text-xs">Matrix Image</span>
                    </div>
                </div>
            </CardContent>
          </Card>

        </div>
      </section>
    </div>
  );
}
