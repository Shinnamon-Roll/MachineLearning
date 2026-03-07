import fs from 'fs';
import path from 'path';
import PerformanceChart from "@/components/performance-chart";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import { Trophy, CheckCircle, TrendingUp, AlertTriangle } from 'lucide-react';
import { cn } from "@/lib/utils";

// Function to read data from JSON files
async function getData() {
  const metricsPath = path.join(process.cwd(), 'public', 'data', 'metrics.json');
  const metrics2Path = path.join(process.cwd(), 'public', 'data', 'metrics_model2.json');
  
  let metrics1 = null;
  let metrics2 = null;

  try {
    if (fs.existsSync(metricsPath)) {
      metrics1 = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    }
    if (fs.existsSync(metrics2Path)) {
      metrics2 = JSON.parse(fs.readFileSync(metrics2Path, 'utf8'));
    }
  } catch (error) {
    console.error("Error reading data", error);
  }

  return { metrics1, metrics2 };
}

export default async function Dashboard() {
  const { metrics1, metrics2 } = await getData();

  // Comparison Data: Only Model 1 and Model 2
  const models = [
    { 
      id: 'model-1', 
      name: 'ImprovedDenseNet121', 
      shortName: 'DenseNet121',
      description: 'High accuracy research-based architecture.',
      status: 'Completed', 
      accuracy: metrics1 ? metrics1.accuracy : 0,
      f1: metrics1 ? metrics1.f1_score : 0,
      precision: metrics1 ? metrics1.precision : 0,
      recall: metrics1 ? metrics1.recall : 0,
      link: '/model-1'
    },
    { 
      id: 'model-2', 
      name: metrics2 ? 'CustomMobileNetV2' : 'CustomMobileNetV2 (Training...)', 
      shortName: 'MobileNetV2',
      description: 'Lightweight model optimized for speed and texture analysis.',
      status: metrics2 ? 'Completed' : 'Training', 
      accuracy: metrics2 ? metrics2.accuracy : 0,
      f1: metrics2 ? metrics2.f1_score : 0,
      precision: metrics2 ? metrics2.precision : 0,
      recall: metrics2 ? metrics2.recall : 0,
      link: '/model-2'
    }
  ];

  // Determine Best Model
  const bestModel = models.reduce((prev, current) => (prev.f1 > current.f1) ? prev : current);
  const bestModelName = bestModel.f1 > 0 ? bestModel.name : "N/A";

  // Chart Data: Grouped Bar Chart for Comprehensive Comparison
  const comparisonSeries = [
    {
      name: 'Accuracy',
      data: models.map(m => parseFloat((m.accuracy * 100).toFixed(2)))
    },
    {
      name: 'Precision',
      data: models.map(m => parseFloat((m.precision * 100).toFixed(2)))
    },
    {
      name: 'Recall',
      data: models.map(m => parseFloat((m.recall * 100).toFixed(2)))
    },
    {
      name: 'F1-Score',
      data: models.map(m => parseFloat((m.f1 * 100).toFixed(2)))
    }
  ];

  const comparisonOptions: any = {
    chart: { type: 'bar', toolbar: { show: false }, stacked: false },
    plotOptions: {
      bar: { horizontal: false, columnWidth: '60%', borderRadius: 4 }
    },
    dataLabels: { enabled: false },
    stroke: { show: true, width: 2, colors: ['transparent'] },
    xaxis: {
      categories: models.map(m => m.shortName),
      labels: { style: { fontSize: '14px', fontWeight: 'bold' } }
    },
    yaxis: {
      max: 100,
      title: { text: 'Score (%)' }
    },
    fill: { opacity: 1 },
    theme: {
        monochrome: { enabled: false }, // Disable monochrome to use colors
        mode: 'dark' 
    },
    colors: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'], // Blue, Green, Amber, Violet
    legend: { position: 'top' },
    tooltip: {
      y: {
      }
    }
  };

  // Hover Effect Items
  const projectItems = models.map(m => ({
    title: m.name,
    description: m.description,
    link: m.link
  }));

  return (
    <div className="min-h-screen bg-neutral-950 text-foreground p-4 md:p-8 font-sans transition-colors duration-300 relative overflow-hidden">
      
      {/* Background Ambience */}
      <div className="absolute inset-0 w-full h-full bg-neutral-950 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(59,130,246,0.15),rgba(255,255,255,0))] pointer-events-none" />
      
      {/* Header */}
      <header className="mb-12 text-center relative z-10">
        <div className="inline-block p-2 px-4 rounded-full bg-blue-500/10 border border-blue-500/20 mb-4 backdrop-blur-sm">
            <span className="text-blue-400 text-xs font-bold tracking-widest uppercase">Salmon vs Trout Classification</span>
        </div>
        <h1 className="text-4xl md:text-6xl font-extrabold mb-4 tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-white to-white/60">
          Project Dashboard
        </h1>
        <p className="text-neutral-400 max-w-2xl mx-auto text-lg">
          Comparative analysis of deep learning models for fish classification.
        </p>
      </header>

      {/* Quick Stats / Best Model */}
      <section className="mb-16 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-neutral-400">Best Performing Model</CardTitle>
                    <Trophy className="h-4 w-4 text-yellow-500" />
                </CardHeader>
                <CardContent>
                    <div className="text-2xl font-bold text-white">{bestModelName}</div>
                    <p className="text-xs text-neutral-500 mt-1">Based on F1-Score</p>
                </CardContent>
            </Card>
            <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-neutral-400">Highest Accuracy</CardTitle>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                </CardHeader>
                <CardContent>
                    <div className="text-2xl font-bold text-white">{(bestModel.accuracy * 100).toFixed(2)}%</div>
                    <p className="text-xs text-neutral-500 mt-1">Achieved by {bestModel.shortName}</p>
                </CardContent>
            </Card>
            <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-neutral-400">Project Status</CardTitle>
                    <TrendingUp className="h-4 w-4 text-blue-500" />
                </CardHeader>
                <CardContent>
                    <div className="text-2xl font-bold text-white">2 Models</div>
                    <p className="text-xs text-neutral-500 mt-1">Ready for Comparison</p>
                </CardContent>
            </Card>
        </div>
      </section>

      {/* Model Navigation with Hover Effect */}
      <section className="mb-16 relative z-10">
        <h2 className="text-2xl font-bold mb-6 text-white border-l-4 border-blue-500 pl-4">Explore Models</h2>
        <HoverEffect items={projectItems} />
      </section>

      {/* Comprehensive Comparison Chart */}
      <section className="mb-16 relative z-10">
        <h2 className="text-2xl font-bold mb-6 text-white border-l-4 border-purple-500 pl-4">Performance Comparison</h2>
        <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md">
            <CardHeader>
                <CardTitle className="text-white">Metrics Overview</CardTitle>
                <CardDescription>Side-by-side comparison of Accuracy, Precision, Recall, and F1-Score.</CardDescription>
            </CardHeader>
            <CardContent>
                <div className="h-[400px] w-full">
                    <PerformanceChart type="bar" series={comparisonSeries} options={comparisonOptions} height={400} yAxisFormat="percentage" />
                </div>
            </CardContent>
        </Card>
      </section>

      {/* Conclusion & Analysis */}
      <section className="mb-16 relative z-10">
        <h2 className="text-2xl font-bold mb-6 text-white border-l-4 border-green-500 pl-4">Experimental Conclusion</h2>
        <Card className="bg-neutral-900/50 border-neutral-800 backdrop-blur-md p-6">
            <div className="prose prose-invert max-w-none">
                <h3 className="text-xl font-semibold text-blue-400 mb-4">Analysis Summary</h3>
                <ul className="list-disc pl-5 space-y-2 text-neutral-300">
                    <li>
                        <strong className="text-white">Model Selection:</strong> Based on current experiments, <strong className="text-yellow-400">{bestModelName}</strong> is the superior model with an F1-Score of <strong>{(bestModel.f1 * 100).toFixed(2)}%</strong>.
                    </li>
                    <li>
                        <strong className="text-white">Trade-offs:</strong> 
                        {metrics1 && metrics2 && metrics1.accuracy > metrics2.accuracy ? 
                            " ImprovedDenseNet121 offers higher overall accuracy, making it suitable for general classification tasks." : 
                            " CustomMobileNetV2 demonstrates competitive performance with likely faster inference speeds due to its lightweight architecture."
                        }
                    </li>
                    <li>
                        <strong className="text-white">Recommendation:</strong> For deployment where accuracy is paramount, use <strong>{bestModelName}</strong>. If speed on edge devices is critical, consider <strong>CustomMobileNetV2</strong> if its accuracy is within acceptable range.
                    </li>
                </ul>
            </div>
        </Card>
      </section>

    </div>
  );
}
