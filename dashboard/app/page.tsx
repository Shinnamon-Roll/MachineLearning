import HeroSection from "@/components/hero-section";
import PerformanceChart from "@/components/performance-chart";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "lucide-react"; // Wait, Badge is not in lucide-react. I'll use a simple span or create a badge component.
// Actually I don't need Badge component, I can style a span.

export default function Home() {
  // Mock Data for Charts
  const datasetDistribution = {
    series: [25, 20, 25, 15, 10, 5],
    options: {
      labels: ["Wild Salmon Steak", "Wild Salmon Fillet", "Farmed Salmon Steak", "Farmed Salmon Fillet", "Trout Meat", "Other Meat"],
      colors: ["#ef4444", "#f87171", "#fb923c", "#fdba74", "#38bdf8", "#94a3b8"],
      legend: { position: 'bottom' as const },
    }
  };

  const learningCurve = {
    series: [
      { name: "Training Accuracy", data: [10, 40, 60, 75, 80, 85, 88, 90, 92, 94] },
      { name: "Validation Accuracy", data: [5, 30, 50, 65, 72, 78, 80, 82, 83, 84.58] }
    ],
    options: {
      xaxis: { categories: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], title: { text: "Epochs" } },
      colors: ["#38bdf8", "#f472b6"],
      stroke: { curve: 'smooth' as const, width: 3 },
    }
  };

  const modelMetrics = {
    series: [
      { name: "Improved DenseNet121", data: [84.58, 82.1, 79.5, 77.8] },
      { name: "Baseline CNN", data: [68.4, 65.2, 60.1, 62.5] }
    ],
    options: {
      xaxis: { categories: ["Accuracy", "Precision", "Recall", "F1-Score"] },
      colors: ["#34d399", "#94a3b8"],
      fill: { opacity: 0.2 },
      stroke: { width: 2 },
      markers: { size: 4 },
    }
  };

  return (
    <main className="min-h-screen bg-black text-white p-4 md:p-8 font-sans selection:bg-cyan-500/30">
      
      {/* 1. Hero & Interactive Testing Section */}
      <section className="mb-8">
        <HeroSection />
      </section>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
        
        {/* 2. Data Management & Preprocessing (Spans 2 columns) */}
        <Card className="col-span-1 md:col-span-2 bg-neutral-900/50 border-neutral-800">
          <CardHeader>
            <CardTitle className="text-xl text-cyan-400">Dataset Distribution</CardTitle>
            <CardDescription>6 Classes of Salmon & Trout Data</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col md:flex-row items-center gap-4">
            <div className="w-full md:w-1/2 h-64">
              <PerformanceChart type="donut" series={datasetDistribution.series} options={datasetDistribution.options} height={250} />
            </div>
            <div className="w-full md:w-1/2 space-y-4">
              <div className="p-4 rounded-lg bg-neutral-800/50 border border-neutral-700">
                <h4 className="font-semibold text-white mb-2">Data Management</h4>
                <ul className="text-sm text-neutral-400 space-y-2 list-disc list-inside">
                  <li>Collected from multiple online sources & supermarkets.</li>
                  <li>Preprocessing: Resize (224x224), Normalization, Augmentation.</li>
                  <li>Split: <span className="text-cyan-400">Train (70%)</span>, <span className="text-blue-400">Val (15%)</span>, <span className="text-purple-400">Test (15%)</span>.</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 3. Model Comparison (Spans 2 columns) */}
        <Card className="col-span-1 md:col-span-2 lg:col-span-2 bg-neutral-900/50 border-neutral-800">
          <CardHeader>
            <CardTitle className="text-xl text-purple-400">Model Comparison</CardTitle>
            <CardDescription>Two-Stage Improved DenseNet121 vs Baseline</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-neutral-400">
                <thead className="text-xs text-neutral-200 uppercase bg-neutral-800">
                  <tr>
                    <th className="px-4 py-3 rounded-tl-lg">Model</th>
                    <th className="px-4 py-3">Architecture</th>
                    <th className="px-4 py-3">Accuracy</th>
                    <th className="px-4 py-3 rounded-tr-lg">Key Advantage</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="bg-neutral-800/30 border-b border-neutral-700">
                    <td className="px-4 py-3 font-medium text-white">Baseline CNN</td>
                    <td className="px-4 py-3">Standard CNN (3 Layers)</td>
                    <td className="px-4 py-3">68.40%</td>
                    <td className="px-4 py-3">Simple, Fast</td>
                  </tr>
                  <tr className="bg-cyan-900/20 border-b border-cyan-800/50">
                    <td className="px-4 py-3 font-medium text-cyan-400">Improved DenseNet121</td>
                    <td className="px-4 py-3">DenseNet121 + GAP + Dropout</td>
                    <td className="px-4 py-3 font-bold text-white">84.58%</td>
                    <td className="px-4 py-3 text-cyan-200">High Feature Reuse, Robust to Blur</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="mt-4 p-3 rounded-md bg-cyan-950/30 border border-cyan-900/50 text-xs text-cyan-200">
              <strong>Highlight:</strong> The Two-Stage pipeline (Classification &rarr; Freshness) significantly reduces false positives in freshness grading compared to single-stage models.
            </div>
          </CardContent>
        </Card>

        {/* 4. Performance Evaluation - Learning Curve (Spans 2 columns) */}
        <Card className="col-span-1 md:col-span-2 lg:col-span-2 bg-neutral-900/50 border-neutral-800">
          <CardHeader>
            <CardTitle className="text-xl text-green-400">Learning Curve</CardTitle>
            <CardDescription>Training vs Validation Accuracy over Epochs</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <PerformanceChart type="line" series={learningCurve.series} options={learningCurve.options} height={280} />
            </div>
          </CardContent>
        </Card>

        {/* 4. Performance Evaluation - Metrics Radar (Spans 1 column) */}
        <Card className="col-span-1 md:col-span-1 lg:col-span-1 bg-neutral-900/50 border-neutral-800">
          <CardHeader>
            <CardTitle className="text-xl text-yellow-400">Metrics Radar</CardTitle>
            <CardDescription>Performance Overview</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <PerformanceChart type="radar" series={modelMetrics.series} options={modelMetrics.options} height={250} />
            </div>
          </CardContent>
        </Card>

         {/* Conclusion Box (Spans 1 column) */}
         <Card className="col-span-1 md:col-span-1 lg:col-span-1 bg-gradient-to-br from-neutral-900 to-neutral-800 border-neutral-700 shadow-lg">
          <CardHeader>
            <CardTitle className="text-xl text-white">Conclusion</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-invert prose-sm">
              <p className="text-neutral-300 leading-relaxed">
                The <span className="text-cyan-400 font-semibold">Improved DenseNet121</span> demonstrates superior performance with an accuracy of <span className="text-green-400 font-bold">84.58%</span>.
              </p>
              <p className="text-neutral-300 leading-relaxed mt-2">
                The two-stage approach effectively handles the complexity of classifying fish types before assessing freshness, achieving an F1-Score of over <span className="text-purple-400 font-bold">77%</span>.
              </p>
            </div>
          </CardContent>
        </Card>

      </div>
    </main>
  );
}
