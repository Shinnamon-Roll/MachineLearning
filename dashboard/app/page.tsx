import fs from 'fs';
import path from 'path';
import PerformanceChart from "@/components/performance-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Link from 'next/link';
import { ArrowRight, BarChart3, Microscope } from 'lucide-react';
import { Button } from "@/components/ui/button";

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

  // Comparison Data
  const models = [
    { 
      id: 'model-1', 
      name: 'ImprovedDenseNet121', 
      status: 'Completed', 
      accuracy: metrics1 ? metrics1.accuracy : 0,
      f1: metrics1 ? metrics1.f1_score : 0,
      precision: metrics1 ? metrics1.precision : 0,
      recall: metrics1 ? metrics1.recall : 0
    },
    { 
      id: 'model-2', 
      name: metrics2 ? 'CustomMobileNetV2' : 'Model 2 (Pending)', 
      status: metrics2 ? 'Completed' : 'In Development', 
      accuracy: metrics2 ? metrics2.accuracy : 0,
      f1: metrics2 ? metrics2.f1_score : 0,
      precision: metrics2 ? metrics2.precision : 0,
      recall: metrics2 ? metrics2.recall : 0
    },
    { 
      id: 'model-3', 
      name: 'Model 3 (Pending)', 
      status: 'In Development', 
      accuracy: 0,
      f1: 0,
      precision: 0,
      recall: 0
    }
  ];

  // Chart Data: Accuracy Comparison
  const accuracySeries = [{
    name: 'Accuracy',
    data: models.map(m => parseFloat((m.accuracy * 100).toFixed(2)))
  }];

  const accuracyOptions: any = {
    chart: { type: 'bar', toolbar: { show: false } },
    plotOptions: {
      bar: { borderRadius: 4, columnWidth: '50%', distributed: true }
    },
    colors: [
      models[0].status === 'Completed' ? '#3b82f6' : '#94a3b8',
      models[1].status === 'Completed' ? '#3b82f6' : '#94a3b8',
      models[2].status === 'Completed' ? '#3b82f6' : '#94a3b8'
    ],
    xaxis: {
      categories: models.map(m => m.name),
      labels: { style: { fontSize: '12px' } }
    },
    yaxis: {
      max: 100,
      title: { text: 'Accuracy (%)' }
    },
    legend: { show: false }
  };

  // Chart Data: F1-Score Comparison
  const f1Series = [{
    name: 'F1-Score',
    data: models.map(m => parseFloat((m.f1 * 100).toFixed(2)))
  }];

  const f1Options: any = {
    chart: { type: 'bar', toolbar: { show: false } },
    plotOptions: {
      bar: { borderRadius: 4, columnWidth: '50%', distributed: true }
    },
    colors: [
      models[0].status === 'Completed' ? '#10b981' : '#94a3b8',
      models[1].status === 'Completed' ? '#10b981' : '#94a3b8',
      models[2].status === 'Completed' ? '#10b981' : '#94a3b8'
    ],
    xaxis: {
      categories: models.map(m => m.name),
      labels: { style: { fontSize: '12px' } }
    },
    yaxis: {
      max: 100,
      title: { text: 'F1-Score (%)' }
    },
    legend: { show: false }
  };

  // Calculate statistics
  const completedModels = models.filter(m => m.status === 'Completed').length;
  const bestModel = models.reduce((prev, current) => (prev.accuracy > current.accuracy) ? prev : current);

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8 font-sans transition-colors duration-300">
      
      {/* Header */}
      <header className="mb-10 text-center relative">
        <h1 className="text-4xl font-bold mb-2 uppercase tracking-widest border-b-4 border-foreground inline-block pb-2">
          Project Dashboard
        </h1>
        <p className="text-muted-foreground mt-2">Salmon & Trout Classification Model Comparison</p>
      </header>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <Card className="bg-card border-primary/20 shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Best Performing Model</CardTitle>
                <Microscope className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">{bestModel.accuracy > 0 ? bestModel.name : 'N/A'}</div>
                <p className="text-xs text-muted-foreground">
                    Accuracy: {bestModel.accuracy > 0 ? (bestModel.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                </p>
            </CardContent>
        </Card>
        <Card className="bg-card border-primary/20 shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Models Developed</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">{completedModels} / 3</div>
                <p className="text-xs text-muted-foreground">
                    Target: 3 Models Comparison
                </p>
            </CardContent>
        </Card>
        <Card className="bg-card border-primary/20 shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Current Status</CardTitle>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">Phase {completedModels} Complete</div>
                <p className="text-xs text-muted-foreground">
                    Developing Model {completedModels + 1}
                </p>
            </CardContent>
        </Card>
      </div>

      {/* Comparison Section */}
      <section className="mb-16">
        <div className="flex items-center gap-2 mb-6">
           <span className="bg-primary text-primary-foreground px-3 py-1 text-sm font-bold rounded-sm">COMPARISON</span>
           <h2 className="text-2xl font-bold">Model Performance Comparison</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* Accuracy Chart */}
            <Card className="bg-card shadow-sm">
                <CardHeader>
                    <CardTitle>Accuracy Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                    <PerformanceChart type="bar" series={accuracySeries} options={accuracyOptions} height={300} />
                </CardContent>
            </Card>

            {/* F1 Chart */}
            <Card className="bg-card shadow-sm">
                <CardHeader>
                    <CardTitle>F1-Score Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                    <PerformanceChart type="bar" series={f1Series} options={f1Options} height={300} />
                </CardContent>
            </Card>
        </div>

        {/* Detailed Table */}
        <Card className="bg-card border-2 border-primary shadow-none rounded-none overflow-hidden">
            <CardHeader className="bg-muted border-b-2 border-primary">
              <CardTitle className="text-xl font-bold">Detailed Metrics Table</CardTitle>
            </CardHeader>
            <CardContent className="p-0 overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-primary text-primary-foreground uppercase">
                  <tr>
                    <th className="px-6 py-4">Model Name</th>
                    <th className="px-6 py-4">Status</th>
                    <th className="px-6 py-4">Accuracy</th>
                    <th className="px-6 py-4">Precision</th>
                    <th className="px-6 py-4">Recall</th>
                    <th className="px-6 py-4">F1 Score</th>
                    <th className="px-6 py-4">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {models.map((model, index) => (
                    <tr key={index} className={model.status === 'Completed' ? 'bg-background' : 'bg-muted/30'}>
                        <td className="px-6 py-4 font-bold">{model.name}</td>
                        <td className="px-6 py-4">
                            <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                                model.status === 'Completed' 
                                ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-100' 
                                : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-100'
                            }`}>
                                {model.status}
                            </span>
                        </td>
                        <td className="px-6 py-4 font-mono">{model.accuracy > 0 ? (model.accuracy * 100).toFixed(2) + '%' : '-'}</td>
                        <td className="px-6 py-4 font-mono">{model.precision > 0 ? (model.precision * 100).toFixed(2) + '%' : '-'}</td>
                        <td className="px-6 py-4 font-mono">{model.recall > 0 ? (model.recall * 100).toFixed(2) + '%' : '-'}</td>
                        <td className="px-6 py-4 font-mono">{model.f1 > 0 ? (model.f1 * 100).toFixed(2) + '%' : '-'}</td>
                        <td className="px-6 py-4">
                            {model.status === 'Completed' || model.id === 'model-2' ? (
                                <Link href={`/${model.id}`}>
                                    <Button variant="outline" size="sm">View Details</Button>
                                </Link>
                            ) : (
                                <span className="text-muted-foreground text-xs">Coming Soon</span>
                            )}
                        </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
        </Card>
      </section>

    </div>
  );
}
