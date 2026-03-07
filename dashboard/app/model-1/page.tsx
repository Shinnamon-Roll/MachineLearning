import fs from 'fs';
import path from 'path';
import HeroSection from "@/components/hero-section";
import PerformanceChart from "@/components/performance-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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
  };


  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8 font-sans transition-colors duration-300">
      
      {/* Header */}
      <header className="mb-10 text-center relative">
        <h1 className="text-3xl font-bold mb-2 uppercase tracking-widest border-b-4 border-foreground inline-block pb-2">
          Model 1: ImprovedDenseNet121
        </h1>
        <p className="text-muted-foreground mt-2">Research Based Architecture (Binary Classification)</p>
      </header>

      {/* Part 1: Inference */}
      <section className="mb-16 border-b border-border pb-10">
        <div className="flex items-center gap-2 mb-6">
           <span className="bg-primary text-primary-foreground px-3 py-1 text-sm font-bold rounded-sm">INTERACTIVE</span>
           <h2 className="text-2xl font-bold">Try Model 1</h2>
        </div>
        <HeroSection />
      </section>

      {/* Part 2: Model Details */}
      <section className="mb-16">
        <div className="flex items-center gap-2 mb-6">
           <span className="bg-primary text-primary-foreground px-3 py-1 text-sm font-bold rounded-sm">DETAILS</span>
           <h2 className="text-2xl font-bold">Model Performance & Metrics</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Data Management */}
          <Card className="bg-card border-2 border-primary shadow-none rounded-none lg:col-span-2">
            <CardHeader className="bg-muted border-b-2 border-primary">
              <CardTitle className="text-xl font-bold">Data Management & Preprocessing</CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                    <div className="flex justify-between border-b border-border pb-1">
                        <span className="font-semibold">Dataset Source</span>
                        <span className="text-muted-foreground">Local (/Image)</span>
                    </div>
                    <div className="flex justify-between border-b border-border pb-1">
                        <span className="font-semibold">Classes</span>
                        <span className="text-muted-foreground">{metrics?.classes.join(", ") || "Loading..."}</span>
                    </div>
                    <div className="flex justify-between border-b border-border pb-1">
                        <span className="font-semibold">Test Samples</span>
                        <span className="text-muted-foreground">{metrics?.test_samples || "N/A"}</span>
                    </div>
                </div>
                <div className="md:col-span-2 bg-muted/50 p-4 border border-border">
                  <h4 className="font-bold mb-2">Preprocessing Steps:</h4>
                  <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                    <li>Image Resizing (224x224)</li>
                    <li>Normalization (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])</li>
                    <li>Data Augmentation (Random Horizontal Flip, Rotation)</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Evaluation & Results */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Metrics Cards */}
            <div className="lg:col-span-1 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                <div className="bg-primary text-primary-foreground p-4 text-center">
                    <div className="text-3xl font-bold">{(metrics?.accuracy * 100).toFixed(1)}%</div>
                    <div className="text-xs uppercase tracking-wider opacity-80">Accuracy</div>
                </div>
                <div className="bg-card border border-primary p-4 text-center">
                    <div className="text-3xl font-bold">{(metrics?.precision * 100).toFixed(1)}%</div>
                    <div className="text-xs uppercase tracking-wider text-muted-foreground">Precision</div>
                </div>
                <div className="bg-card border border-primary p-4 text-center">
                    <div className="text-3xl font-bold">{(metrics?.recall * 100).toFixed(1)}%</div>
                    <div className="text-xs uppercase tracking-wider text-muted-foreground">Recall</div>
                </div>
                <div className="bg-muted text-foreground p-4 text-center">
                    <div className="text-3xl font-bold">{(metrics?.f1_score * 100).toFixed(1)}%</div>
                    <div className="text-xs uppercase tracking-wider text-muted-foreground">F1 Score</div>
                </div>
                </div>
                
                {/* Bar Chart */}
                <div className="border border-border p-2 bg-card">
                <PerformanceChart type="bar" series={metricsSeries} options={metricsOptions} height={200} />
                </div>
            </div>

            {/* Learning Curve */}
            <div className="lg:col-span-2 border border-border bg-card p-4">
                <h4 className="font-bold mb-4 text-center uppercase text-sm tracking-widest">Learning Curve (Loss & Accuracy)</h4>
                {history ? (
                <PerformanceChart type="line" series={learningCurveSeries} options={learningCurveOptions} height={300} />
                ) : (
                <div className="h-64 flex items-center justify-center text-muted-foreground italic">
                    Training history not available. Please re-train the model.
                </div>
                )}
            </div>

            {/* Confusion Matrix */}
            <div className="lg:col-span-3 mt-8">
                <h4 className="font-bold mb-4 uppercase text-sm tracking-widest border-b border-border pb-2">Confusion Matrix Analysis</h4>
                <div className="flex flex-col md:flex-row gap-8 items-start">
                <div className="border border-primary p-2 bg-card inline-block">
                    {/* Display the saved image */}
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src="/data/confusion_matrix.png" alt="Confusion Matrix" className="max-w-full h-auto dark:invert" style={{ maxHeight: '400px' }} />
                </div>
                <div className="flex-1 space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                        <strong>Analysis:</strong> The confusion matrix shows the model&apos;s performance on the test set. 
                        Diagonal elements represent correct predictions. Off-diagonal elements represent misclassifications.
                    </p>
                    <ul className="list-disc list-inside text-muted-foreground space-y-2">
                        <li><strong>True Positives (Salmon):</strong> Correctly identified Salmon images.</li>
                        <li><strong>True Negatives (Trout):</strong> Correctly identified Trout images.</li>
                        <li><strong>False Positives/Negatives:</strong> Images where the model confused Salmon for Trout or vice versa.</li>
                    </ul>
                    <div className="bg-muted p-4 border-l-4 border-primary">
                        <p className="text-sm italic text-muted-foreground">
                            &quot;The model demonstrates {metrics?.accuracy > 0.8 ? "strong" : "moderate"} capability in distinguishing between Salmon and Trout, 
                            with an F1-score of {(metrics?.f1_score).toFixed(2)}. Further improvements could be made by increasing the dataset size or fine-tuning hyperparameters.&quot;
                        </p>
                    </div>
                </div>
                </div>
            </div>

        </div>
      </section>

    </div>
  );
}
