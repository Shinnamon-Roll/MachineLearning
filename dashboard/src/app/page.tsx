"use client";

import React, { useState, useEffect } from "react";
import { Spotlight } from "@/components/ui/spotlight";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
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
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { Database, FileImage, Layers, Server } from "lucide-react";

// --- Mock Data Fallback ---
const fallbackLearningCurveData = [
  { epoch: 1, loss_m1: 0.8, loss_m2: 0.9, val_loss_m1: 0.75, val_loss_m2: 0.85 },
];

const COLORS = ["#ffffff", "#a3a3a3", "#525252"];

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
  split_counts?: {
    train: number;
    val: number;
    test: number;
  };
};

type TrainingHistory = {
  train_loss: number[];
  train_acc: number[];
  val_loss: number[];
  val_acc: number[];
};

export default function Home() {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const [metricsM1, setMetricsM1] = useState<Metrics | null>(null);
  const [metricsM2, setMetricsM2] = useState<Metrics | null>(null);
  const [chartData, setChartData] = useState<any[]>(fallbackLearningCurveData);
  const [dataSplitData, setDataSplitData] = useState([
    { name: "Train", value: 80 },
    { name: "Validation", value: 10 },
    { name: "Test", value: 10 },
  ]);

  const [predictionM1, setPredictionM1] = useState<any>(null);
  const [predictionM2, setPredictionM2] = useState<any>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        // Fetch Metrics
        const m1Res = await fetch('/data/metrics.json');
        const m1Data = await m1Res.json();
        setMetricsM1(m1Data);

        if (m1Data.split_counts) {
            setDataSplitData([
                { name: "Train", value: m1Data.split_counts.train },
                { name: "Validation", value: m1Data.split_counts.val },
                { name: "Test", value: m1Data.split_counts.test },
            ]);
        }

        const m2Res = await fetch('/data/metrics_model2.json');
        const m2Data = await m2Res.json();
        setMetricsM2(m2Data);

        // Fetch Training History
        const h1Res = await fetch('/data/training_history.json');
        const h1Data: TrainingHistory = await h1Res.json();

        const h2Res = await fetch('/data/training_history_model2.json');
        const h2Data: TrainingHistory = await h2Res.json();

        // Merge History for Chart
        const epochs = Math.max(h1Data.train_loss.length, h2Data.train_loss.length);
        const mergedData = [];
        for (let i = 0; i < epochs; i++) {
            mergedData.push({
                epoch: i + 1,
                loss_m1: h1Data.train_loss[i] || 0,
                val_loss_m1: h1Data.val_loss[i] || 0,
                loss_m2: h2Data.train_loss[i] || 0,
                val_loss_m2: h2Data.val_loss[i] || 0,
            });
        }
        setChartData(mergedData);

      } catch (error) {
        console.error("Failed to fetch dashboard data", error);
      }
    }

    fetchData();
  }, []);

  const handleFileUpload = async (files: File[]) => {
    setUploadedFiles(files);
    setIsProcessing(true);
    setPredictionM1(null);
    setPredictionM2(null);
    
    // Create preview URL
    if (files && files[0]) {
        const objectUrl = URL.createObjectURL(files[0]);
        setPreviewImage(objectUrl);
    }

    const formData = new FormData();
    formData.append("file", files[0]);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        
        if (data.model1 && !data.model1.error) {
            setPredictionM1(data.model1);
        }
        if (data.model2 && !data.model2.error) {
            setPredictionM2(data.model2);
        }
    } catch (error) {
        console.error("Prediction Error:", error);
    } finally {
        setIsProcessing(false);
    }
  };

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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-24 pb-24">
        
        {/* 2. Interactive Model Comparison */}
        <section>
          <h2 className="text-3xl font-bold mb-8 text-white border-l-4 border-white pl-4">
            Interactive Model Comparison
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
            <div className="w-full space-y-4">
              <Card className="h-full">
                 <CardHeader>
                    <CardTitle>Upload Image</CardTitle>
                    <CardDescription>Drag & drop a Salmon or Trout image to analyze.</CardDescription>
                 </CardHeader>
                 <CardContent>
                    <FileUpload onChange={handleFileUpload} />
                 </CardContent>
              </Card>

              {/* Image Preview */}
              {previewImage && (
                  <Card className="overflow-hidden border-neutral-800 bg-neutral-950">
                      <CardHeader className="pb-2">
                          <CardTitle className="text-sm text-neutral-400">Input Image</CardTitle>
                      </CardHeader>
                      <CardContent className="flex justify-center p-6 bg-black/40">
                          <div className="relative w-64 h-64 rounded-lg overflow-hidden border border-neutral-800 shadow-2xl">
                              <img 
                                src={previewImage} 
                                alt="Uploaded input" 
                                className="w-full h-full object-cover"
                              />
                          </div>
                      </CardContent>
                  </Card>
              )}
            </div>

            <div className="space-y-6">
               {/* Model 1 Result */}
               <Card className="bg-neutral-950 border-neutral-800">
                 <CardHeader className="pb-2">
                   <div className="flex justify-between items-center">
                     <CardTitle className="text-xl text-white">Model 1: DenseNet121</CardTitle>
                     <span className="text-xs font-mono text-neutral-400">Acc: {metricsM1 ? (metricsM1.accuracy * 100).toFixed(1) : "..."}%</span>
                   </div>
                 </CardHeader>
                 <CardContent>
                   {isProcessing ? (
                     <div className="animate-pulse space-y-2">
                       <div className="h-4 bg-neutral-800 rounded w-3/4"></div>
                       <div className="h-4 bg-neutral-800 rounded w-1/2"></div>
                     </div>
                   ) : predictionM1 ? (
                     <div className="space-y-4">
                       <div className="flex justify-between text-sm">
                         <span className="text-neutral-300">Prediction:</span>
                         <span className="font-bold text-white">{predictionM1.class}</span>
                       </div>
                       <div className="space-y-1">
                          <div className="flex justify-between text-xs text-neutral-400">
                            <span>Confidence</span>
                            <span>{predictionM1.confidence.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-neutral-800 h-2 rounded-none">
                            <div className="bg-white h-2 rounded-none" style={{ width: `${predictionM1.confidence}%` }}></div>
                          </div>
                       </div>
                     </div>
                   ) : (
                     <div className="text-neutral-500 text-sm py-4 italic">Waiting for input...</div>
                   )}
                 </CardContent>
               </Card>

               {/* Model 2 Result */}
               <Card className="bg-neutral-950 border-neutral-800">
                 <CardHeader className="pb-2">
                   <div className="flex justify-between items-center">
                     <CardTitle className="text-xl text-white">Model 2: MobileNetV2</CardTitle>
                     <span className="text-xs font-mono text-neutral-400">Acc: {metricsM2 ? (metricsM2.accuracy * 100).toFixed(1) : "..."}%</span>
                   </div>
                 </CardHeader>
                 <CardContent>
                    {isProcessing ? (
                     <div className="animate-pulse space-y-2">
                       <div className="h-4 bg-neutral-800 rounded w-3/4"></div>
                       <div className="h-4 bg-neutral-800 rounded w-1/2"></div>
                     </div>
                   ) : predictionM2 ? (
                     <div className="space-y-4">
                       <div className="flex justify-between text-sm">
                         <span className="text-neutral-300">Prediction:</span>
                         <span className="font-bold text-white">{predictionM2.class}</span>
                       </div>
                       <div className="space-y-1">
                          <div className="flex justify-between text-xs text-neutral-400">
                            <span>Confidence</span>
                            <span>{predictionM2.confidence.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-neutral-800 h-2 rounded-none">
                            <div className="bg-neutral-400 h-2 rounded-none" style={{ width: `${predictionM2.confidence}%` }}></div>
                          </div>
                       </div>
                     </div>
                   ) : (
                     <div className="text-neutral-500 text-sm py-4 italic">Waiting for input...</div>
                   )}
                 </CardContent>
               </Card>
            </div>
          </div>
        </section>

        {/* 3. Data Management & Preprocessing */}
        <section>
          <h2 className="text-3xl font-bold mb-8 text-white border-l-4 border-white pl-4">
            Data Management & Preprocessing Pipeline
          </h2>
          
          <BentoGrid className="max-w-7xl mx-auto">
            {/* Card A: Data Sources */}
            <BentoGridItem
              title="Robust Data Sources"
              description="Data gathered from multiple diverse sources to ensure generalization and prevent bias."
              header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-none bg-neutral-900 items-center justify-center"><Database className="h-10 w-10 text-neutral-300" /></div>}
              icon={<Server className="h-4 w-4 text-neutral-500" />}
              className="md:col-span-1"
            />
            
            {/* Card B: Preprocessing */}
            <BentoGridItem
              title="Preprocessing Pipeline"
              description="Resize (224x224) -> Normalization -> Augmentation (CLAHE, Random Crop)."
              header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-none bg-neutral-900 items-center justify-center"><Layers className="h-10 w-10 text-neutral-300" /></div>}
              icon={<FileImage className="h-4 w-4 text-neutral-500" />}
              className="md:col-span-1"
            />
            
            {/* Card C: Data Split */}
            <BentoGridItem
              title="Dataset Split"
              description="Train (80%) / Validation (10%) / Test (10%) strategy."
              header={
                <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-none bg-neutral-900 items-center justify-center">
                  <ResponsiveContainer width="100%" height={150}>
                    <PieChart>
                      <Pie
                        data={dataSplitData}
                        cx="50%"
                        cy="50%"
                        innerRadius={30}
                        outerRadius={50}
                        fill="#525252"
                        paddingAngle={5}
                        dataKey="value"
                        stroke="none"
                      >
                        {dataSplitData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#000', borderColor: '#333', color: '#fff' }}
                        itemStyle={{ color: '#fff' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              }
              icon={<Database className="h-4 w-4 text-neutral-500" />}
              className="md:col-span-1"
            />
          </BentoGrid>
        </section>

        {/* 4. Performance Metrics */}
        <section>
          <h2 className="text-3xl font-bold mb-8 text-white border-l-4 border-white pl-4">
            Performance Metrics
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
               <Card>
                 <CardHeader>
                   <CardTitle>Learning Curve</CardTitle>
                   <CardDescription>Training vs. Validation Loss (Epochs)</CardDescription>
                 </CardHeader>
                 <CardContent className="pl-0">
                   <div className="h-[300px] w-full">
                     <ResponsiveContainer width="100%" height="100%">
                       <LineChart data={chartData}>
                         <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                         <XAxis dataKey="epoch" stroke="#888" />
                         <YAxis stroke="#888" />
                         <Tooltip 
                            contentStyle={{ backgroundColor: '#000', borderColor: '#333', color: '#fff' }}
                         />
                         <Line type="monotone" dataKey="loss_m1" name="M1 Train Loss" stroke="#ffffff" strokeWidth={2} dot={false} />
                         <Line type="monotone" dataKey="val_loss_m1" name="M1 Val Loss" stroke="#a3a3a3" strokeDasharray="5 5" dot={false} />
                         <Line type="monotone" dataKey="loss_m2" name="M2 Train Loss" stroke="#525252" strokeWidth={2} dot={false} />
                         <Line type="monotone" dataKey="val_loss_m2" name="M2 Val Loss" stroke="#262626" strokeDasharray="5 5" dot={false} />
                       </LineChart>
                     </ResponsiveContainer>
                   </div>
                 </CardContent>
               </Card>
            </div>

            <div className="lg:col-span-1">
              <Card className="h-full">
                <CardHeader>
                  <CardTitle>Model Comparison</CardTitle>
                  <CardDescription>Key technical specifications</CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow className="hover:bg-transparent border-neutral-800">
                        <TableHead className="text-neutral-400">Metric</TableHead>
                        <TableHead className="text-white">DenseNet</TableHead>
                        <TableHead className="text-neutral-400">MobileNet</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow className="border-neutral-800 hover:bg-neutral-900/50">
                        <TableCell className="font-medium text-neutral-300">Accuracy</TableCell>
                        <TableCell className="text-white font-bold">{metricsM1 ? (metricsM1.accuracy * 100).toFixed(1) : "..."}%</TableCell>
                        <TableCell className="text-neutral-400">{metricsM2 ? (metricsM2.accuracy * 100).toFixed(1) : "..."}%</TableCell>
                      </TableRow>
                      <TableRow className="border-neutral-800 hover:bg-neutral-900/50">
                        <TableCell className="font-medium text-neutral-300">Params</TableCell>
                        <TableCell className="text-white">{metricsM1?.params}</TableCell>
                        <TableCell className="text-neutral-400 font-bold">{metricsM2?.params}</TableCell>
                      </TableRow>
                      <TableRow className="border-neutral-800 hover:bg-neutral-900/50">
                        <TableCell className="font-medium text-neutral-300">Inference</TableCell>
                        <TableCell className="text-white">{metricsM1?.inference}</TableCell>
                        <TableCell className="text-neutral-400 font-bold">{metricsM2?.inference}</TableCell>
                      </TableRow>
                      <TableRow className="border-neutral-800 hover:bg-neutral-900/50">
                        <TableCell className="font-medium text-neutral-300">Size</TableCell>
                        <TableCell className="text-white">{metricsM1?.size}</TableCell>
                        <TableCell className="text-neutral-400 font-bold">{metricsM2?.size}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
