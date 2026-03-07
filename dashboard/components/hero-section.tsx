"use client";
/* eslint-disable @next/next/no-img-element */
import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileUp, CheckCircle, AlertCircle, Scan } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export default function HeroSection() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "processing" | "result">("idle");
  const [result, setResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      processImage(selectedFile);
    }
  };

  const processImage = async (file: File) => {
    setStatus("processing");
    // Simulate AI Processing
    setTimeout(() => {
      setStatus("result");
      setResult({
        type: "Salmon",
        confidence: 98.5,
        details: "Atlantic Salmon",
      });
    }, 2000);
  };

  const reset = () => {
    setFile(null);
    setStatus("idle");
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="relative w-full min-h-[600px] flex flex-col items-center justify-center overflow-hidden rounded-md bg-neutral-950 border border-neutral-800">
      {/* Background Effect (Simplified Aceternity Grid) */}
      <div className="absolute inset-0 w-full h-full bg-neutral-950 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,120,120,0.3),rgba(255,255,255,0))] pointer-events-none" />
      <div className="absolute inset-0 h-full w-full bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      <div className="z-10 flex flex-col items-center text-center space-y-6 max-w-4xl p-6">
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl md:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-white to-neutral-400"
        >
          Salmon & Trout Analysis
        </motion.h1>
        <p className="text-neutral-400 text-lg max-w-lg mx-auto">
          Deep Learning-based classification and freshness grading system using Improved DenseNet121.
        </p>

        <Card className="w-full max-w-md bg-neutral-900/50 border-neutral-800 backdrop-blur-sm p-8 relative overflow-hidden group">
          <AnimatePresence mode="wait">
            {status === "idle" && (
              <motion.div
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center space-y-4 cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="p-4 rounded-full bg-neutral-800 group-hover:bg-neutral-700 transition-colors">
                  <Upload className="w-8 h-8 text-neutral-400" />
                </div>
                <div className="text-center">
                  <p className="font-medium text-neutral-200">Click to upload or drag and drop</p>
                  <p className="text-sm text-neutral-500">SVG, PNG, JPG (max 800x400px)</p>
                </div>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </motion.div>
            )}

            {status === "processing" && (
              <motion.div
                key="processing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center space-y-6 w-full py-8"
              >
                <div className="relative w-48 h-48 rounded-lg overflow-hidden border border-neutral-700">
                  {file && (
                    <img 
                      src={URL.createObjectURL(file)} 
                      alt="Analysis Preview" 
                      className="w-full h-full object-cover opacity-80" 
                    />
                  )}
                  {/* Scanning Line Animation */}
                  <motion.div
                    className="absolute top-0 left-0 w-full h-1 bg-white shadow-[0_0_20px_rgba(255,255,255,0.8)]"
                    animate={{ top: ["0%", "100%", "0%"] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Scan className="w-12 h-12 text-white animate-pulse" />
                  </div>
                </div>
                <p className="text-neutral-300 font-mono animate-pulse">Analyzing image features...</p>
              </motion.div>
            )}

            {status === "result" && result && (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="w-full space-y-6"
              >
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-semibold text-white">Analysis Result</h3>
                  <Button variant="ghost" size="sm" onClick={reset}>Analyze New</Button>
                </div>
                
                <div className="grid grid-cols-1 gap-4">
                  <div className="bg-neutral-800/50 p-6 rounded-lg border border-neutral-700 text-center">
                    <p className="text-sm text-neutral-400 mb-2">Predicted Class</p>
                    <p className="text-3xl font-bold text-white flex items-center justify-center gap-2">
                      <CheckCircle className="w-6 h-6 text-green-500" />
                      {result.type}
                    </p>
                    <p className="text-sm text-neutral-500 mt-2">{result.details}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-400">Confidence Score</span>
                    <span className="text-white font-mono">{result.confidence}%</span>
                  </div>
                  <div className="h-2 w-full bg-neutral-800 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence}%` }}
                      transition={{ duration: 0.8, delay: 0.2 }}
                      className="h-full bg-white"
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
      </div>
    </div>
  );
}
