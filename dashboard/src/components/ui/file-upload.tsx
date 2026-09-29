"use client";
import { cn } from "@/lib/utils";
import React, { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload } from "lucide-react";

export const FileUpload = ({
  onChange,
  previewUrl,
  scanning = false,
}: {
  onChange?: (files: File[]) => void;
  previewUrl?: string | null;
  scanning?: boolean;
}) => {
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (newFiles: File[]) => {
    setFile(newFiles[0]);
    onChange?.(newFiles);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    handleFileChange(Array.from(e.target.files));
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const [isDragging, setIsDragging] = useState(false);

  return (
    <div className="w-full">
      <div
        className={cn(
          "min-h-72 flex items-center justify-center p-10 group/file rounded-none cursor-pointer w-full relative overflow-hidden",
          "border border-dashed border-neutral-700 transition duration-500",
          isDragging ? "bg-neutral-900 border-white shadow-[0_0_15px_rgba(255,255,255,0.2)]" : "bg-black hover:bg-neutral-900"
        )}
        onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileChange(Array.from(e.dataTransfer.files));
                e.dataTransfer.clearData();
            }
        }}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          id="file-upload-handle"
          type="file"
          accept="image/*"
          onChange={onFileChange}
          // input.click() bubbles to the wrapper's onClick; stop it so the picker opens once
          onClick={(e) => e.stopPropagation()}
          className="hidden"
        />
        
        {previewUrl && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={previewUrl} alt="Uploaded fish" className="absolute inset-0 w-full h-full object-contain bg-black" />
        )}

        <div className={cn("relative z-10 flex flex-col items-center justify-center text-center", previewUrl && "self-end")}>
          {!previewUrl && <Upload className="w-10 h-10 text-neutral-400 mb-4 group-hover/file:text-white transition" />}
          <p className={cn("font-bold text-neutral-300", previewUrl ? "text-xs bg-black/70 px-3 py-1" : "text-lg mt-2")}>
            {file ? `${file.name} · click to change` : "Drag & Drop or Click to Upload"}
          </p>
          {!file && <p className="font-normal text-neutral-500 text-sm mt-2">JPG or PNG of a salmon or trout</p>}
        </div>

        {/* Scanning effect while the models run */}
        {scanning && (
          <motion.div
            initial={{ top: 0 }}
            animate={{ top: "100%" }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="absolute left-0 right-0 h-[2px] bg-white shadow-[0_0_20px_rgba(255,255,255,0.8)] z-20"
          />
        )}
      </div>
    </div>
  );
};
