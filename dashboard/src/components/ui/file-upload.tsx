"use client";
import { cn } from "@/lib/utils";
import React, { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload } from "lucide-react";

export const FileUpload = ({
  onChange,
}: {
  onChange?: (files: File[]) => void;
}) => {
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (newFiles: File[]) => {
    setFile(newFiles[0]);
    onChange && onChange(newFiles);
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
          "p-10 group/file block rounded-none cursor-pointer w-full relative overflow-hidden",
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
          onChange={onFileChange}
          className="hidden"
        />
        
        <div className="flex flex-col items-center justify-center">
          {!file && (
             <Upload className="w-10 h-10 text-neutral-400 mb-4" />
          )}
          <p className="font-bold text-neutral-300 text-lg mt-2">
            {file ? file.name : "Drag & Drop or Click to Upload"}
          </p>
          {!file && (
             <p className="font-normal text-neutral-500 text-sm mt-2">
               Upload Salmon or Trout Image
             </p>
          )}
        </div>

        {/* Scanning Effect */}
        {file && (
          <motion.div
            initial={{ top: 0 }}
            animate={{ top: "100%" }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear",
            }}
            className="absolute left-0 right-0 h-[2px] bg-white shadow-[0_0_20px_rgba(255,255,255,0.8)] z-20"
          />
        )}
        
        {file && (
           <div className="absolute inset-0 bg-black/50 z-10 flex items-center justify-center pointer-events-none">
             {/* Optional: Preview image could go here if we read it */}
           </div>
        )}
      </div>
    </div>
  );
};
