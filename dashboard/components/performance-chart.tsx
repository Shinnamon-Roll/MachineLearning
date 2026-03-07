"use client";

import dynamic from "next/dynamic";
import { ApexOptions } from "apexcharts";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

const ReactApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface PerformanceChartProps {
  type: "line" | "area" | "bar" | "pie" | "donut" | "radar";
  series: any[];
  options?: ApexOptions;
  height?: number;
  yAxisFormat?: "percentage" | "number";
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({
  type,
  series,
  options,
  height = 350,
  yAxisFormat
}) => {
  const { theme, resolvedTheme } = useTheme();
  const [chartOptions, setChartOptions] = useState<ApexOptions>({});

  useEffect(() => {
    const isDark = resolvedTheme === "dark";
    const textColor = isDark ? "#f8fafc" : "#0f172a"; // slate-50 : slate-900
    const gridColor = isDark ? "#334155" : "#e2e8f0"; // slate-700 : slate-200

    const newOptions: ApexOptions = {
      chart: {
        toolbar: { show: false },
        background: "transparent",
        foreColor: textColor,
      },
      theme: {
        mode: isDark ? "dark" : "light",
        monochrome: {
          enabled: true,
          color: isDark ? "#3b82f6" : "#2563eb", // blue-500 : blue-600
          shadeTo: isDark ? "dark" : "light",
          shadeIntensity: 0.65,
        },
      },
      dataLabels: { enabled: false },
      stroke: { curve: "smooth", width: 2 },
      grid: { borderColor: gridColor },
      xaxis: {
        labels: { style: { colors: textColor } },
      },
      yaxis: {
        labels: {
          style: { colors: textColor },
          formatter: yAxisFormat === "percentage" ? (val: number) => val + "%" : undefined,
        },
      },
      legend: {
        labels: { colors: textColor },
      },
      tooltip: {
        theme: isDark ? "dark" : "light",
        y: yAxisFormat === "percentage" ? {
          formatter: (val: number) => val + "%"
        } : (options?.tooltip?.y ? { ...options.tooltip.y } : undefined),
      },
      ...options, // Allow overriding
    };
    setChartOptions(newOptions);
  }, [resolvedTheme, options, yAxisFormat]);

  return (
    <div className="w-full h-full min-h-[200px]">
      <ReactApexChart
        options={chartOptions}
        series={series}
        type={type}
        height={height}
      />
    </div>
  );
};

export default PerformanceChart;
