"use client";

import dynamic from "next/dynamic";
import { ApexOptions } from "apexcharts";

const ReactApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface PerformanceChartProps {
  type: "line" | "area" | "bar" | "pie" | "donut" | "radar";
  series: any[];
  options?: ApexOptions;
  height?: number;
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({
  type,
  series,
  options,
  height = 350,
}) => {
  const defaultOptions: ApexOptions = {
    chart: {
      toolbar: {
        show: false,
      },
      background: "transparent",
    },
    theme: {
      mode: "dark",
      palette: "palette1",
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      curve: "smooth",
    },
    grid: {
      borderColor: "#334155",
    },
    xaxis: {
      labels: {
        style: {
          colors: "#94a3b8",
        },
      },
    },
    yaxis: {
      labels: {
        style: {
          colors: "#94a3b8",
        },
      },
    },
    legend: {
      labels: {
        colors: "#94a3b8",
      },
    },
    tooltip: {
      theme: "dark",
    },
    ...options,
  };

  return (
    <div className="w-full h-full min-h-[300px]">
      <ReactApexChart
        options={defaultOptions}
        series={series}
        type={type}
        height={height}
      />
    </div>
  );
};

export default PerformanceChart;
