"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";
import { Cpu, HardDrive, Loader2 } from "lucide-react";

const hourlyDataMock = Array.from({ length: 24 }).map((_, i) => ({
  time: `${i}:00`,
  alerts: Math.floor(Math.random() * 3) + (i > 9 && i < 18 ? 2 : 0),
}));

export default function StatsCharts() {
  const [chartData, setChartData] = useState<any[]>([]);
  const [systemStats, setSystemStats] = useState<any>({ cpu: 0, ram: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/stats`);
        if (res.ok) {
          const data = await res.json();
          const today = new Date();
          const formatted = [];
          for (let i = 6; i >= 0; i--) {
            const d = new Date(today);
            d.setDate(today.getDate() - i);
            const dayLabel = ["日", "一", "二", "三", "四", "五", "六"][d.getDay()];
            const val = data.weekly_data[6 - i] || 0;
            formatted.push({
              name: `周${dayLabel}`,
              thefts: val,
              falseAlarms: Math.max(0, Math.floor(val * 0.15))
            });
          }
          setChartData(formatted);
          setSystemStats({ cpu: data.cpu_load, ram: data.ram_load });
        }
      } catch (err) {
        console.error("Stats fetch error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-panel p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-lg bg-blue-100 text-blue-600">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-medium text-foreground/75">CPU 使用率</h4>
              <p className="text-2xl font-bold">{systemStats.cpu}%</p>
            </div>
          </div>
          <div className="w-32 bg-slate-200 h-2 rounded-full overflow-hidden">
            <div 
              className="bg-blue-500 h-full transition-all duration-500" 
              style={{ width: `${systemStats.cpu}%` }}
            ></div>
          </div>
        </div>

        <div className="glass-panel p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-lg bg-purple-100 text-purple-600">
              <HardDrive className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-medium text-foreground/75">内存占用</h4>
              <p className="text-2xl font-bold">{systemStats.ram}%</p>
            </div>
          </div>
          <div className="w-32 bg-slate-200 h-2 rounded-full overflow-hidden">
            <div 
              className="bg-purple-500 h-full transition-all duration-500" 
              style={{ width: `${systemStats.ram}%` }}
            ></div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-5">
          <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
            近7日安全告警统计
          </h3>
          <div className="h-64 flex items-center justify-center">
            {loading ? (
              <Loader2 className="w-8 h-8 animate-spin text-brand" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.15)" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip 
                    cursor={{ fill: "rgba(37,99,235,0.06)" }}
                    contentStyle={{ backgroundColor: "#fff", border: "1px solid rgba(15,23,42,0.1)", borderRadius: "8px", boxShadow: "0 4px 12px rgba(15,23,42,0.08)" }}
                  />
                  <Bar dataKey="thefts" fill="#dc2626" radius={[4, 4, 0, 0]} name="可疑行为" />
                  <Bar dataKey="falseAlarms" fill="#2563eb" radius={[4, 4, 0, 0]} name="已排除" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="glass-panel p-5">
          <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
            今日告警趋势
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={hourlyDataMock}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.15)" vertical={false} />
                <XAxis dataKey="time" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} interval={3} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#fff", border: "1px solid rgba(15,23,42,0.1)", borderRadius: "8px", boxShadow: "0 4px 12px rgba(15,23,42,0.08)" }}
                />
                <Line type="monotone" dataKey="alerts" stroke="#2563eb" strokeWidth={3} dot={false} activeDot={{ r: 6, fill: "#2563eb" }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
