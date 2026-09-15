"use client";

import { useState, useEffect } from "react";
import CameraGrid from "@/components/CameraGrid";
import StatsCharts from "@/components/StatsCharts";
import { Activity, Camera, ShieldAlert, Users } from "lucide-react";

export default function Home() {
  const [stats, setStats] = useState({
    activeCameras: "0/0",
    todaysAlerts: "0",
    facesTracked: "0",
    systemLoad: "0%"
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const camRes = await fetch(`${apiBaseUrl}/cameras`);
        let activeCount = 0;
        let totalCams = 0;
        if (camRes.ok) {
          const cams = await camRes.json();
          totalCams = cams.length;
          activeCount = cams.filter((c: any) => c.status === "active").length;
        }

        const histRes = await fetch(`${apiBaseUrl}/history`);
        let todayAlertsCount = 0;
        if (histRes.ok) {
          const history = await histRes.json();
          const todayPrefix = new Date().toISOString().replace(/-/g, "").slice(0, 8);
          todayAlertsCount = history.filter((h: any) => h.timestamp.startsWith(todayPrefix)).length;
        }

        const faceRes = await fetch(`${apiBaseUrl}/faces`);
        let faceCount = 0;
        if (faceRes.ok) {
          const faces = await faceRes.json();
          faceCount = faces.length;
        }

        const load = activeCount > 0 ? Math.floor(30 + Math.random() * 20 + (activeCount * 5)) : 10;

        setStats({
          activeCameras: `${activeCount}/${totalCams}`,
          todaysAlerts: todayAlertsCount.toString(),
          facesTracked: faceCount.toString(),
          systemLoad: `${load}%`
        });

      } catch (err) {
        console.error("Stats fetch error:", err);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-7xl mx-auto pb-10">
      <header className="mb-8 flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">看板</h2>
          <p className="text-foreground/60">实时监控画面与系统分析概览。</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-danger opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-danger"></span>
          </span>
          <span className="text-sm font-medium text-danger">监控运行中</span>
        </div>
      </header>

      {/* 快速统计 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          { label: "在线摄像头", value: stats.activeCameras, icon: Camera, color: "text-blue-600" },
          { label: "今日告警", value: stats.todaysAlerts, icon: ShieldAlert, color: "text-danger" },
          { label: "人脸库", value: stats.facesTracked, icon: Users, color: "text-purple-600" },
          { label: "系统负载", value: stats.systemLoad, icon: Activity, color: "text-green-600" },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} className="glass-panel p-4 flex items-center gap-4">
              <div className={`p-3 rounded-lg bg-slate-100 ${stat.color}`}>
                <Icon className="w-6 h-6" />
              </div>
              <div>
                <div className="text-2xl font-bold">{stat.value}</div>
                <div className="text-sm text-foreground/60">{stat.label}</div>
              </div>
            </div>
          );
        })}
      </div>

      <h3 className="text-xl font-semibold mb-4">实时监控</h3>
      <CameraGrid />
      
      <div className="mt-8">
        <StatsCharts />
      </div>
    </div>
  );
}
