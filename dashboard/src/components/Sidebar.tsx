"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, History, Settings, Users, Video, ScanSearch } from "lucide-react";

export default function Sidebar() {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "看板", icon: LayoutDashboard },
    { href: "/cameras", label: "摄像头", icon: Video },
    { href: "/analyze", label: "离线分析", icon: ScanSearch },
    { href: "/faces", label: "人脸录入", icon: Users },
    { href: "/history", label: "告警历史", icon: History },
    { href: "/settings", label: "设置", icon: Settings },
  ];

  return (
    <aside className="w-64 glass-panel border-l-0 rounded-l-none min-h-screen flex flex-col p-4">
      <div className="flex items-center mb-10 px-2 mt-4">
        <h1 className="text-xl font-bold tracking-wider">
          <span className="text-brand">Theft</span>Guard
        </h1>
      </div>

      <nav className="flex-1 space-y-2">
        {links.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;

          return (
            <Link
              key={link.href}
              href={link.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                isActive
                  ? "bg-brand/15 text-brand font-medium border border-brand/25 shadow-[0_0_12px_rgba(37,99,235,0.1)]"
                  : "text-foreground/70 hover:bg-slate-100 hover:text-foreground"
              }`}
            >
              <Icon className="w-5 h-5" />
              {link.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto pt-8 pb-4">
        <div className="px-4 py-3 rounded-lg bg-slate-100 border border-glass-border">
          <div className="text-xs text-foreground/50 uppercase tracking-wider mb-1">系统状态</div>
          <div className="flex items-center gap-2 text-sm text-green-600">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            在线运行中
          </div>
        </div>
      </div>
    </aside>
  );
}
