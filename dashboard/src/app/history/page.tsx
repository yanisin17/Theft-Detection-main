"use client";

import { useState, useEffect } from "react";
import { Search, Download, ExternalLink, Calendar, Loader2, Image as ImageIcon, Trash2, Play, X } from "lucide-react";

interface AlertHistory {
  id: string;
  message: string;
  timestamp: string;
  image_path: string;
  confidence?: number | null;
  behavior_type?: string | null;
  video_path?: string | null;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<AlertHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState("全部事件类型");
  const [playingAlertId, setPlayingAlertId] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/history`);
        if (response.ok) {
          const data = await response.json();
          setHistory(data);
        }
      } catch (err) {
        console.error("Failed to fetch history:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  // ESC 关闭弹窗
  useEffect(() => {
    if (!playingAlertId) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPlayingAlertId(null);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [playingAlertId]);

  const formatTime = (ts: string) => {
    // 兼容两种格式：YYYYMMDD_HHMMSS（15位）和 YYYYMMDD_HHMMSS_microseconds（22位，微秒防覆盖）
    const m = ts?.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:_(\d{6}))?$/);
    if (!m) return ts;
    const [, year, month, day, hour, min, sec] = m;
    return `${year}-${month}-${day} ${hour}:${min}:${sec}`;
  };

  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const filteredHistory = history.filter((event) => {
    const matchesSearch = 
      event.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      event.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      event.image_path.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (event.behavior_type || "").toLowerCase().includes(searchQuery.toLowerCase());

    let matchesType = true;
    const behaviorType = (event.behavior_type || "").toLowerCase();
    if (selectedType === "可疑行为") {
      matchesType = event.message.includes("SUSPICION") || event.message.includes("RESTRICTED") || event.message.includes("CRIMINAL") || event.message.includes("BEHAVIOR:");
    } else if (selectedType === "黑名单人脸") {
      matchesType = event.message.includes("BLACKLIST");
    } else if (selectedType === "物品藏匿") {
      matchesType = event.message.includes("THEFT") || event.message.includes("Concealed") ||
        behaviorType.includes("conceal") || behaviorType.includes("hiding") ||
        behaviorType.includes("theft") || behaviorType.includes("shielding");
    }

    return matchesSearch && matchesType;
  });

  const handleExportCSV = () => {
    if (filteredHistory.length === 0) return;
    const headers = ["事件ID", "日期时间", "检测类型", "行为类型", "最高置信度", "证据路径"];
    const rows = filteredHistory.map(event => [
      event.id,
      formatTime(event.timestamp),
      event.message,
      event.behavior_type || "",
      event.confidence != null ? `${Math.round(event.confidence * 100)}%` : "",
      event.image_path
    ]);
    
    const csvContent = "data:text/csv;charset=utf-8,\uFEFF" 
      + [headers.join(","), ...rows.map(r => r.map(val => `"${val.replace(/"/g, '""')}"`).join(","))].join("\n");
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `TheftGuard_Alarmlar_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const playingEvent = history.find((h) => h.id === playingAlertId);

  return (
    <div className="max-w-6xl mx-auto pb-10">
      <header className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">告警历史</h2>
          <p className="text-foreground/60">查看以往的安全事件并导出证据。</p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 px-4 py-2 bg-glass border border-glass-border rounded-lg hover:bg-slate-200/60 transition-colors text-sm font-medium">
            <Calendar className="w-4 h-4" />
            最近7天
          </button>
          <button 
            onClick={handleExportCSV}
            disabled={filteredHistory.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-brand/20 border border-brand/35 hover:bg-brand/30 text-brand rounded-lg transition-colors text-sm font-bold cursor-pointer disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            导出 CSV
          </button>
        </div>
      </header>

      <div className="glass-panel overflow-hidden">
        <div className="p-4 border-b border-glass-border flex gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground/50" />
            <input 
              type="text" 
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="按事件ID、类型或摄像头搜索..." 
              className="w-full bg-white border border-glass-border rounded-lg pl-9 pr-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand text-foreground"
            />
          </div>
          <select 
            value={selectedType}
            onChange={e => setSelectedType(e.target.value)}
            className="bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand text-foreground"
          >
            <option className="bg-white">全部事件类型</option>
            <option className="bg-white">可疑行为</option>
            <option className="bg-white">黑名单人脸</option>
            <option className="bg-white">物品藏匿</option>
          </select>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-100 border-b border-glass-border text-sm text-foreground/70">
                <th className="p-4 font-medium">事件ID</th>
                <th className="p-4 font-medium">日期时间</th>
                <th className="p-4 font-medium">检测类型</th>
                <th className="p-4 font-medium text-center">最高置信度</th>
                <th className="p-4 font-medium">图片路径</th>
                <th className="p-4 font-medium text-center">快照</th>
                <th className="p-4 font-medium text-center">视频</th>
                <th className="p-4 font-medium text-center">操作</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {loading ? (
                <tr>
                  <td colSpan={8} className="p-8 text-center text-foreground/60">
                    <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                    正在加载历史记录...
                  </td>
                </tr>
              ) : filteredHistory.length === 0 ? (
                <tr>
                  <td colSpan={8} className="p-8 text-center text-foreground/60">
                    未找到符合搜索条件的告警记录。
                  </td>
                </tr>
              ) : (
                filteredHistory.map((event) => (
                  <tr key={event.id} className="border-b border-glass-border/50 hover:bg-slate-100 transition-colors">
                    <td className="p-4 font-mono text-brand text-xs">{event.id.slice(0, 8)}...</td>
                    <td className="p-4 text-foreground/80">{formatTime(event.timestamp)}</td>
                    <td className="p-4">
                      <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${
                        event.message.includes('THEFT') || event.message.includes('CRIMINAL') || event.message.startsWith('BEHAVIOR:') ? 'bg-danger/20 text-danger border border-danger/20' : 
                        event.message.includes('BLACKLIST') || event.message.includes('RESTRICTED') ? 'bg-orange-500/20 text-orange-400 border border-orange-500/20' :
                        'bg-blue-500/20 text-blue-400 border border-blue-500/20'
                      }`}>
                        {event.behavior_type || event.message}
                      </span>
                    </td>
                    <td className="p-4 text-center">
                      {event.confidence != null ? (
                        <span className={`inline-block min-w-[52px] px-2 py-1 rounded text-xs font-bold ${
                          event.confidence >= 0.8 ? 'bg-danger/20 text-danger border border-danger/30' :
                          event.confidence >= 0.7 ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                          'bg-yellow-500/15 text-yellow-400 border border-yellow-500/25'
                        }`}>
                          {Math.round(event.confidence * 100)}%
                        </span>
                      ) : (
                        <span className="text-foreground/30 text-xs">—</span>
                      )}
                    </td>
                    <td className="p-4 font-mono text-xs text-foreground/60">{event.image_path}</td>
                    <td className="p-4 text-center">
                      <a 
                        href={`${apiBase}/${event.image_path}`} 
                        target="_blank" 
                        rel="noreferrer" 
                        className="p-1.5 rounded hover:bg-slate-200/60 text-slate-500 hover:text-brand transition-colors inline-block cursor-pointer"
                      >
                        <ImageIcon className="w-4 h-4" />
                      </a>
                    </td>
                    <td className="p-4 text-center">
                      {event.video_path ? (
                        <button
                          onClick={() => setPlayingAlertId(event.id)}
                          className="flex items-center justify-center gap-1.5 px-3 py-1.5 bg-brand/15 border border-brand/25 hover:bg-brand/25 text-brand rounded-lg transition-colors cursor-pointer"
                          title="点击播放证据视频"
                        >
                          <Play className="w-3.5 h-3.5" />
                          <span className="text-xs font-bold">查看证据</span>
                        </button>
                      ) : (
                        <span className="text-foreground/30 text-xs">—</span>
                      )}
                    </td>
                    <td className="p-4 text-center">
                      <button
                        onClick={async () => {
                          if (!confirm("确定删除此告警记录？相关图片和证据视频也会被删除。")) return;
                          try {
                            const res = await fetch(`${apiBase}/history/${event.id}`, { method: "DELETE" });
                            if (res.ok) {
                              setHistory((prev) => prev.filter((h) => h.id !== event.id));
                            }
                          } catch (err) {
                            console.error("删除失败:", err);
                          }
                        }}
                        className="p-1.5 rounded hover:bg-red-100 text-slate-500 hover:text-danger transition-colors inline-block cursor-pointer"
                        title="删除此告警记录"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        
        {!loading && filteredHistory.length > 0 && (
          <div className="p-4 border-t border-glass-border flex items-center justify-between text-sm text-foreground/60">
            <div>共 {filteredHistory.length} 条记录</div>
            <div className="flex gap-1">
              <button className="px-3 py-1 border border-glass-border rounded hover:bg-slate-200/60 disabled:opacity-50" disabled>上一页</button>
              <button className="px-3 py-1 bg-brand text-white rounded font-bold">1</button>
              <button className="px-3 py-1 border border-glass-border rounded hover:bg-slate-200/60 disabled:opacity-50" disabled>下一页</button>
            </div>
          </div>
        )}
      </div>

      {/* 证据视频弹窗 */}
      {playingEvent && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4"
          onClick={() => setPlayingAlertId(null)}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-glass-border">
              <div>
                <h3 className="text-lg font-bold text-foreground">证据视频</h3>
                <p className="text-xs text-foreground/60 mt-0.5">
                  {playingEvent.behavior_type || playingEvent.message} ·{" "}
                  {formatTime(playingEvent.timestamp)} · 前5秒 + 后5秒
                </p>
              </div>
              <button
                onClick={() => setPlayingAlertId(null)}
                className="p-1.5 rounded hover:bg-slate-100 text-slate-500 hover:text-foreground transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="bg-slate-900">
              <img
                key={playingAlertId}
                src={`${apiBase}/api/evidence/stream/${playingAlertId}`}
                alt="证据视频"
                className="w-full max-h-[70vh] object-contain block"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
