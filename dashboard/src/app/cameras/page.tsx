"use client";

import { useState, useEffect, useRef } from "react";
import { Camera, Plus, Trash2, Edit, Loader2, AlertCircle, CheckCircle, X, RefreshCw, HelpCircle } from "lucide-react";

interface CameraData {
  id: string;
  name: string;
  source: string;
  status: "active" | "error";
  roi_points: number[][];
}

interface CameraFeed {
  camera_id: string;
  name: string;
  data: string;
}

export default function CamerasPage() {
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" } | null>(null);

  const [name, setName] = useState("");
  const [source, setSource] = useState("");

  const [selectedCam, setSelectedCam] = useState<CameraData | null>(null);
  const [roiPoints, setRoiPoints] = useState<number[][]>([]);
  const [activeFrameBase64, setActiveFrameBase64] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchCameras = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/cameras`);
      if (res.ok) {
        const data = await res.json();
        setCameras(data);
      }
    } catch (err) {
      console.error("Error fetching cameras:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCameras();
  }, []);

  useEffect(() => {
    if (!selectedCam) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setActiveFrameBase64(null);
      return;
    }

    const wsUrl = apiBaseUrl.replace("http", "ws") + "/ws";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "multi_frame" && payload.cameras) {
          const matched = payload.cameras.find((c: CameraFeed) => c.camera_id === selectedCam.id);
          if (matched) {
            setActiveFrameBase64(matched.data);
          }
        }
      } catch (err) {
        console.error("Error parsing WS in ROI modal", err);
      }
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [selectedCam]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drawCanvas = () => {
      if (activeFrameBase64) {
        const img = new Image();
        img.src = `data:image/jpeg;base64,${activeFrameBase64}`;
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          drawPolygon();
        };
      } else {
        ctx.fillStyle = "#e2e8f0";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
        ctx.font = "20px var(--font-geist-sans), sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("监控画面加载中...", canvas.width / 2, canvas.height / 2 - 10);
        ctx.font = "14px var(--font-geist-sans), sans-serif";
        ctx.fillText("（请确认后端已启动并正在传输视频流）", canvas.width / 2, canvas.height / 2 + 20);
        
        drawPolygon();
      }
    };

    const drawPolygon = () => {
      if (roiPoints.length === 0) return;

      ctx.beginPath();
      ctx.moveTo(roiPoints[0][0], roiPoints[0][1]);
      for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i][0], roiPoints[i][1]);
      }
      
      ctx.closePath();

      ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
      ctx.fill();
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 3;
      ctx.stroke();

      roiPoints.forEach((pt, index) => {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 6, 0, 2 * Math.PI);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 12px sans-serif";
        ctx.fillText((index + 1).toString(), pt[0] + 10, pt[1] - 5);
      });
    };

    drawCanvas();
  }, [roiPoints, activeFrameBase64, selectedCam]);

  const handleAddCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !source) {
      setMessage({ text: "请填写摄像头名称和输入源。", type: "error" });
      return;
    }

    setSubmitting(true);
    setMessage(null);

    try {
      const res = await fetch(`${apiBaseUrl}/cameras`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, source }),
      });
      const data = await res.json();

      if (res.ok) {
        setMessage({ text: "摄像头添加成功！", type: "success" });
        setName("");
        setSource("");
        await fetchCameras();
      } else {
        setMessage({ text: data.detail || "添加摄像头失败。", type: "error" });
      }
    } catch (err) {
      console.error(err);
      setMessage({ text: "连接错误，请确认后端已启动。", type: "error" });
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteCamera = async (id: string) => {
    if (!confirm("确定要删除这个摄像头流吗？")) return;
    setMessage(null);

    try {
      const res = await fetch(`${apiBaseUrl}/cameras/${id}`, {
        method: "DELETE",
      });
      const data = await res.json();

      if (res.ok) {
        setMessage({ text: "摄像头删除成功。", type: "success" });
        await fetchCameras();
      } else {
        setMessage({ text: data.detail || "删除摄像头失败。", type: "error" });
      }
    } catch (err) {
      console.error(err);
      setMessage({ text: "连接错误。", type: "error" });
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
    const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));

    setRoiPoints(prev => [...prev, [x, y]]);
  };

  const clearRoiPoints = () => {
    setRoiPoints([]);
  };

  const openRoiModal = (cam: CameraData) => {
    setSelectedCam(cam);
    setRoiPoints(cam.roi_points || []);
  };

  const handleSaveRoi = async () => {
    if (!selectedCam) return;
    try {
      const res = await fetch(`${apiBaseUrl}/cameras/${selectedCam.id}/roi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: roiPoints }),
      });
      const data = await res.json();

      if (res.ok && data.status === "success") {
        setMessage({ text: "感兴趣区域（ROI）坐标保存成功！", type: "success" });
        setSelectedCam(null);
        await fetchCameras();
      } else {
        alert(data.detail || "ROI 保存失败。");
      }
    } catch (err) {
      console.error(err);
      alert("保存 ROI 时网络错误。");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <Loader2 className="w-8 h-8 animate-spin text-brand" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto pb-10">
      <header className="mb-8">
        <h2 className="text-3xl font-bold tracking-tight mb-2">摄像头配置管理</h2>
        <p className="text-foreground/60">
          管理 USB 摄像头索引和 RTSP 网络流，并交互式地为每个摄像头定义感兴趣区域（ROI）。
        </p>
      </header>

      {message && (
        <div 
          className={`mb-6 p-4 rounded-lg flex items-center gap-3 border ${
            message.type === "success" 
              ? "bg-green-500/10 border-green-500/30 text-green-600" 
              : "bg-danger/10 border-danger/30 text-danger"
          }`}
        >
          {message.type === "success" ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
          <span className="text-sm font-medium">{message.text}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="glass-panel p-6 sticky top-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded bg-brand/20 text-brand">
                <Camera className="w-5 h-5" />
              </div>
              <h3 className="text-xl font-semibold">连接新摄像头</h3>
            </div>

            <form onSubmit={handleAddCamera} className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">摄像头名称</label>
                <input 
                  type="text" 
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="例如：收银台 A"
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground placeholder:text-foreground/30"
                  required
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-foreground/80">输入源</label>
                  <span className="text-[10px] text-foreground/45 flex items-center gap-1">
                    <HelpCircle className="w-3 h-3" />
                    摄像头索引或 RTSP 地址
                  </span>
                </div>
                <input 
                  type="text" 
                  value={source}
                  onChange={e => setSource(e.target.value)}
                  placeholder="例如：0 或 rtsp://用户名:密码@ip:端口/h264"
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground placeholder:text-foreground/30"
                  required
                />
              </div>

              <button 
                type="submit" 
                disabled={submitting}
                className="w-full mt-4 flex items-center justify-center gap-2 bg-brand hover:bg-brand/90 disabled:opacity-50 text-white py-2 rounded-lg font-medium transition-colors cursor-pointer"
              >
                {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                {submitting ? "正在连接..." : "添加摄像头流"}
              </button>
            </form>
          </div>
        </div>

        <div className="lg:col-span-2">
          <div className="glass-panel p-6">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded bg-purple-500/20 text-purple-500">
                  <Plus className="w-5 h-5" />
                </div>
                <h3 className="text-xl font-semibold">活动摄像头列表</h3>
              </div>
              <button 
                onClick={fetchCameras}
                className="p-2 hover:bg-glass border border-glass-border rounded-lg text-slate-500 hover:text-foreground transition-colors cursor-pointer"
                title="刷新"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>

            {cameras.length === 0 ? (
              <div className="text-center p-12 text-foreground/40 border border-glass-border border-dashed rounded-lg bg-slate-100">
                尚未配置任何活动摄像头。请使用上方表单添加本地摄像头（0）或网络 RTSP 流。
              </div>
            ) : (
              <div className="space-y-4">
                {cameras.map(cam => (
                  <div 
                    key={cam.id}
                    className="glass-panel p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border border-glass-border bg-slate-100"
                  >
                    <div className="flex items-center gap-3">
                      <div 
                        className={`w-10 h-10 rounded-full flex items-center justify-center border ${
                          cam.status === "active" 
                            ? "bg-green-500/10 border-green-500/30 text-green-600" 
                            : "bg-danger/10 border-danger/30 text-danger"
                        }`}
                      >
                        <Camera className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-foreground flex items-center gap-2">
                          {cam.name}
                          <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${
                            cam.status === "active"
                              ? "bg-green-500/10 border-green-500/20 text-green-600"
                              : "bg-danger/10 border-danger/20 text-danger"
                          }`}>
                            {cam.status === "active" ? "在线" : "离线/错误"}
                          </span>
                        </h4>
                        <p className="text-xs text-foreground/50 mt-1 font-mono">来源：{cam.source}</p>
                        <p className="text-xs text-brand font-medium mt-1">
                          {cam.roi_points && cam.roi_points.length > 0 
                            ? `✓ 已配置感兴趣区域（ROI）：${cam.roi_points.length} 个点`
                            : "⚠️ 未配置 ROI，当前监控整个画面。"}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 self-end md:self-auto">
                      <button
                        onClick={() => openRoiModal(cam)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-brand/20 border border-brand/35 text-brand hover:bg-brand/30 rounded-lg text-xs font-semibold transition-colors cursor-pointer"
                      >
                        <Edit className="w-3.5 h-3.5" />
                        配置 ROI
                      </button>
                      <button
                        onClick={() => handleDeleteCamera(cam.id)}
                        className="p-2 text-slate-500 hover:text-danger hover:bg-danger/10 rounded-lg transition-colors cursor-pointer"
                        title="删除摄像头"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {selectedCam && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/30 backdrop-blur-md transition-all duration-300">
          <div className="glass-panel w-full max-w-4xl overflow-hidden border border-glass-border shadow-2xl relative animate-in fade-in zoom-in-95 duration-200">
            <div className="glass-header px-6 py-4 flex justify-between items-center">
              <div>
                <h3 className="text-lg font-bold text-foreground">交互式 ROI 画布绘制工具</h3>
                <p className="text-xs text-foreground/60">{selectedCam.name} ({selectedCam.source})</p>
              </div>
              <button 
                onClick={() => setSelectedCam(null)}
                className="p-1.5 hover:bg-slate-200 rounded-lg text-slate-500 hover:text-foreground transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="p-3 bg-brand/10 border border-brand/20 text-brand rounded-lg text-xs flex items-start gap-2 leading-relaxed">
                <HelpCircle className="w-4 h-4 mt-0.5 shrink-0" />
                <span>
                  <strong>如何绘制 ROI：</strong>在实时画面上用鼠标点击依次放置多边形顶点，连线会自动将这些点按顺序连接起来。至少放置 3 个点即可定义滞留（徘徊）和禁入区域。完成后点击下方的 <strong>保存 ROI 坐标</strong> 按钮。
                </span>
              </div>

              <div className="aspect-video w-full bg-slate-200 rounded-lg overflow-hidden border border-glass-border relative flex items-center justify-center">
                <canvas
                  ref={canvasRef}
                  width={1280}
                  height={720}
                  onClick={handleCanvasClick}
                  className="w-full h-auto aspect-video cursor-crosshair object-contain"
                />
              </div>

              <div className="flex justify-between items-center text-xs text-foreground/50 px-1">
                <span>坐标：[{roiPoints.map(p => `(${p[0]},${p[1]})`).join(", ")}]</span>
                <span>点数：{roiPoints.length}</span>
              </div>
            </div>

            <div className="glass-header border-t border-b-0 px-6 py-4 flex justify-between items-center gap-4">
              <button 
                onClick={clearRoiPoints}
                className="px-4 py-2 border border-glass-border hover:bg-glass rounded-lg text-sm font-semibold transition-colors cursor-pointer text-foreground/80 hover:text-foreground"
              >
                清除所有点
              </button>
              
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => setSelectedCam(null)}
                  className="px-4 py-2 border border-glass-border hover:bg-glass rounded-lg text-sm font-semibold transition-colors cursor-pointer text-foreground/80 hover:text-foreground"
                >
                  取消
                </button>
                <button 
                  onClick={handleSaveRoi}
                  className="px-6 py-2 bg-brand hover:bg-brand/90 text-white rounded-lg text-sm font-bold transition-colors cursor-pointer"
                >
                  保存 ROI 坐标
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
