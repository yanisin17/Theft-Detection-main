"use client";

import { useState, useEffect, useRef } from "react";
import { Camera, Maximize2, AlertTriangle, WifiOff } from "lucide-react";

interface CameraFeed {
  camera_id: string;
  name: string;
  data: string;
}

interface AlertData {
  id: string;
  message: string;
  timestamp: string;
  camera_id: string;
}

interface WsPayload {
  type: string;
  cameras: CameraFeed[];
  alert: AlertData | null;
  audio: string | null;
}

export default function CameraGrid() {
  const [cameras, setCameras] = useState<CameraFeed[]>([]);
  const [alertCam, setAlertCam] = useState<string | null>(null);
  const [alertMessage, setAlertMessage] = useState<string>("");
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const alertTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const playSiren = () => {
    try {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextClass) return;
      const ctx = new AudioContextClass();
      
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      
      osc.type = "sine";
      const now = ctx.currentTime;
      
      osc.frequency.setValueAtTime(580, now);
      osc.frequency.linearRampToValueAtTime(950, now + 0.35);
      osc.frequency.linearRampToValueAtTime(580, now + 0.7);
      osc.frequency.linearRampToValueAtTime(950, now + 1.05);
      osc.frequency.linearRampToValueAtTime(580, now + 1.4);
      
      osc.connect(gain);
      gain.connect(ctx.destination);
      
      gain.gain.setValueAtTime(0.2, now);
      gain.gain.linearRampToValueAtTime(0.2, now + 1.2);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 1.4);
      
      osc.start(now);
      osc.stop(now + 1.4);
    } catch (e) {
      console.error("Siren error:", e);
    }
  };

  useEffect(() => {
    let ws: WebSocket;
    let reconnectInterval: NodeJS.Timeout;

    const connectWebSocket = () => {
      const wsUrl = process.env.NEXT_PUBLIC_API_URL ? process.env.NEXT_PUBLIC_API_URL.replace("http", "ws") + "/ws" : "ws://localhost:8000/ws";
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log("WebSocket connected");
      };

      ws.onmessage = (event) => {
        try {
          const payload: WsPayload = JSON.parse(event.data);
          if (payload.type === "multi_frame") {
            setCameras(payload.cameras);
            
            if (payload.alert) {
              setAlertCam(payload.alert.camera_id);
              setAlertMessage(payload.alert.message);
              playSiren();
              
              if (alertTimeoutRef.current) {
                clearTimeout(alertTimeoutRef.current);
              }
              alertTimeoutRef.current = setTimeout(() => {
                setAlertCam(null);
                setAlertMessage("");
              }, 3000);
            }
          }
        } catch (err) {
          console.error("Error parsing WS data", err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log("WebSocket disconnected. Reconnecting...");
        reconnectInterval = setTimeout(connectWebSocket, 3000);
      };
      
      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        ws.close();
      };
    };

    connectWebSocket();

    return () => {
      clearTimeout(reconnectInterval);
      if (alertTimeoutRef.current) clearTimeout(alertTimeoutRef.current);
      if (ws) ws.close();
    };
  }, []);

  return (
    <div className="mb-6">
      {!isConnected && (
        <div className="mb-4 p-3 bg-red-50 border border-red-300 text-red-700 rounded flex items-center gap-2 text-sm">
          <WifiOff className="w-5 h-5" />
          <span>已断开连接，请确认后端服务是否运行，正在重连...</span>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {cameras.length === 0 && isConnected ? (
          <div className="col-span-1 lg:col-span-2 text-center p-10 text-foreground/50 border border-glass-border border-dashed rounded-lg">
            暂无在线摄像头，请在摄像头页面添加。
          </div>
        ) : (
          cameras.map((cam) => {
            const isAlerting = alertCam === cam.camera_id;
            return (
              <div 
                key={cam.camera_id} 
                className={`glass-panel overflow-hidden relative group transition-all duration-300 ${
                  isAlerting ? "alert-pulse ring-2 ring-danger" : ""
                }`}
              >
                <div className="absolute top-0 left-0 right-0 glass-header p-2 flex justify-between items-center z-10">
                  <div className="flex items-center gap-2">
                    <Camera className={`w-4 h-4 ${isAlerting ? "text-danger" : "text-brand"}`} />
                    <span className="text-sm font-semibold">{cam.name}</span>
                  </div>
                  <div className="flex gap-2 items-center">
                    {isAlerting && (() => {
                      const m = alertMessage.match(/^BEHAVIOR:\s*(.+?)\s*\(([0-9.]+)\)$/);
                      if (m) {
                        const btype = m[1];
                        const conf = parseFloat(m[2]);
                        const pct = Math.round(conf * 100);
                        const cls = conf >= 0.7
                          ? "bg-danger/15 text-danger border border-danger/25"
                          : conf >= 0.5
                          ? "bg-orange-100 text-orange-600 border border-orange-300"
                          : "bg-green-100 text-green-700 border border-green-300";
                        return (
                          <span className={`flex items-center gap-1.5 text-xs font-bold animate-pulse px-2 py-0.5 rounded ${cls}`}>
                            <AlertTriangle className="w-3 h-3" />
                            <span>{btype}</span>
                            <span className="px-1.5 py-0.5 bg-slate-700 rounded text-[10px] font-mono text-white">{pct}%</span>
                          </span>
                        );
                      }
                      return (
                        <span className="flex items-center gap-1 text-xs text-danger font-bold animate-pulse bg-danger/15 px-2 py-0.5 rounded border border-danger/25">
                          <AlertTriangle className="w-3 h-3" />
                          {alertMessage}
                        </span>
                      );
                    })()}
                    <button className="p-1 hover:bg-slate-200/60 rounded transition-colors">
                      <Maximize2 className="w-4 h-4 text-slate-500" />
                    </button>
                  </div>
                </div>
                
                <div className="aspect-video bg-slate-200 relative flex items-center justify-center overflow-hidden">
                  <img 
                    src={`data:image/jpeg;base64,${cam.data}`} 
                    alt={cam.name}
                    className="w-full h-full object-contain"
                  />
                  {isAlerting && (
                    <div className="absolute inset-0 border-4 border-danger/60 z-20 pointer-events-none"></div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
