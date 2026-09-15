"use client";

import { useState, useEffect } from "react";
import { Save, Bell, Mail, Send, Loader2, CheckCircle2 } from "lucide-react";

export default function SettingsPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const [settings, setSettings] = useState({
    emailEnabled: false,
    smtpServer: "smtp.gmail.com",
    smtpPort: "587",
    senderEmail: "",
    senderPassword: "",
    receiverEmail: "",
    telegramEnabled: false,
    telegramBotToken: "",
    telegramChatId: "",
    roiPoints: [] as number[][],
    showHeatmap: false
  });

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/settings`)
      .then(res => res.json())
      .then(data => {
        setSettings(prev => ({
          ...prev,
          ...data,
          senderPassword: data.senderPassword || "********",
          telegramBotToken: data.telegramBotToken || "********"
        }));
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch settings", err);
        setLoading(false);
      });
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value
    }));
  };

  const handleSave = async () => {
    setSaving(true);
    setMessage("");
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings)
      });
      const data = await response.json();
      if (response.ok) {
        setMessage("设置保存成功！");
      } else {
        setMessage(data.message || "保存设置失败。");
      }
    } catch (err) {
      console.error(err);
      setMessage("网络错误，请确认后端已启动。");
    } finally {
      setSaving(false);
      setTimeout(() => setMessage(""), 3000);
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
    <div className="max-w-4xl mx-auto pb-10">
      <header className="mb-8">
        <h2 className="text-3xl font-bold tracking-tight mb-2">通知设置</h2>
        <p className="text-foreground/60">配置接收安全告警的方式和渠道。</p>
      </header>

      <div className="space-y-6">
        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded bg-blue-500/20 text-blue-500">
              <Send className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold">Telegram 集成</h3>
          </div>
          <p className="text-sm text-foreground/60 mb-6">
            直接在 Telegram 应用中接收实时图片和说明告警。
          </p>

          <form className="space-y-4" onSubmit={e => e.preventDefault()}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">机器人令牌</label>
                <input 
                  type="password" 
                  name="telegramBotToken"
                  value={settings.telegramBotToken}
                  onChange={handleChange}
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground" 
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">聊天 ID</label>
                <input 
                  type="text" 
                  name="telegramChatId"
                  value={settings.telegramChatId}
                  onChange={handleChange}
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground" 
                />
              </div>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <input 
                type="checkbox" 
                id="enable-telegram" 
                name="telegramEnabled"
                checked={settings.telegramEnabled}
                onChange={handleChange}
                className="rounded bg-white border-glass-border text-brand" 
              />
              <label htmlFor="enable-telegram" className="text-sm text-foreground/80">启用 Telegram 告警</label>
            </div>
          </form>
        </div>

        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded bg-purple-500/20 text-purple-500">
              <Mail className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold">邮件（SMTP）设置</h3>
          </div>
          <p className="text-sm text-foreground/60 mb-6">
            通过邮件接收详细的文字告警和报告。
          </p>

          <form className="space-y-4" onSubmit={e => e.preventDefault()}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">SMTP 服务器</label>
                <input 
                  type="text" 
                  name="smtpServer"
                  value={settings.smtpServer}
                  onChange={handleChange}
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground" 
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">端口</label>
                <input 
                  type="number" 
                  name="smtpPort"
                  value={settings.smtpPort}
                  onChange={handleChange}
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground" 
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">发件人邮箱</label>
                <input 
                  type="email" 
                  name="senderEmail"
                  value={settings.senderEmail}
                  onChange={handleChange}
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground" 
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">密码 / 应用专用密码</label>
                <input 
                  type="password" 
                  name="senderPassword"
                  value={settings.senderPassword}
                  onChange={handleChange}
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground" 
                />
              </div>
              <div className="space-y-2 md:col-span-2">
                <label className="text-sm font-medium text-foreground/80">收件人邮箱</label>
                <input 
                  type="text" 
                  name="receiverEmail"
                  value={settings.receiverEmail}
                  onChange={handleChange}
                  className="w-full bg-white border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground" 
                />
              </div>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <input 
                type="checkbox" 
                id="enable-email" 
                name="emailEnabled"
                checked={settings.emailEnabled}
                onChange={handleChange}
                className="rounded bg-white border-glass-border text-brand" 
              />
              <label htmlFor="enable-email" className="text-sm text-foreground/80">启用邮件告警</label>
            </div>
          </form>
        </div>

        <div className="flex items-center justify-between pt-4">
          <div>
            {message && (
              <span className={`flex items-center gap-2 text-sm font-medium px-3 py-1.5 rounded ${message.includes('成功') ? 'text-green-600 bg-green-500/10' : 'text-danger bg-danger/10'}`}>
                {message.includes('成功') && <CheckCircle2 className="w-4 h-4" />}
                {message}
              </span>
            )}
          </div>
          <button 
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 bg-brand hover:bg-brand/90 disabled:opacity-50 text-white px-6 py-2 rounded-lg font-medium transition-colors cursor-pointer"
          >
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            {saving ? "保存中..." : "保存更改"}
          </button>
        </div>
      </div>
    </div>
  );
}
