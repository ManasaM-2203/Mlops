import React, { useState, useEffect, useMemo, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Languages, Shield, History, Trash2, 
  Network, Info, PlusCircle,
  BarChart3, X, Maximize2, Minimize2
} from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';

interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
  confidence: number;
}

interface TranslateResponse {
  source_text: string;
  translated_text: string;
  entities: Entity[];
  target_lang: string;
}

const API_BASE = 'http://localhost:8000';

const App: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [targetLang, setTargetLang] = useState('hi');
  const [selectedBackend, setSelectedBackend] = useState(() => localStorage.getItem('backend') || 'google');
  const [result, setResult] = useState<TranslateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<Entity[]>([]);
  const [apiStatus, setApiStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  const [manualEntities, setManualEntities] = useState<Entity[]>([]);
  const [transliterateEnabled, setTransliterateEnabled] = useState(true);
  const [showGraph, setShowGraph] = useState(false);
  const [graphExpanded, setGraphExpanded] = useState(false);
  
  // Force-Tagging State
  const [selection, setSelection] = useState<{ text: string, start: number, end: number, x: number, y: number } | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      await axios.get(`${API_BASE}/health`);
      setApiStatus('online');
    } catch {
      setApiStatus('offline');
    }
  };

  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    try {
      const resp = await axios.post(`${API_BASE}/api/translate`, {
        text: inputText,
        target_lang: targetLang,
        backend: selectedBackend,
        manual_entities: manualEntities,
        transliterate: transliterateEnabled
      });
      setResult(resp.data);
      
      // Add new unique entities to history
      const newEntities = resp.data.entities.filter(
        (e: Entity) => !history.some(h => h.text.toLowerCase() === e.text.toLowerCase())
      );
      setHistory(prev => [...newEntities, ...prev]);
    } catch (err) {
      console.error(err);
      alert('Translation failed. Verify API status.');
    } finally {
      setLoading(false);
    }
  };

  const handleTextSelect = () => {
    const el = textareaRef.current;
    if (!el) return;
    
    const start = el.selectionStart;
    const end = el.selectionEnd;
    const text = el.value.substring(start, end);

    if (text && text.trim().length > 0) {
      // Basic position calculation for the popover
      // In a production app, we'd use a more robust way to get selection coordinates
      setSelection({ text, start, end, x: 100, y: 100 }); 
    } else {
      setSelection(null);
    }
  };

  const addManualEntity = (label: string) => {
    if (!selection) return;
    const newEnt: Entity = {
      text: selection.text,
      label,
      start: selection.start,
      end: selection.end,
      confidence: 1.0 // Manual tags are always 100% confident
    };
    setManualEntities(prev => [...prev, newEnt]);
    setSelection(null);
  };

  const clearHistory = () => {
    setHistory([]);
    setManualEntities([]);
  };

  // Clustering Logic
  const clusters = useMemo(() => {
    const map = new Map<string, { label: string, count: number }>();
    history.forEach(ent => {
      const key = `${ent.text.toLowerCase()}|${ent.label}`;
      const existing = map.get(key);
      map.set(key, { label: ent.label, count: (existing?.count || 0) + 1 });
    });
    return Array.from(map.entries()).map(([key, val]) => ({
      text: key.split('|')[0],
      label: val.label,
      count: val.count
    }));
  }, [history]);

  // Graph Data Logic (simple co-occurrence)
  const graphData = useMemo(() => {
    if (!result?.entities) return { nodes: [], links: [] };
    const nodes = result.entities.map((e, i) => ({ id: i, name: e.text, label: e.label }));
    const links: any[] = [];
    
    // Connect entities that appear within 100 chars of each other
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const e1 = result.entities[i];
            const e2 = result.entities[j];
            if (Math.abs(e1.start - e2.start) < 100) {
                links.push({ source: i, target: j, relation: 'NEARBY' });
            }
        }
    }
    return { nodes, links };
  }, [result]);

  return (
    <div className="flex h-screen w-screen overflow-hidden font-sans bg-slate-950 text-white selection:bg-primary/30">
      {/* Sidebar - Clustered Entities */}
      <aside className="w-80 bg-slate-900/40 backdrop-blur-2xl border-r border-white/5 p-6 flex flex-col gap-6 relative z-20">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-primary/20 rounded-xl shadow-lg shadow-primary/10">
            <Shield className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">Turing Tag</h1>
            <span className="text-[10px] text-slate-500 font-black tracking-widest uppercase">Intelligence Unit</span>
          </div>
        </div>

        <div className="flex flex-col gap-2 flex-1 overflow-hidden">
          <div className="flex items-center justify-between text-xs font-bold text-slate-500 uppercase tracking-widest px-1">
            <span className="flex items-center gap-2">
              <BarChart3 className="w-3.5 h-3.5" /> Entity Clusters
            </span>
            <button onClick={clearHistory} className="hover:text-red-400 transition-colors p-1">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
          
          <div className="flex flex-col gap-2 mt-2 overflow-y-auto pr-2 custom-scrollbar">
            <AnimatePresence mode='popLayout'>
              {clusters.map((cluster) => (
                <motion.div
                  key={cluster.text + cluster.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.05] hover:bg-white/[0.07] transition-all group relative cursor-default"
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className={`text-[10px] font-black uppercase tracking-widest ${getLabelColor(cluster.label)}`}>
                      {cluster.label}
                    </span>
                    <span className="bg-white/10 px-1.5 py-0.5 rounded text-[10px] font-bold text-slate-400">
                      {cluster.count}x
                    </span>
                  </div>
                  <p className="text-sm font-semibold text-slate-200 capitalize">{cluster.text}</p>
                </motion.div>
              ))}
            </AnimatePresence>
            {clusters.length === 0 && (
              <div className="flex flex-col items-center justify-center py-20 opacity-20 gap-3 grayscale">
                <History className="w-10 h-10" />
                <p className="text-xs font-bold uppercase tracking-widest">Awaiting Data</p>
              </div>
            )}
          </div>
        </div>

        <div className="pt-6 border-t border-white/5 space-y-4">
          {manualEntities.length > 0 && (
            <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl">
              <p className="text-[10px] font-bold text-amber-500 uppercase mb-2">Active Overrides</p>
              <div className="flex flex-wrap gap-1">
                {manualEntities.map((e, i) => (
                  <span key={i} className="text-[9px] bg-amber-500/20 text-amber-200 px-1.5 py-0.5 rounded">
                    {e.text}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between px-2">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">API Vector</span>
            <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-tighter">
              {apiStatus === 'online' ? (
                <><div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" /> <span className="text-green-400">Stable</span></>
              ) : (
                <><div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" /> <span className="text-red-400">Offline</span></>
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 flex flex-col gap-6 relative">
        <header className="flex items-center justify-between">
          <div>
            <h2 className="text-4xl font-black tracking-tight">Intelligence Dashboard</h2>
            <p className="text-slate-400 mt-1.5 font-medium">Neural Translation with Protected Entity Mapping.</p>
          </div>
          
          <div className="flex items-center gap-4">
            <button 
                onClick={() => setTransliterateEnabled(!transliterateEnabled)}
                className={`flex items-center justify-center gap-2 w-[140px] py-3 rounded-2xl font-bold text-xs transition-all border ${
                    transliterateEnabled 
                        ? 'bg-amber-500/10 border-amber-500/30 text-amber-500' 
                        : 'bg-white/5 border-white/10 text-slate-500 opacity-50'
                }`}
                title="Transliterate entities to target script"
            >
                <Languages className="w-4 h-4" /> 
                {transliterateEnabled ? 'Translit ON' : 'Translit OFF'}
            </button>

            <button 
                onClick={() => setShowGraph(!showGraph)}
                className={`flex items-center justify-center gap-2 w-[220px] py-3 rounded-2xl font-bold text-xs transition-all ${
                    showGraph ? 'bg-primary text-white shadow-xl shadow-primary/20' : 'bg-white/5 text-slate-400 hover:bg-white/10'
                }`}
            >
                <Network className="w-4 h-4" /> 
                {showGraph ? 'Hide Relationship Map' : 'Show Relationship Map'}
            </button>

            <div className="flex items-center gap-2 bg-white/[0.03] p-1.5 rounded-2xl border border-white/5">
                <div className="flex flex-col px-4 border-r border-white/10">
                    <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-0.5">Translation Core</span>
                    <select 
                        value={selectedBackend}
                        onChange={(e) => {
                            setSelectedBackend(e.target.value);
                            localStorage.setItem('backend', e.target.value);
                        }}
                        className="bg-transparent border-none outline-none text-xs font-bold text-primary cursor-pointer"
                    >
                        <option value="google" className="bg-slate-900">Google Cloud</option>
                        <option value="mymemory" className="bg-slate-900">MyMemory Hub</option>
                        <option value="marian" className="bg-slate-900">Marian (Local Llama)</option>
                    </select>
                </div>

                <div className="flex flex-col px-4">
                    <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-0.5">Target Vector</span>
                    <select 
                        value={targetLang}
                        onChange={(e) => setTargetLang(e.target.value)}
                        className="bg-transparent border-none outline-none text-xs font-bold text-white cursor-pointer"
                    >
                        <option value="hi" className="bg-slate-900">Hindi (IN)</option>
                        <option value="es" className="bg-slate-900">Spanish (ES)</option>
                        <option value="fr" className="bg-slate-900">French (FR)</option>
                        <option value="de" className="bg-slate-900">German (DE)</option>
                    </select>
                </div>

                <button 
                    onClick={handleTranslate}
                    disabled={loading}
                    className={`ml-2 flex items-center justify-center w-[180px] py-3.5 rounded-xl font-black text-xs uppercase tracking-widest transition-all ${
                        loading 
                            ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                            : 'bg-primary text-white shadow-2xl shadow-primary/30 hover:brightness-110 active:scale-95'
                    }`}
                >
                    {loading ? 'Processing...' : 'Run Extraction'}
                </button>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1 overflow-hidden relative">
          {/* Source Area */}
          <div className="flex flex-col gap-4 overflow-hidden group">
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] px-2 flex items-center justify-between">
                Source Document
                {selection && <span className="text-primary animate-pulse flex items-center gap-1.5"><PlusCircle className="w-3 h-3"/> Active Selection</span>}
            </h3>
            <div className="relative flex-1 rounded-[2.5rem] bg-white/[0.02] border border-white/[0.05] p-1 transition-all group-focus-within:border-primary/30 shadow-2xl">
              <textarea
                ref={textareaRef}
                value={inputText}
                onMouseUp={handleTextSelect}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste sensitive data here for NER-aware translation..."
                className="w-full h-full p-10 bg-transparent border-none resize-none text-xl leading-relaxed placeholder:text-slate-800 focus:outline-none custom-scrollbar"
              />
              
              {/* Force Tag Menu */}
              <AnimatePresence>
                {selection && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10, scale: 0.9 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="absolute top-10 left-1/2 -translate-x-1/2 bg-slate-900/90 backdrop-blur-xl border border-white/10 p-2 rounded-2xl shadow-2xl z-50 flex gap-2"
                  >
                    <button onClick={() => addManualEntity('per')} className="px-4 py-2 hover:bg-blue-500/20 text-blue-400 rounded-xl text-[10px] font-black uppercase transition-colors">Person</button>
                    <button onClick={() => addManualEntity('loc')} className="px-4 py-2 hover:bg-green-500/20 text-green-400 rounded-xl text-[10px] font-black uppercase transition-colors">Location</button>
                    <button onClick={() => addManualEntity('org')} className="px-4 py-2 hover:bg-purple-500/20 text-purple-400 rounded-xl text-[10px] font-black uppercase transition-colors">Org</button>
                    <div className="w-[1px] bg-white/10 mx-1" />
                    <button onClick={() => setSelection(null)} className="p-2 hover:bg-white/5 rounded-xl"><X className="w-3.5 h-3.5"/></button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Result Area / Graph Panel */}
          <div className="flex flex-col gap-4 overflow-hidden relative">
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] px-2">
                Protected Result
            </h3>
            <div className="relative flex-1 rounded-[2.5rem] bg-white/[0.02] border border-white/[0.05] p-1 shadow-2xl overflow-hidden">
                <AnimatePresence mode="wait">
                    {showGraph ? (
                        <motion.div 
                            key="graph"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 1.05 }}
                            className={`absolute inset-0 bg-slate-950/40 backdrop-blur-md rounded-[2.5rem] z-10 transition-all ${graphExpanded ? 'fixed inset-4 z-[100] m-4 border border-white/10' : ''}`}
                        >
                            <div className="absolute top-6 right-6 z-20 flex gap-2">
                                <button onClick={() => setGraphExpanded(!graphExpanded)} className="p-2 bg-white/5 hover:bg-white/10 rounded-xl transition-colors">
                                    {graphExpanded ? <Minimize2 className="w-4 h-4"/> : <Maximize2 className="w-4 h-4"/>}
                                </button>
                                <button onClick={() => setShowGraph(false)} className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-xl transition-colors">
                                    <X className="w-4 h-4"/>
                                </button>
                            </div>
                            <div className="w-full h-full flex items-center justify-center">
                                {graphData.nodes.length > 0 ? (
                                    <ForceGraph2D
                                        graphData={graphData}
                                        nodeLabel="name"
                                        nodeColor={node => {
                                            const label = (node as any).label.toLowerCase();
                                            if (label === 'per') return '#3b82f6';
                                            if (label === 'loc') return '#22c55e';
                                            if (label === 'org') return '#a855f7';
                                            return '#64748b';
                                        }}
                                        nodeRelSize={8}
                                        linkColor={() => 'rgba(255,255,255,0.15)'}
                                        linkWidth={1.5}
                                        backgroundColor="transparent"
                                        width={graphExpanded ? window.innerWidth - 100 : 600}
                                        height={graphExpanded ? window.innerHeight - 100 : 500}
                                        nodeCanvasObjectMode={() => 'after'}
                                        nodeCanvasObject={(node: any, ctx, globalScale) => {
                                            const label = node.name;
                                            const fontSize = 12 / globalScale;
                                            ctx.font = `${fontSize}px Inter, sans-serif`;
                                            ctx.textAlign = 'center';
                                            ctx.textBaseline = 'middle';
                                            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                                            ctx.fillText(label, node.x, node.y + 12);
                                        }}
                                        linkCanvasObjectMode={() => 'after'}
                                        linkCanvasObject={(link: any, ctx, globalScale) => {
                                            const start = link.source;
                                            const end = link.target;
                                            if (typeof start !== 'object' || typeof end !== 'object') return;
                                            const textPos = {
                                                x: start.x + (end.x - start.x) / 2,
                                                y: start.y + (end.y - start.y) / 2
                                            };
                                            const label = link.relation || 'NEARBY';
                                            const fontSize = 8 / globalScale;
                                            ctx.font = `600 ${fontSize}px Inter, sans-serif`;
                                            ctx.textAlign = 'center';
                                            ctx.textBaseline = 'middle';
                                            const textWidth = ctx.measureText(label).width;
                                            const padding = 2 / globalScale;
                                            ctx.fillStyle = 'rgba(2, 6, 23, 0.8)';
                                            ctx.fillRect(textPos.x - textWidth/2 - padding, textPos.y - fontSize/2 - padding, textWidth + padding*2, fontSize + padding*2);
                                            ctx.fillStyle = 'rgba(148, 163, 184, 1)';
                                            ctx.fillText(label, textPos.x, textPos.y);
                                        }}
                                    />
                                ) : (
                                    <div className="flex flex-col items-center gap-4 opacity-20 capitalize italic">
                                        <Network className="w-16 h-16" />
                                        <p>No entity relationships discovered yet.</p>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div 
                            key="text"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="w-full h-full p-10 overflow-y-auto custom-scrollbar"
                        >
                            {loading ? (
                                <div className="h-full flex flex-col items-center justify-center gap-6">
                                    <div className="w-14 h-14 border-[5px] border-primary/20 border-t-primary rounded-full animate-spin shadow-2xl shadow-primary/20" />
                                    <div className="space-y-4 w-full max-w-sm">
                                        <div className="h-4 bg-white/5 rounded-full animate-pulse w-full"></div>
                                        <div className="h-4 bg-white/5 rounded-full animate-pulse w-3/4 mx-auto"></div>
                                    </div>
                                </div>
                            ) : result ? (
                                <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700">
                                    <p className="text-2xl font-semibold leading-relaxed tracking-tight text-white/90">
                                        {highlightEntities(result.translated_text, result.entities)}
                                    </p>
                                    
                                    <div className="space-y-6 pt-10 border-t border-white/5">
                                        <div className="flex items-center justify-between">
                                            <h4 className="text-[10px] font-black text-primary uppercase tracking-[0.3em]">XAI Audit Log</h4>
                                        </div>
                                        <div className="grid grid-cols-2 gap-3">
                                            {result.entities.map((e, i) => (
                                                <div key={i} className="group flex flex-col bg-white/[0.03] border border-white/[0.05] rounded-2xl p-4 hover:border-primary/30 transition-all cursor-default relative overflow-hidden">
                                                    <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-100 transition-opacity">
                                                        <Info className="w-3 h-3 text-primary" />
                                                    </div>
                                                    <div className="flex items-center gap-2 mb-1.5">
                                                        <span className={`text-[8px] font-black tracking-widest uppercase ${getLabelColor(e.label)}`}>{e.label}</span>
                                                        <span className="text-[8px] font-bold text-slate-500">{(e.confidence * 100).toFixed(1)}% Match</span>
                                                    </div>
                                                    <span className="text-sm font-bold text-slate-200">{e.text}</span>
                                                    <div className="mt-2 w-full h-1 bg-white/5 rounded-full overflow-hidden">
                                                        <motion.div 
                                                            initial={{ width: 0 }} 
                                                            animate={{ width: `${e.confidence * 100}%` }} 
                                                            className={`h-full ${getLabelBg(e.label)}`}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-slate-700 gap-6 opacity-40">
                                    <Shield className="w-20 h-20" />
                                    <div className="text-center">
                                        <p className="text-xl font-bold">Secure Corridor Alpha</p>
                                        <p className="text-sm mt-1">Pending instruction sequence...</p>
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
          </div>
        </div>
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.1); }
      `}</style>
    </div>
  );
};

const getLabelColor = (label: string) => {
    switch(label.toLowerCase()) {
        case 'per': return 'text-blue-400';
        case 'loc': return 'text-green-400';
        case 'org': return 'text-purple-400';
        default: return 'text-slate-400';
    }
};

const getLabelBg = (label: string) => {
    switch(label.toLowerCase()) {
        case 'per': return 'bg-blue-500';
        case 'loc': return 'bg-green-500';
        case 'org': return 'bg-purple-500';
        default: return 'bg-slate-500';
    }
};

const highlightEntities = (text: string, entities: Entity[]) => {
  return (
    <span>
      {text.split(/(__ENT\d+__)/).map((segment, i) => {
        const match = segment.match(/__ENT(\d+)__/);
        if (match) {
          const index = parseInt(match[1]);
          const entity = entities[index];
          return (
            <span 
                key={i} 
                className={`group relative px-2 py-0.5 mx-0.5 rounded-md border cursor-help inline-block transition-all hover:scale-105 ${
                    entity?.label.toLowerCase() === 'per' ? 'bg-blue-500/10 border-blue-500/30 text-blue-400' :
                    entity?.label.toLowerCase() === 'loc' ? 'bg-green-500/10 border-green-500/30 text-green-400' :
                    entity?.label.toLowerCase() === 'org' ? 'bg-purple-500/10 border-purple-500/30 text-purple-400' :
                    'bg-white/10 border-white/20 text-white'
                }`}
            >
              {entity ? entity.text : segment}
              
              {/* XAI Tooltip */}
              {entity && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 p-2 bg-slate-900 border border-white/10 rounded-lg shadow-2xl opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none">
                      <p className="text-[9px] font-black uppercase text-slate-500">{entity.label} Confidence</p>
                      <p className="text-xs font-bold">{(entity.confidence * 100).toFixed(2)}%</p>
                  </div>
              )}
            </span>
          );
        }
        return segment;
      })}
    </span>
  );
};

export default App;
