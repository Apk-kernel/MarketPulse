import React, { useState, useEffect } from 'react';
import { MetadataResponse } from './types/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'forecasting' | 'hierarchy' | 'inventory' | 'monitoring' | 'benchmarks'>('forecasting');
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean>(true);

  useEffect(() => {
    fetch('/api/v1/health')
      .then(res => {
        if (res.ok) {
          setApiOnline(true);
          return fetch('/api/v1/metadata');
        }
        throw new Error('API Offline');
      })
      .then(res => res.json())
      .then(data => setMetadata(data))
      .catch(() => setApiOnline(false));
  }, []);

  return (
    <div className="min-h-screen bg-[#0b0f19] text-gray-100 flex flex-col font-sans">
      {/* Header */}
      <header className="bg-[#111827] border-b border-[#374151] sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white shadow-md">
              M
            </div>
            <span className="text-lg font-bold tracking-tight">MarketPulse</span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400">
              v0.1.0-prod
            </span>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <span className="font-mono text-gray-400">
              Series: {metadata?.n_series ?? 60}
            </span>
            <span className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-mono border ${apiOnline ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-amber-500/10 border-amber-500/30 text-amber-400'}`}>
              <span className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span>{apiOnline ? 'API Connected' : 'Offline'}</span>
            </span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="max-w-7xl mx-auto px-4 flex space-x-2 border-t border-[#1f2937]/50">
          {[
            { id: 'forecasting', label: 'Demand Forecasting', icon: '📈' },
            { id: 'hierarchy', label: 'Hierarchical Coherence', icon: '🌳' },
            { id: 'inventory', label: 'Inventory & Monte Carlo', icon: '📦' },
            { id: 'monitoring', label: 'Drift & Retraining Gate', icon: '🛡️' },
            { id: 'benchmarks', label: 'Research Benchmarks', icon: '🔬' },
          ].map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id as any)}
              className={`flex items-center space-x-2 px-4 py-3 text-xs font-semibold border-b-2 transition-colors ${
                activeTab === t.id
                  ? 'border-blue-500 text-white bg-[#1f2937]/40'
                  : 'border-transparent text-gray-400 hover:text-gray-200'
              }`}
            >
              <span>{t.icon}</span>
              <span>{t.label}</span>
            </button>
          ))}
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-8">
        <div className="p-6 bg-[#111827] border border-[#374151] rounded-xl shadow-lg">
          <h2 className="text-xl font-bold mb-2">Demand Planning &amp; Decision Command Center</h2>
          <p className="text-sm text-gray-400">
            Active view: <span className="font-mono text-blue-400 uppercase">{activeTab}</span>. Connects to backend services at <code className="text-emerald-400 font-mono">/api/v1/</code>.
          </p>
        </div>
      </main>

      <footer className="border-t border-[#1f2937] py-4 text-center text-xs text-gray-500">
        MarketPulse Enterprise Demand Planning Engine &bull; sktime &amp; M5 Hierarchical Architecture
      </footer>
    </div>
  );
};

export default App;

