// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // 文件与工作流管理
    pickLas: () => ipcRenderer.invoke('pick-las'),
    prepareFile: (path) => ipcRenderer.invoke('prepare-file', path),

    // 步骤 1: 粗提取
    runStep1: () => ipcRenderer.invoke('run-step1-process'),
    onStep1Log: (callback) => ipcRenderer.on('step1-log', (_event, ...args) => callback(...args)),
    onStep1Complete: (callback) => ipcRenderer.on('step1-process-complete', (_event, ...args) => callback(...args)),

    // 步骤 2: 电塔和跨段提取
    runStep2: () => ipcRenderer.invoke('run-step2-process'),
    onStep2Log: (callback) => ipcRenderer.on('step2-log', (_event, ...args) => callback(...args)),
    onStep2Complete: (callback) => ipcRenderer.on('step2-process-complete', (_event, ...args) => callback(...args)),

    // 获取属性信息
    getLineData: () => ipcRenderer.invoke('get-line-data'),
    getTowerData: () => ipcRenderer.invoke('get-tower-data'),
});