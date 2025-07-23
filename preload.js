// preload.js ─ 在渲染进程与主进程之间架设安全桥梁
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    pickLas       : ()        => ipcRenderer.invoke('pick-las'),
    runLasProcess : (path)    => ipcRenderer.invoke('run-las-process', path),
    onProcessDone : (handler) => ipcRenderer.once('las-process-complete', handler) // 备用；目前未使用
});
