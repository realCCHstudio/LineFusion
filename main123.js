// main.js
const { app, BrowserWindow } = require('electron')

// —— 1. 确保没有误触全局禁用 ——
// 如果之前有 `app.disableHardwareAcceleration()`，一定删掉它。
// —— 2. 重新显式开启硬件加速（Electron v25+ 提供） ——
if (app.enableHardwareAcceleration) {
  app.enableHardwareAcceleration()
}

// —— 3. 强制忽略 Chromium GPU 黑名单 —— 
app.commandLine.appendSwitch('ignore-gpu-blacklist')

// —— 4. 强制使用 ANGLE (Direct3D11) 渲染 —— 
app.commandLine.appendSwitch('use-angle', 'd3d11')

// —— 5. 强制启用 WebGL / GPU 光栅化 —— 
app.commandLine.appendSwitch('enable-webgl')
app.commandLine.appendSwitch('enable-gpu-rasterization')

// （可选）启用一些实验特性：
// app.commandLine.appendSwitch('enable-features', 'WebGL2ComputeContext,AcceleratedCanvas2D')

const path = require('path')
const log  = require('electron-log')

log.info('应用程序启动（GPU 强制模式）')

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200, height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
      allowRunningInsecureContent: true
    }
  })

  mainWindow.loadFile(path.join(__dirname, 'build', 'index.html'))
  // 方便调试：
  mainWindow.webContents.openDevTools()

  mainWindow.webContents.on('did-finish-load', () => {
    log.info('页面加载完成')
  })
  mainWindow.webContents.on('did-fail-load', (_, code, desc) => {
    log.error('载入失败', desc)
  })
}

app.whenReady().then(() => {
  // 再次打印状态，确保生效：
  console.log('GPU Status After Flags:', app.getGPUFeatureStatus())
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

process.on('uncaughtException', e => {
  log.error('未捕获异常：', e)
})
