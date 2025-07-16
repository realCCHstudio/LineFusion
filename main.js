const { app, BrowserWindow } = require('electron');
const path = require('path');
app.commandLine.appendSwitch('ignore-gpu-blacklist')         // 忽略 Chromium 的 GPU 黑名单
app.commandLine.appendSwitch('enable-webgl')                 // 强制启用 WebGL
app.commandLine.appendSwitch('enable-gpu-rasterization')     // 强制启用 GPU 光栅化
// 添加日志
const log = require('electron-log');
log.transports.file.level = 'info';
log.info('应用程序启动');
// const { app, BrowserWindow } = require('electron');

// —— 保证硬件加速没有被全局关闭 ——
// （这行如果你写过就删掉，或者确认没写）
// app.disableHardwareAcceleration();

// app.whenReady().then(() => {
//   // 输出当前 GPU 支持情况
//   console.log('GPU Feature Status:', app.getGPUFeatureStatus());
  
//   createWindow();
//   // … 其余逻辑
// });

function createWindow() {
    // 创建浏览器窗口
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            webSecurity: false, // 允许加载本地资源
            allowRunningInsecureContent: true, // 允许加载混合内容
            webgl: true  // 确保 WebGL 启用
        }
    });

    log.info('创建主窗口');
    
    // 加载构建目录中的 index.html
    mainWindow.loadFile(path.join(__dirname, 'build', 'index.html'));
    
    // 打开开发者工具以便调试
    // mainWindow.webContents.openDevTools();

    // 监听页面加载完成事件
    mainWindow.webContents.on('did-finish-load', () => {
        log.info('页面加载完成');
    });

    // 监听页面加载失败事件
    mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
        log.error('页面加载失败:', errorDescription);
    });

    mainWindow.on('closed', () => {
        log.info('主窗口关闭');
    });
}

// 当 Electron 完成初始化时创建窗口
app.whenReady().then(() => {
    console.log('GPU Feature Status:', app.getGPUFeatureStatus());
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// 当所有窗口都被关闭时退出
app.on('window-all-closed', () => {
    log.info('所有窗口关闭，准备退出应用');
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

process.on('uncaughtException', (error) => {
    log.error('未捕获的异常：', error);
}); 