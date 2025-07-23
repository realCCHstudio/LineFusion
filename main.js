// main.js  ─ 主进程入口 (最终版)
const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path        = require('path');
const { execFile }= require('child_process');

/* ────────────── 全局常量 ────────────── */
const isDev      = process.env.NODE_ENV === 'development' || !app.isPackaged;
const preloadPath= path.join(__dirname, 'preload.js');
const pythonExe  = isDev
    ? 'python'
    : path.join(process.resourcesPath, 'python', 'python.exe');

/* ────────────── 工具函数 ────────────── */
function runPython(inputPath, cb) {
    const scriptPath = isDev
        ? path.join(__dirname, 'lasprocess.py')
        : path.join(process.resourcesPath, 'app.asar.unpacked', 'lasprocess.py');

    const outputPath = path.join(
        app.getPath('downloads'),
        `processed_${path.basename(inputPath)}`
    );

    execFile(
        pythonExe,
        [scriptPath, inputPath, outputPath],
        { env: { ...process.env, PYTHONHOME: '', PYTHONPATH: '' } },
        (err, stdout, stderr) => {
            if (err) return cb(err);
            shell.showItemInFolder(outputPath);
            cb(null, { stdout, outputPath });
        }
    );
}

/* ────────────── 主窗口 ────────────── */
function createWindow () {
    const win = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: preloadPath,
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false
        }
    });

    if (isDev) {
        win.loadURL('http://localhost:8000/index.html');
        win.webContents.openDevTools();
    } else {
        win.loadFile(path.join(__dirname, 'build', 'index.html'));
    }
}

/* ────────────── IPC ────────────── */
ipcMain.handle('pick-las', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        title: '选择 LAS / LAZ 文件',
        filters: [{ name: 'LAS/LAZ', extensions: ['las', 'laz'] }],
        properties: ['openFile']
    });
    return canceled ? null : filePaths[0];
});

ipcMain.handle('run-las-process', async (_evt, inputPath) =>
    new Promise((resolve, reject) => {
        if (!inputPath) return reject(new Error('空路径'));
        runPython(inputPath, (err, res) => {
            if (err) return reject(err);
            resolve(res);
        });
    })
);

/* ────────────── 生命周期 ────────────── */
app.whenReady().then(createWindow);
app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
