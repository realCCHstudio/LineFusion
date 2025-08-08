// main.js  ─ 主进程入口 (工作流版本)
const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path        = require('path');
const { execFile }= require('child_process');
const fs          = require('fs');
const iconv = require('iconv-lite');

/* ────────────── 全局常量 ────────────── */
const isDev      = process.env.NODE_ENV === 'development' || !app.isPackaged;
const preloadPath= path.join(__dirname, 'preload.js');
const pythonExe  = isDev
    ? 'python'
    : path.join(process.resourcesPath, 'python', 'python.exe');

// 定义一个专用的、持久化的目录来处理所有文件
const processDir = path.join(app.getPath('userData'), 'process');


/* ────────────── 主窗口 ────────────── */
let mainWindow;
function createWindow () {
    mainWindow  = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false
        }
    });

    if (isDev) {
        mainWindow .loadURL('http://localhost:8000/index.html');
        mainWindow .webContents.openDevTools();
    } else {
        mainWindow .loadFile(path.join(__dirname, 'build', 'index.html'));
        mainWindow .webContents.openDevTools();
    }
}

/* ────────────── IPC (进程间通信) 处理器 ────────────── */

// 处理器1: 让用户选择 LAS/LAZ 文件
ipcMain.handle('pick-las', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        title: '选择 LAS / LAZ 文件',
        filters: [{ name: 'LAS/LAZ', extensions: ['las', 'laz'] }],
        properties: ['openFile']
    });
    return canceled ? null : filePaths[0];
});

// 处理器2: 准备工作流的初始文件 (0.las)
ipcMain.handle('prepare-file', async (_evt, sourcePath) => {
    if (!sourcePath) throw new Error('未提供有效的源文件路径。');

    try {
        // 清理并重新创建 process 目录，确保环境干净
        if (fs.existsSync(processDir)) {
            fs.rmSync(processDir, { recursive: true, force: true });
        }
        fs.mkdirSync(processDir, { recursive: true });

        // 将用户选择的文件复制到 process 目录并重命名为 0.las
        const targetPath = path.join(processDir, '0.las');
        fs.copyFileSync(sourcePath, targetPath);

        console.log(`文件已准备好: ${targetPath}`);
        return { success: true, path: targetPath };
    } catch (error) {
        console.error('准备文件时出错:', error);
        throw error;
    }
});

// 处理器3: 执行工作流步骤一 (lasprocess.py)
ipcMain.handle('run-step1-process', (_evt) => {
    const scriptName = 'lasprocess.py';
    const scriptPath = isDev
        ? path.join(__dirname, scriptName)
        : path.join(process.resourcesPath, 'app.asar.unpacked', scriptName);

    const inputPath = path.join(processDir, '0.las');
    const outputPath = path.join(processDir, '1.las');

    console.log(`正在运行步骤 1: ${scriptPath} ${inputPath} ${outputPath}`);

    const child = execFile(
        pythonExe,
        [scriptPath, inputPath, outputPath], // 将输入和输出路径作为参数传递
        { env: { ...process.env, PYTHONHOME: '', PYTHONPATH: '' }, cwd: processDir, encoding: 'buffer' }
    );
    child.stdout.on('data', d => mainWindow.webContents.send('step1-log', iconv.decode(d, 'gbk')));
    child.stderr.on('data', d => mainWindow.webContents.send('step1-log', `[错误] ${iconv.decode(d, 'gbk')}`));

    // 脚本执行完毕后
    child.on('close', code => {
        if (code !== 0) {
            mainWindow.webContents.send('step1-log', `[错误] 步骤 1 进程以代码 ${code} 退出`);
            return;
        }
        if (!fs.existsSync(outputPath)) {
            mainWindow.webContents.send('step1-log', `[错误] 步骤 1 未生成输出文件: ${outputPath}`);
            return;
        }

        console.log(`步骤 1 完成。输出文件: ${outputPath}`);
        // 通知前端步骤1已完成，并发送输出文件路径
        mainWindow.webContents.send('step1-process-complete', outputPath);
    });
});

// 处理器4: 执行工作流步骤二 (fit.py)
ipcMain.handle('run-step2-process', (_evt) => {
    const scriptName = 'fit.py';
    const scriptPath = isDev
        ? path.join(__dirname, scriptName)
        : path.join(process.resourcesPath, 'app.asar.unpacked', scriptName);

    const inputPath = path.join(processDir, '1.las');
    const outputPath = path.join(processDir, '2.las');
    // 注意：fit.py 还会生成一个 'linedata.json' 文件在同一目录下

    console.log(`正在运行步骤 2: ${scriptPath} ${inputPath} ${outputPath}`);

    const child = execFile(
        pythonExe,
        [scriptPath, inputPath, outputPath], // 将输入和输出路径作为参数传递
        { env: { ...process.env, PYTHONHOME: '', PYTHONPATH: '' }, cwd: processDir, encoding: 'buffer' }
    );
    child.stdout.on('data', d => mainWindow.webContents.send('step1-log', iconv.decode(d, 'gbk')));
    child.stderr.on('data', d => mainWindow.webContents.send('step1-log', `[错误] ${iconv.decode(d, 'gbk')}`));

    child.on('close', code => {
        if (code !== 0) {
            mainWindow.webContents.send('step2-log', `[错误] 步骤 2 进程以代码 ${code} 退出`);
            return;
        }
        if (!fs.existsSync(outputPath)) {
            mainWindow.webContents.send('step2-log', `[错误] 步骤 2 未生成输出文件: ${outputPath}`);
            return;
        }

        console.log(`步骤 2 完成。输出文件: ${outputPath}`);
        // 通知前端步骤2已完成，并发送输出文件路径
        mainWindow.webContents.send('step2-process-complete', outputPath);
    });
});

// 处理器5: 读取属性信息文件 (linedata.json)
ipcMain.handle('get-line-data', async () => {
    const jsonPath = path.join(processDir, 'linedata.json');

    if (!fs.existsSync(jsonPath)) {
        return { success: false, message: '属性文件 (linedata.json) 不存在。' };
    }

    try {
        const rawData = fs.readFileSync(jsonPath, 'utf8');
        const data = JSON.parse(rawData);
        return { success: true, data: data };
    } catch (error) {
        console.error(`读取或解析JSON文件失败: ${jsonPath}`, error);
        return { success: false, message: `读取或解析JSON文件失败: ${error.message}` };
    }
});


/* ────────────── 生命周期 ────────────── */
app.whenReady().then(createWindow);
app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});