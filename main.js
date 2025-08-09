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

// 处理器3: 执行工作流步骤一 (process_1.py)
ipcMain.handle('run-step1-process', (_evt) => {
    const scriptName = 'process_1.py';
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

// 处理器4: 执行工作流步骤二 (process_2.py)
ipcMain.handle('run-step2-process', (_evt) => {
    const scriptName = 'process_2.py';
    const scriptPath = isDev
        ? path.join(__dirname, scriptName)
        : path.join(process.resourcesPath, 'app.asar.unpacked', scriptName);

    const inputPath = path.join(processDir, '1.las');
    const outputPath = path.join(processDir, '2.las');
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

// 处理器5: 读取电塔属性信息文件 (tower.json)
ipcMain.handle('get-tower-data', async () => {
    const jsonPath = path.join(processDir, 'tower.json');

    if (!fs.existsSync(jsonPath)) {
        return { success: false, message: '电塔属性文件 (tower.json) 不存在。' };
    }

    try {
        const rawData = fs.readFileSync(jsonPath, 'utf8');
        const data = JSON.parse(rawData);
        return { success: true, data: data };
    } catch (error) {
        console.error(`读取或解析电塔JSON文件失败: ${jsonPath}`, error);
        return { success: false, message: `读取或解析电塔JSON文件失败: ${error.message}` };
    }
});

// 处理器6: 执行工作流步骤三 (process_3.py)
ipcMain.handle('run-step3-process', (_evt) => {
    const scriptName = 'process_3.py';
    const scriptPath = isDev
        ? path.join(__dirname, scriptName)
        : path.join(process.resourcesPath, 'app.asar.unpacked', scriptName);

    const inputPath = path.join(processDir, '2.las');
    const outputPath = path.join(processDir, '3.las');

    console.log(`正在运行步骤 3: ${scriptPath} ${inputPath} ${outputPath}`);

    const child = execFile(
        pythonExe,
        [scriptPath, inputPath, outputPath],
        { env: { ...process.env, PYTHONHOME: '', PYTHONPATH: '' }, cwd: processDir, encoding: 'buffer' }
    );
    child.stdout.on('data', d => mainWindow.webContents.send('step3-log', iconv.decode(d, 'gbk')));
    child.stderr.on('data', d => mainWindow.webContents.send('step3-log', `[错误] ${iconv.decode(d, 'gbk')}`));

    child.on('close', code => {
        if (code !== 0) {
            mainWindow.webContents.send('step3-log', `[错误] 步骤 3 进程以代码 ${code} 退出`);
            return;
        }
        if (!fs.existsSync(outputPath)) {
            mainWindow.webContents.send('step3-log', `[错误] 步骤 3 未生成输出文件: ${outputPath}`);
            return;
        }

        console.log(`步骤 3 完成。输出文件: ${outputPath}`);
        mainWindow.webContents.send('step3-process-complete', outputPath);
    });
});

// 处理器8: 读取电力线(跨段)属性信息文件 (span.json)
ipcMain.handle('get-span-data', async () => {
    const jsonPath = path.join(processDir, 'span.json');

    if (!fs.existsSync(jsonPath)) {
        return { success: false, message: '电力线属性文件 (span.json) 不存在。' };
    }

    try {
        const rawData = fs.readFileSync(jsonPath, 'utf8');
        const data = JSON.parse(rawData);
        return { success: true, data: data };
    } catch (error) {
        console.error(`读取或解析span.json文件失败: ${jsonPath}`, error);
        return { success: false, message: `读取或解析span.json文件失败: ${error.message}` };
    }
});

ipcMain.handle('open-reconstruction-window', () => {
    // 1. 立刻创建一个新窗口
    let reconWindow = new BrowserWindow({
        width: 1000,
        height: 700,
        title: '三维重建视图',
        webPreferences: {
            preload: preloadPath, // 复用同一个 preload.js
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false
        }
    });

    if (isDev) {
        reconWindow.loadURL('http://localhost:8000/reconstruction.html');
    } else {
        reconWindow.loadFile(path.join(__dirname, 'build', 'reconstruction.html'));
    }

    // !!! 关键：为新窗口打开开发者工具，方便调试！
    reconWindow.webContents.openDevTools();

    reconWindow.on('closed', () => {
        reconWindow = null;
    });

    // 2. 在窗口显示后，再在后台运行Python脚本
    const scriptName = 'process_4_export.py';
    const scriptPath = isDev
        ? path.join(__dirname, scriptName)
        : path.join(process.resourcesPath, 'app.asar.unpacked', scriptName);

    const inputPath = path.join(processDir, '3.las');
    const outputPath = path.join(processDir, 'webgl_data.json');

    console.log(`后台导出数据: ${inputPath} -> ${outputPath}`);

    const child = execFile(
        pythonExe,
        [scriptPath, inputPath, outputPath],
        { env: { ...process.env, PYTHONHOME: '', PYTHONPATH: '' }, cwd: processDir, encoding: 'buffer' }
    );

    child.stdout.on('data', d => console.log(iconv.decode(d, 'gbk')));
    child.stderr.on('data', d => console.error(`[导出错误] ${iconv.decode(d, 'gbk')}`));

    child.on('close', code => {
        if (code !== 0 || !fs.existsSync(outputPath)) {
            dialog.showErrorBox('三维重建错误', '无法生成用于重建的数据文件。请查看主进程日志获取详细信息。');
            return;
        }

        console.log('数据导出成功，正在通知渲染窗口...');

        // 3. 脚本成功后，向新窗口发送“数据已就绪”的通知
        if (reconWindow) {
            reconWindow.webContents.send('data-ready');
        }
    });
});

// 处理器: 为新窗口提供重建所需的数据
ipcMain.handle('get-reconstruction-data', async () => {
    const jsonPath = path.join(processDir, 'webgl_data.json');
    if (!fs.existsSync(jsonPath)) {
        return { success: false, message: '重建数据文件 (webgl_data.json) 不存在。' };
    }
    try {
        const rawData = fs.readFileSync(jsonPath, 'utf8');
        return { success: true, data: JSON.parse(rawData) };
    } catch (error) {
        console.error('读取 webgl_data.json 失败:', error);
        return { success: false, message: error.message };
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