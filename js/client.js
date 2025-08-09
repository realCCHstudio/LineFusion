// client.js
// Client side stuffs for greyhound web viewer
//

// --- 新增：定义一个空的 AppState 对象以兼容旧的 ui.js 代码 ---
var AppState = {};

// Import all modules here even if we don't want to use them
// so that browserify can pick them up
var $ = require('jquery'),
    render = require('./render'),
    ui = require('./ui'),
    laslaz = require('./laslaz');


function endsWith(str, s) {
    return str.indexOf(s) === (str.length - s.length);
}

(function(w) {
    "use strict";
    window.handleMessage = laslaz.handleMessage;
    var errorOut = function(msg) {
        $("#messages").html("<p class='error'>" + msg + "</p>");
        console.log('Error : ' + msg);
    };
    var message = function(msg) {
        $("#messages").html("<p class='message'>" + msg + "</p>");
        console.log('Status: ' + msg);
    };
    $(document).on("plasio.start", function() {
        render.startRenderer($("#container").get(0), message);
    });
})(window);

$(function() {
    // ---- 新工作流逻辑的开始 ----
    const customStyles = `
        .scrollable-table-container {
            width: 100%; /* 容器宽度占满父元素 */
            max-height: 300px; /* 设置一个最大高度，超出则出现垂直滚动条 */
            overflow-x: auto; /* 这是关键：当内容超出时，启用水平滚动条 */
            overflow-y: auto; /* 允许垂直滚动 */
            margin-bottom: 10px; /* 和其他控件保持一些间距 */
            border: 1px solid #ddd; /* 添加一个边框，让容器区域更明显 */
            border-radius: 4px; /* 轻微的圆角 */
        }

        .scrollable-table-container table th,
        .scrollable-table-container table td {
            white-space: nowrap; /* 这是关键：强制表格单元格内容不换行 */
        }
    `;

    // 创建一个<style>标签并将其附加到<head>
    $('<style>')
        .prop('type', 'text/css')
        .html(customStyles)
        .appendTo('head');

    const step1Btn = $('#run-step1-btn');
    const step2Btn = $('#run-step2-btn');
    const step3Btn = $('#run-step3-btn');
    const step1Status = $('#step1-status');
    const step2Status = $('#step2-status');
    const step3Status = $('#step3-status');
    const reconstructionBtn = $('#reconstruction-btn');
    const reconstructionStatus = $('#reconstruction-status');
    const consoleOutput = $('#fit-console-output');
    const browseBtn = $('#browse');
    const propertiesContent = $('#properties-content');

    let workflowState = 'initial';

    async function updateSpanPropertiesModule() {
        propertiesContent.html('<p class="text-muted">正在加载电力线属性...</p>');
        // 注意：我们现在调用 getSpanData
        const result = await window.electronAPI.getSpanData();
        if (result.success && result.data) {
            const data = result.data;
            if (!Array.isArray(data) || data.length === 0) {
                propertiesContent.html('<p class="text-muted">电力线属性文件为空。</p>');
                return;
            }

            // 定义表头中文映射
            const headerMap = {
                "line_name": "导线名称",
                "line_id": "导线ID",
                "span_id": "跨段ID",
                "point_count": "点云数量",
                "estimated_length_m": "估算长度(米)"
            };

            let tableHtml = '<table class="table table-striped table-hover table-condensed">';
            tableHtml += '<thead><tr>';
            Object.keys(data[0]).forEach(key => {
                tableHtml += `<th>${headerMap[key] || key}</th>`;
            });
            tableHtml += '</tr></thead>';
            tableHtml += '<tbody>';
            data.forEach(row => {
                tableHtml += '<tr>';
                Object.values(row).forEach(value => {
                    const displayValue = typeof value === 'number' && !Number.isInteger(value)
                        ? value.toFixed(2)
                        : value;
                    tableHtml += `<td>${displayValue}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';
            propertiesContent.html(tableHtml);
        } else {
            propertiesContent.html(`<p class="text-muted">暂无电力线属性信息。</p>`);
        }
    }

    async function updateTowerPropertiesModule() {
        const towerPropertiesContent = $('#tower-properties-content');
        towerPropertiesContent.html('<p class="text-muted">正在加载电塔信息...</p>');
        const result = await window.electronAPI.getTowerData();

        if (result.success && result.data) {
            const data = result.data;
            if (!Array.isArray(data) || data.length === 0) {
                towerPropertiesContent.html('<p class="text-muted">暂无电塔属性信息。请继续运行处理步骤。</p>');
                return;
            }

            // 定义表头中文映射
            const headerMap = {
                "tower_id": "电塔编号",
                "point_count": "点云数量",
                "height_m": "高度 (米)",
                "center_x": "中心X",
                "center_y": "中心Y",
                "center_z": "中心Z"
            };

            let tableHtml = '<table class="table table-striped table-hover table-condensed">';
            tableHtml += '<thead><tr>';
            Object.keys(data[0]).forEach(key => {
                tableHtml += `<th>${headerMap[key] || key}</th>`; // 使用中文表头
            });
            tableHtml += '</tr></thead>';
            tableHtml += '<tbody>';
            data.forEach(row => {
                tableHtml += '<tr>';
                Object.values(row).forEach(value => {
                    tableHtml += `<td>${value}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';
            towerPropertiesContent.html(tableHtml);
        } else {
            towerPropertiesContent.html(`<p class="text-muted">暂无电塔信息。</p>`);
        }
    }

    function updateUI(state) {
        workflowState = state;
        console.log("正在为状态更新UI:", workflowState);
        step1Btn.prop('disabled', true).find('.spinner').hide();
        step2Btn.prop('disabled', true).find('.spinner').hide();
        step3Btn.prop('disabled', true).find('.spinner').hide();
        reconstructionBtn.prop('disabled', true);
        switch (workflowState) {
            case 'initial':
                step1Status.text('请先通过“浏览”按钮加载文件。');
                step2Status.text('');
                step3Status.text('');
                reconstructionStatus.text('完成所有处理步骤后可用。');
                consoleOutput.text('等待操作...');
                propertiesContent.html('<p class="text-muted">请运行处理流程以查看属性。</p>');
                $('#tower-properties-content').html('<p class="text-muted">请运行处理流程以查看电塔属性。</p>');
                break;
            case 'file_prepared':
                step1Btn.prop('disabled', false);
                step1Status.text('文件已就绪。点击按钮开始处理。');
                break;
            case 'step1_running':
                step1Btn.prop('disabled', true).find('.spinner').show();
                step1Status.text('正在运行粗提取...');
                break;
            case 'step1_complete':
                step1Btn.prop('disabled', true);
                step2Btn.prop('disabled', false);
                step1Status.text('粗提取完成！');
                step2Status.text('准备好进行电塔和分段处理。');
                break;
            case 'step2_running':
                step2Btn.prop('disabled', true).find('.spinner').show();
                step2Status.text('正在运行电塔提取和分段...');
                break;
            case 'step2_complete':
                step1Btn.prop('disabled', true);
                step2Btn.prop('disabled', true);
                step3Btn.prop('disabled', false);
                step2Status.text('电塔提取和分段完成！');
                step3Status.text('准备好进行电力线细提取。');
                updateTowerPropertiesModule();
                break;
            case 'step3_running':
                step3Btn.prop('disabled', true).find('.spinner').show();
                step3Status.text('正在运行细提取和分根...');
                break;
            case 'step3_complete':
                step1Btn.prop('disabled', true);
                step2Btn.prop('disabled', true);
                step3Btn.prop('disabled', true);
                reconstructionBtn.prop('disabled', false);
                reconstructionStatus.text('可以启动三维重建。');
                step3Status.text('电力线细提取和分根完成！');
                updateTowerPropertiesModule(); // 重新加载，以防万一
                updateSpanPropertiesModule(); // 加载最终的电力线信息
                break;
        }
    }

    function loadFileInViewer(filePath, fileName) {
        $.event.trigger({
            type: "plasio.loadfiles.remote",
            url: filePath,
            name: fileName
        });
    }

    browseBtn.on('click', async () => {
        const sourcePath = await window.electronAPI.pickLas();
        if (!sourcePath) return;

        console.log(`正在准备文件: ${sourcePath}`);

        // --- 修改之处：修复 path is not defined 错误 ---
        // 使用简单的字符串操作来获取文件名，而不是依赖 'path' 模块
        const fileName = sourcePath.substring(sourcePath.replace(/\\/g, '/').lastIndexOf('/') + 1);
        consoleOutput.text(`正在准备文件: ${fileName}...\n`);

        updateUI('initial');
        try {
            const result = await window.electronAPI.prepareFile(sourcePath);
            if (result.success) {
                consoleOutput.append('文件 `0.las` 已成功创建。\n');
                updateUI('file_prepared');
                loadFileInViewer(result.path, '0.las');
            }
        } catch (err) {
            console.error('文件准备失败:', err);
            consoleOutput.append(`[错误] 文件准备失败: ${err.message}\n`);
            updateUI('initial');
        }
    });

    step1Btn.on('click', () => {
        updateUI('step1_running');
        consoleOutput.text('--- 开始步骤 1: 电力线与地面粗提取 ---\n');
        window.electronAPI.runStep1();
    });

    step2Btn.on('click', () => {
        updateUI('step2_running');
        consoleOutput.append('\n--- 开始步骤 2: 电塔提取和电力线分段 ---\n');
        window.electronAPI.runStep2();
    });

    window.electronAPI.onStep1Log((data) => consoleOutput.append(data));
    window.electronAPI.onStep2Log((data) => consoleOutput.append(data));

    window.electronAPI.onStep1Complete((outputPath) => {
        console.log('步骤 1 完成, 输出路径:', outputPath);
        consoleOutput.append('--- 步骤 1 完成. 正在加载 `1.las`... ---\n');
        updateUI('step1_complete');
        loadFileInViewer(outputPath, '1.las');
    });

    window.electronAPI.onStep2Complete((outputPath) => {
        console.log('步骤 2 完成, 输出路径:', outputPath);
        consoleOutput.append('--- 步骤 2 完成. 正在加载 `2.las`... ---\n');
        updateUI('step2_complete');
        loadFileInViewer(outputPath, '2.las');
    });

    step3Btn.on('click', () => {
        updateUI('step3_running');
        consoleOutput.append('\n--- 开始步骤 3: 电力线细提取和分根 ---\n');
        window.electronAPI.runStep3();
    });

    window.electronAPI.onStep3Log((data) => {
        // 步骤3的日志也输出到同一个控制台
        consoleOutput.append(data);
        consoleOutput.scrollTop(consoleOutput[0].scrollHeight);
    });

    window.electronAPI.onStep3Complete((outputPath) => {
        console.log('步骤 3 完成, 输出路径:', outputPath);
        consoleOutput.append('--- 步骤 3 完成. 正在加载 `3.las`... ---\n');
        updateUI('step3_complete');
        loadFileInViewer(outputPath, '3.las');
    });

    reconstructionBtn.on('click', () => {
        console.log('请求打开三维重建窗口...');
        // 直接调用main进程的句柄，它会处理后台脚本运行和新窗口创建
        window.electronAPI.openReconstructionWindow();
    });

    // ---- 新工作流逻辑的结束 ----

    const loader = document.getElementById('loader-wrapper');
    const hideLoader = () => {
        if (loader) {
            loader.classList.add('hidden');
            loader.addEventListener('transitionend', () => {
                loader.style.display = 'none';
            });
        }
    };
    const loadTimeout = setTimeout(hideLoader, 40000);
    const isWebGLSupported = function() {
        if ("WebGLRenderingContext" in window) {
            const e = document.createElement("canvas");
            let webgl = e.getContext("webgl");
            let experimental = false;
            if (webgl === null) {
                webgl = e.getContext("experimental-webgl");
                experimental = true;
            }
            return [webgl !== null, experimental];
        }
        return [false, false];
    };
    const r = isWebGLSupported();
    const supported = r[0];
    const experimental = r[1];
    if (supported) {
        window.onload = () => {
            clearTimeout(loadTimeout);
            setTimeout(() => {
                hideLoader();
                $.event.trigger({ type: "plasio.startUI" });
                $.event.trigger({ type: "plasio.start" });
                if (experimental) {
                    $.event.trigger({ type: "plasio.webglIsExperimental" });
                }
                updateUI('initial');
            }, 1500);

            const parseURL = function(qs) {
                const name = qs.match(/\?s=(\S+)/);
                return name ? name[1] : null;
            };
            const figureFilename = function(path) {
                const i = path.lastIndexOf("/");
                if (i === -1) return path;
                return path.substr(i + 1);
            };
            const query = window.location.search;
            if (query) {
                const parsedQuery = parseURL(query);
                if (parsedQuery && parsedQuery.length > 0) {
                    $.event.trigger({
                        type: "plasio.loadfiles.remote",
                        url: parsedQuery,
                        name: figureFilename(parsedQuery)
                    });
                }
            }
        };
    } else {
        clearTimeout(loadTimeout);
        hideLoader();
        $("#no-support").css("opacity", 1.0);
    }
});