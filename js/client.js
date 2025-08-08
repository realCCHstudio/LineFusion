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
    const step1Btn = $('#run-step1-btn');
    const step2Btn = $('#run-step2-btn');
    const step1Status = $('#step1-status');
    const step2Status = $('#step2-status');
    const consoleOutput = $('#fit-console-output');
    const browseBtn = $('#browse');
    const propertiesContent = $('#properties-content');

    let workflowState = 'initial';

    async function updatePropertiesModule() {
        propertiesContent.html('<p class="text-muted">正在加载属性信息...</p>');
        const result = await window.electronAPI.getLineData();
        if (result.success && result.data) {
            const data = result.data;
            if (!Array.isArray(data) || data.length === 0) {
                propertiesContent.html('<p class="text-muted">属性文件为空或格式不正确。</p>');
                return;
            }
            let tableHtml = '<table class="table table-striped table-hover table-condensed">';
            tableHtml += '<thead><tr>';
            Object.keys(data[0]).forEach(key => {
                tableHtml += `<th>${key}</th>`;
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
            propertiesContent.html(`<p class="text-muted">暂无属性信息。请确保已成功运行所有处理步骤。</p>`);
        }
    }

    function updateUI(state) {
        workflowState = state;
        console.log("正在为状态更新UI:", workflowState);
        step1Btn.prop('disabled', true).find('.spinner').hide();
        step2Btn.prop('disabled', true).find('.spinner').hide();
        switch (workflowState) {
            case 'initial':
                step1Status.text('请先通过“浏览”按钮加载文件。');
                step2Status.text('');
                consoleOutput.text('等待操作...');
                propertiesContent.html('<p class="text-muted">请运行处理流程以查看属性。</p>');
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
                step2Status.text('电塔提取和分段完成！');
                updatePropertiesModule();
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