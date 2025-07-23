// client.js
// Client side stuffs for greyhound web viewer
//

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

    // The NACL module calls method on the window, make sure the laslaz module
    // can see messages sent by NACL
    //
    window.handleMessage = laslaz.handleMessage;

    // show an error message to the user
    //
    var errorOut = function(msg) {
        $("#messages").html("<p class='error'>" + msg + "</p>");
        console.log('Error : ' + msg);
    };

    // show a status message to the user
    var message = function(msg) {
        $("#messages").html("<p class='message'>" + msg + "</p>");
        console.log('Status: ' + msg);
    };

    $(document).on("plasio.start", function() {
        render.startRenderer($("#container").get(0), message);
    });

})(window);

$(function() {
    // 获取加载画面的DOM元素
    const loader = document.getElementById('loader-wrapper');

    // 定义一个隐藏加载画面的函数
    const hideLoader = () => {
        if (loader) {
            loader.classList.add('hidden');
            // 在CSS过渡动画结束后，可以彻底移除或隐藏该元素，以优化性能
            loader.addEventListener('transitionend', () => {
                loader.style.display = 'none';
            });
        }
    };

    // 设置一个40秒的超时，作为后备方案
    // 如果40秒后页面还没加载完，也强制隐藏加载画面
    const loadTimeout = setTimeout(hideLoader, 40000); // 40000 毫秒 = 40 秒

    // 检查 WebGL 支持
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
        // 当页面所有资源（包括图片、脚本等）都加载完成后执行
        window.onload = () => {
            clearTimeout(loadTimeout); // 清除15秒的后备超时
            // 设置一个 3 秒的延时
            setTimeout(() => {
                hideLoader(); // 延时结束后再隐藏加载画面

                // 触发UI和渲染器的初始化 (这是您原来的逻辑)
                $.event.trigger({
                    type: "plasio.startUI"
                });

                $.event.trigger({
                    type: "plasio.start"
                });

                if (experimental) {
                    $.event.trigger({
                        type: "plasio.webglIsExperimental"
                    });
                }
            }, 1500); // 3000 毫秒 = 3 秒

            // 处理URL参数加载的逻辑 (这也是您原来的逻辑)
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
        // 如果不支持 WebGL，则直接显示不支持信息
        clearTimeout(loadTimeout);
        hideLoader();
        $("#no-support").css("opacity", 1.0);
    }
});