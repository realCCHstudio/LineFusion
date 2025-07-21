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
// client.js — 修改/新增部分

(function(w) {
    "use strict";
    // …（原有代码）…

    // ---------- 新增：测量模式开关按钮 ----------
    // 在页面上动态插入一个按钮
    var measureBtn = $('<button id="toggle-measure" style="position:absolute;top:10px;left:10px;z-index:1000;">测量模式</button>');
    $('body').append(measureBtn);

    // 测量状态变量
    var measuring = false;
    var points = [];  // 用来存两个点击点

    measureBtn.on('click', function() {
        measuring = !measuring;
        points = [];
        $('#measurement-overlay').remove();  // 清除旧标注
        $(this).text(measuring ? '退出测量' : '测量模式');
    });

    // ---------- 新增：在渲染容器上监听点击 ----------
    var $container = $("#container");
    $container.css('position', 'relative'); // 确保定位参考

    $container.on('click', function(e) {
        if (!measuring) return;

        // 计算相对于容器左上角的点击位置
        var offset = $(this).offset();
        var x = e.pageX - offset.left;
        var y = e.pageY - offset.top;
        points.push({ x: x, y: y });

        // 标出点击位置
        var dot = $('<div class="measure-dot"></div>').css({
            position: 'absolute',
            width: '8px', height: '8px',
            'border-radius': '4px',
            background: 'red',
            left: (x - 4) + 'px',
            top:  (y - 4) + 'px',
            'z-index': 1001
        });
        $container.append(dot);

        if (points.length === 2) {
            // 两点已选完，进行距离计算
            calculateAndDisplayDistance(points[0], points[1]);
            // 重置状态，或让用户手动清除／继续测量
            measuring = false;
            measureBtn.text('测量模式');
            points = [];
        }
    });

    // ---------- 新增：距离计算与显示 ----------
    function calculateAndDisplayDistance(p1, p2) {
        // 将屏幕坐标转换为三维世界坐标（伪代码，需要结合实际渲染器 API）
        var world1 = render.screenToWorld(p1.x, p1.y);
        var world2 = render.screenToWorld(p2.x, p2.y);

        var dx = world1[0] - world2[0];
        var dy = world1[1] - world2[1];
        var dz = world1[2] - world2[2];
        var dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

        // 在容器上绘制连线
        var svgLine = $(
            '<svg id="measurement-overlay" style="position:absolute;top:0;left:0;pointer-events:none;z-index:1000;" ' +
            'width="'+$container.width()+'" height="'+$container.height()+'">'+
              '<line x1="'+p1.x+'" y1="'+p1.y+'" x2="'+p2.x+'" y2="'+p2.y+'" '+
                    'stroke="red" stroke-width="2"/>' +
              '<text x="'+((p1.x+p2.x)/2+5)+'" y="'+((p1.y+p2.y)/2-5)+'" '+
                    'fill="white" font-size="14px" stroke="black" stroke-width="0.5px">'+
                dist.toFixed(3) +
              '</text>' +
            '</svg>'
        );
        $container.append(svgLine);
    }

    // （注意：需确保 render 对象提供 screenToWorld 方法，或根据你的渲染库自行替换为正确的坐标转换 API）

})(window);

// client.js — 新增 CSS（可放在你的主样式文件中）
/*
.measure-dot {
    box-shadow: 0 0 4px rgba(0,0,0,0.5);
}
*/

$(function() {
	setTimeout(function() {
		var isWebGLSupported = function() {
			if ("WebGLRenderingContext" in window) {
				// might have support
				//
				var e = document.createElement("canvas");
				var webgl = e.getContext("webgl");
				var experimental = false;
				if (webgl === null) {
					webgl = e.getContext("experimental-webgl");
					experimental = true;
				}

				return [webgl !== null, experimental];
			}

			return false;
		};

		// if we're good to go, trigger the plasio.start event, all initializers
		// should be hooked to this event, and not DOMContentLoaded
		//
		var r = isWebGLSupported();
		var supported = r[0];
		var experimental = r[1];
		if(supported) {
			$(".fullscreen").fadeOut(200);
			// we need to intialize the UI first, before we initialize everything else,
			// since UI has to show results and statuses about things as they initialize
			//
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

			var parseURL = function(qs) {
				var name = qs.match(/\?s=(\S+)/);
				return name ? name[1] : null;
			};


			var figureFilename = function(path) {
				var i = path.lastIndexOf("/");
				if (i === -1)
					return path;

				return path.substr(i+1);
			};

			// If a URL is specified, load that now
			var query = window.location.search;
			if (query) {
				query = parseURL(query);
				if (query && query.length > 0) {
					$.event.trigger({
						type: "plasio.loadfiles.remote",
						url: query,
						name: figureFilename(query)
					});
				}
			}
		}
		else {
			$("#no-support").css("opacity", 1.0);
		}
	}, 1000);
});
