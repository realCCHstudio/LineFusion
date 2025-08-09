document.addEventListener('DOMContentLoaded', function () {
    if (typeof THREE === 'undefined' || !THREE.TrackballControls) {
        console.error('Three.js 或 TrackballControls.js 未能成功加载！请检查html文件中的script标签路径。');
        var loader = document.getElementById('loader');
        if (loader) {
            loader.innerHTML = '<p style="color: red;">错误：无法加载渲染库。</p>';
        }
        return;
    }

    var loader = document.getElementById('loader');
    var scene, camera, renderer, controls;

    function init() {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100000);
        camera.position.set(0, 150, 600);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        controls = new THREE.TrackballControls(camera, renderer.domElement);
        controls.rotateSpeed = 1.0;
        controls.zoomSpeed = 1.2;
        controls.panSpeed = 0.8;

        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(200, 500, 300);
        scene.add(dirLight);

        var gridHelper = new THREE.GridHelper(5000, 100);
        scene.add(gridHelper);
        
        var axisHelper = new THREE.AxisHelper(500);
        scene.add(axisHelper);

        animate();
        window.addEventListener('resize', onWindowResize, false);
    }

    window.electronAPI.onDataReady(function () {
        console.log('接收到 "data-ready" 信号，开始获取数据...');
        if (loader) {
            loader.innerHTML = '<div class="spinner"></div><p>数据已就绪，正在创建点云...</p>';
        }
        setTimeout(loadAndRenderData, 50);
    });

    function loadAndRenderData() {
        window.electronAPI.getReconstructionData().then(function (result) {
            if (!result.success || !result.data) {
                throw new Error(result.message || '返回的数据为空。');
            }

            var data = result.data;
            var modelContainer = new THREE.Object3D();
            scene.add(modelContainer);

            // --- VVVV  使用与 r66 完全兼容的 THREE.Geometry VVVV ---
            if (data.other_points && data.other_points.length > 0) {
                var pointsGeometry = new THREE.Geometry();
                
                for (var i = 0; i < data.other_points.length; i++) {
                    var p = data.other_points[i];
                    pointsGeometry.vertices.push(new THREE.Vector3(p[0], p[1], p[2]));
                    pointsGeometry.colors.push(new THREE.Color().setRGB(p[3] / 255, p[4] / 255, p[5] / 255));
                }

                var pointsMaterial = new THREE.ParticleSystemMaterial({
                    size: 2.5,
                    vertexColors: THREE.VertexColors
                });

                var particleSystem = new THREE.ParticleSystem(pointsGeometry, pointsMaterial);
                modelContainer.add(particleSystem);
            }

            if (data.powerlines) {
                Object.values(data.powerlines).forEach(function (lineData) {
                    if (lineData.points.length < 2) return;
                    var curve = new THREE.SplineCurve3(lineData.points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
                    var tubeGeometry = new THREE.TubeGeometry(curve, Math.min(lineData.points.length * 2, 200), 0.7, 8, false);
                    var powerlineMaterial = new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(lineData.color[0] / 255, lineData.color[1] / 255, lineData.color[2] / 255) });
                    modelContainer.add(new THREE.Mesh(tubeGeometry, powerlineMaterial));
                });
            }

            // 自动居中、旋转和相机定位逻辑 (这部分已兼容r66)
            var boundingBox = new THREE.Box3().setFromObject(modelContainer);
            var center = boundingBox.center();
            modelContainer.position.sub(center);

            var size = boundingBox.size();
            if (size.z > size.x) {
                modelContainer.rotation.y = Math.PI / 2;
            }
            
            var rotatedSize = size.z > size.x ? new THREE.Vector3(size.z, size.y, size.x) : size;
            var fov = camera.fov * (Math.PI / 180);
            
            var fitHeightDistance = rotatedSize.y / (2 * Math.tan(fov / 2));
            var fitWidthDistance = rotatedSize.x / (2 * camera.aspect * Math.tan(fov / 2));
            var distance = Math.max(fitHeightDistance, fitWidthDistance);
            
            distance *= 1.4;

            var cameraY = rotatedSize.y * 0.4;
            camera.position.set(0, cameraY, distance);
            
            var modelCenter = new THREE.Vector3(0, 0, 0);
            camera.lookAt(modelCenter);
            controls.target = modelCenter;
            controls.update();

            console.log("相机已自动调整为透视水平视角。");

            if (loader) {
                loader.style.transition = 'opacity 0.5s';
                loader.style.opacity = '0';
                setTimeout(function() { loader.remove(); }, 500);
            }

        }).catch(function (error) {
            console.error('渲染过程中发生错误:', error);
            if (loader) {
                loader.innerHTML = '<p style="color: red;">渲染失败: ' + error.message + '</p>';
            }
        });
    }

    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    init();
});