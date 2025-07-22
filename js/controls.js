/** @jsx React.DOM */

var React = require('react');
var classNames = require('classnames');

var ReactBootstrap = require('react-bootstrap');
var $ = require('jquery');
var render = require('./render');
var _ = require('lodash');

var util = require('./util');

React.createClass = require('create-react-class');

// Import some react-bootstrap componenets
//
var Button = ReactBootstrap.Button,
    Modal = ReactBootstrap.Modal,
    ModalTrigger = ReactBootstrap.ModalTrigger,
    Grid = ReactBootstrap.Grid,
    Row = ReactBootstrap.Row,
    Col = ReactBootstrap.Col,
    Alert = ReactBootstrap.Alert,
    FormGroup = ReactBootstrap.FormGroup,
    FormControl = ReactBootstrap.FormControl;


console.log(ReactBootstrap);

var withRefresh = require('./util').withRefresh;

(function(scope) {
    "use strict";

    var InundationControls = React.createClass({
        getInitialState: function() {
            return { enabled: false,
                slider: false,
                opacity: 50,
                value: 0};
        },

        componentDidMount: function() {
        },

        componentDidUpdate: function() {
            // if we have #inun, make it into a slider
            if (!this.state.slider) {
                var n = $("#inun").get(0);
                var m = $("#inun-opacity").get(0);

                var o = this;

                $(n).noUiSlider({
                    range:[0, 1000],
                    start: o.state.value,
                    handles: 1,
                    connect: "lower",
                    slide: withRefresh(function() {
                        $.event.trigger({
                            type: 'plasio.inundationChanged'
                        });

                        o.setState({value: $(n).val()});
                    })
                });

                $(m).noUiSlider({
                    range:[0, 100],
                    start: o.state.opacity,
                    handles: 1,
                    connect: "lower",
                    slide: withRefresh(function() {
                        $.event.trigger({
                            type: 'plasio.inundationOpacityChanged'
                        });

                        o.setState({opacity: $(m).val()});
                    })
                });

                o.setState({slider: true});
            }
        },

        render: function() {
            var classes = classNames({
                'btn btn-block btn-sm': true,
                'btn-default': !this.state.enabled,
                'btn-success active': this.state.enabled
            });

            var additionControls = this.state.enabled ?
                (<div>
                    <div id="inun"/>
                    <h5 className="not-first">调整淹没平面的透明度</h5>
                    <div id="inun-opacity"/>
                </div>) :
                    <div /> ;


            return (
                <div>
                    <button type="button"
                            className={classes}
                            style={{marginBottom: '15px'}}
                            onClick={withRefresh(this.toggle)}>
                        {this.state.enabled? "禁用" : "启用"}
                    </button>
                    {additionControls}
                </div>
            );
        },

        toggle: function() {
            var nextEnabled = !this.state.enabled;
            this.setState({enabled: nextEnabled,
                          slider: false});

                          $.event.trigger({
                              type: 'plasio.inundationEnable',
                              enable: nextEnabled
                          });
        }
    });

    var LineSegment = React.createClass({
        render: function() {
            return (
                <tr style={{ backgroundColor: '#' + this.props.start.color.getHexString() }}>
                    <td> {this.props.lineIndex} </td>
                    <td style={{textAlign: 'right'}}> {this.props.start.distanceTo(this.props.end).toFixed(1)}</td>
                    <td>
                        <a href="#" onClick={this.addRegion}>
                            <span className="glyphicon glyphicon-picture" />
                        </a>
                    </td>
                </tr>
            );
        },

        addRegion: function(e) {
            render.createNewRegion(this.props.start, this.props.end, this.props.start.color);
        }
    });


    var LineSegmentsBox = React.createClass({
        getInitialState: function() {
            return { points: [] };
        },

        componentWillMount: function() {
            var c = this;
            $(document).on('plasio.mensuration.pointAdded', function(e) {
                c.setState({ points: c.state.points.concat([e.point])});
            });

            $(document).on('plasio.mensuration.pointRemoved', function(e) {
                c.setState({ points: _.without(c.state.points, e.point) });
            });

            $(document).on('plasio.mensuration.pointsReset', function(e) {
                console.log("Resetting all points");
                c.setState({ points: [] });
            });
        },

        render: function() {
            var lines = [];
            var index = 0;
            for (var i = 0 ; i < this.state.points.length - 1 ; i ++) {
                var p1 = this.state.points[i],
                p2 = this.state.points[i+1];
                if (p1.id === p2.id) {
                    lines.push(React.createElement(LineSegment, { key: index, lineIndex: index+1, start: p1, end: p2 }));
                    index ++;
                }
            }

            if (lines.length === 0)
                return (
                    <div className="its-empty">No Measurement Segments</div>
                );

            return (
                <table className="table">
                    <thead>
                    <tr>
                        <td>Index</td>
                        <td style={{textAlign: 'right'}}>Length</td>
                        <td></td>
                    </tr>
                    </thead>
                    <tbody>
                    {lines}
                    </tbody>
                </table>
            );
        }
    });

    var RegionViewport = React.createClass({
        render: function() {
            var classes = classNames({
                'btn btn-block btn-sm': true,
                'btn-default': !this.props.region.active,
                'btn-success active': this.props.region.active
            });

            return (
                <button
                    className={classes}
                    onClick={this.props.toggle}>
                    {this.props.region.active ? "禁用" : "启用"}
                </button>
            );
        }
    });

    var RegionSizeSlider = React.createClass({
        componentDidMount: function() {
            // 使用 this.refs.sliderNode 来安全地获取DOM元素
            var a = this.refs.sliderNode;
            $(a).noUiSlider({
                range: [1, 10],
                start: Math.round(this.props.startScale),
                step: 1,
                handles: 1,
                connect: false,
                slide: this.setSize
            });
        },
        render: function() {
            // 在 render 方法中，通过 ref 属性为DOM元素添加一个引用
            return (
                <div style={{ marginBottom: '15px' }} ref="sliderNode" />
            );
        },
        setSize: function() {
            // 同样，在这里也使用 this.refs.sliderNode
            var v = $(this.refs.sliderNode).val();
            this.props.setSize(v);
        }
    });

    var Region = React.createClass({
        render: function() {
            var cx = classNames;
            var classesFor = function(active) {
                return cx({
                    'btn': true,
                    'btn-default': true,
                    'btn-sm': true,
                    'active': active });
            };

            console.log('Rendering');

            var regionControls =
                this.props.region.type === 1 ? (
                    <div>
                        <RegionSizeSlider
                            region={this.props.region}
                            startScale={this.props.region.widthScale}
                            setSize={_.partial(this.props.setWidth, this.props.index)} />
                        <RegionSizeSlider
                            region={this.props.region}
                            startScale={this.props.region.heightScale}
                            setSize={_.partial(this.props.setHeight, this.props.index)} />
                    </div> ) : <div /> ;

                    return (
                        <div style={{
                        borderLeft: '10px solid #' + this.props.region.color.getHexString(),
                        marginBottom: '5px',
                        paddingLeft: '5px',
                        boxSizing: 'border-box'}}>
                        <div
                            className="btn btn-link btn-sm"
                            onClick={_.partial(this.props.remove, this.props.index)}
                            type="button"
                            style={{
                                float: 'right',
                                padding: '0px'
                            }}>
                            <span className="glyphicon glyphicon-remove" />
                        </div>
                        <div
                            className="btn-group btn-group-justified"
                            style={{marginBottom: '10px'}}>
                            <div
                                className={classesFor(this.props.region.type === 1)}
                                onClick={_.partial(this.props.setRibbon, this.props.index)}
                                type="button">条带</div>
                            <div
                                className={classesFor(this.props.region.type === 2)}
                                onClick={_.partial(this.props.setAxisAligned, this.props.index)}
                                type="button">轴对齐</div>
                        </div>
                        {regionControls}
                        <RegionViewport
                            region={this.props.region}
                            toggle={_.partial(this.props.toggle, this.props.index)} />
                    </div>
                    );
        },
    });

    var RegionsBox = React.createClass({
        getInitialState: function() {
            return { regions: [] };
        },

        componentWillMount: function() {
            var o = this;
            $(document).on("plasio.regions.new", function(e) {

                o.setState({ regions: o.state.regions.concat(e.region) });
            });

            $(document).on("plasio.regions.reset", function() {
                o.setState({ regions: [] });
            });

        },

        // render: function() {
        //     if (this.state.regions.length === 0)
        //         return (
        //             <div className="its-empty">No regions defined</div>
        //         );
        //
        //         var toggleClip = withRefresh(function() {
        //             $.event.trigger({
        //                 type: 'plasio.render.toggleClip'
        //             });
        //         });
        //
        //         var o = this;
        //         var regions = _.times(this.state.regions.length, function(i) {
        //             var r = o.state.regions[i];
        //             return Region({
        //                 index: i,
        //                 region: o.state.regions[i],
        //                 setRibbon: o.setRibbon,
        //                 setAxisAligned: o.setAxisAligned,
        //                 setWidth: o.setWidth,
        //                 setHeight: o.setHeight,
        //                 remove: o.remove,
        //                 toggle: o.toggle });
        //         });
        //
        //     return (
        //         <div>
        //             <button
        //                 className='btn btn-info btn-sm btn-block'
        //                 style={{marginBottom: '10px'}}
        //                 onClick={toggleClip}>
        //                 Toggle Regions View (T)
        //             </button>
        //             {regions}
        //         </div>
        //     );
        // },
        render: function() {
            if (this.state.regions.length === 0)
                return (
                    <div className="its-empty">No regions defined</div>
                );

            var toggleClip = withRefresh(function() {
                $.event.trigger({
                    type: 'plasio.render.toggleClip'
                });
            });

            var o = this;
            var regions = _.times(this.state.regions.length, function(i) {
                var r = o.state.regions[i];
                // 修正：使用 React.createElement 来创建每一个区域控制面板
                return React.createElement(Region, {
                    key: i, // 添加 key 属性，这是React推荐的做法
                    index: i,
                    region: o.state.regions[i],
                    setRibbon: o.setRibbon,
                    setAxisAligned: o.setAxisAligned,
                    setWidth: o.setWidth,
                    setHeight: o.setHeight,
                    remove: o.remove,
                    toggle: o.toggle
                });
            });

            return (
                <div>
                    <button
                        className='btn btn-info btn-sm btn-block'
                        style={{marginBottom: '10px'}}
                        onClick={toggleClip}>
                        切换区域视图 (T)
                    </button>
                    {regions}
                </div>
            );
        },

        setRibbon: withRefresh(function(i) {
            this.state.regions[i].type = 1;
            this.setState({ regions: this.state.regions });
        }),
        setAxisAligned: withRefresh(function(i) {
            this.state.regions[i].type = 2;
            this.setState({ regions: this.state.regions });
        }),

        setWidth: withRefresh(function(i, w) {
            this.state.regions[i].widthScale = w;
            this.setState({ regions: this.state.regions });
        }),

        setHeight: withRefresh(function(i, h) {
            this.state.regions[i].heightScale = h;

            this.setState({ regions: this.state.regions });
        }),

        remove: withRefresh(function(i) {
            console.log('Removing region');
            var r = this.state.regions[i];
            this.setState({ regions: _.without(this.state.regions, r) });

            $.event.trigger({
                type: 'plasio.regions.remove',
                region: r
            });
        }),
        toggle: withRefresh(function(i) {
            this.state.regions[i].active = !this.state.regions[i].active;
            this.setState({ regions: this.state.regions });
        })
    });

    var OpenGreyhoundPipeline = React.createClass({
        getInitialState: function() {
            return {
                canOpen: false
            };
        },

        componentDidMount: function() {
            this.updateControlState();
        },

        updateControlState: function(e) {
            var url = this.refs.pipelineUrl.getValue(),
                server = this.refs.serverAddress.getValue(),
                pipelineId = this.refs.pipelineId.getValue();

            if (e)
                e.stopPropagation();

            console.log(url, server, pipelineId);

            this.setState({
                error: null,
                canOpen: (url.length > 0) || (server.length > 0 && pipelineId.length > 0)
            });
        },

        handleOpen: function() {
            var url = this.refs.pipelineUrl.getValue(),
                server = this.refs.serverAddress.getValue(),
                pipelineId = this.refs.pipelineId.getValue();


            var comps = {};
            if (url.length > 0) {
                comps = util.parseGHComponents(url); // normalize this URL
            }
            else {
                comps = {server: server, pipelineId: pipelineId};
            }

            console.log("Got components:", comps);

            if (!comps) {
                return this.setState({error: 'The specified pipeline settings seem invalid.'}, function() {
                    var node = this.refs.pipelineUrl.getInputDOMNode();

                    node.setSelectionRange(0, node.value.length);
                    node.focus();
                });
            }

            $.event.trigger({
                type: 'plasio.loadfiles.greyhound',
                comps: [comps]
            });

            this.props.onRequestHide();
        },

        openQuickPipeline: function(pipeline) {
            $.event.trigger({
                type: 'plasio.loadfiles.greyhound',
                comps: [{server: 'test.greyhound.io:8080', pipelineId: pipeline}]
            });

            this.props.onRequestHide();
        },

        render: function() {
            var error = this.state.error ? (
                <Row>
                    <Col xs={12}>
                        <Alert bsStyle="danger">
                            <strong>There was a problem processing your request:</strong><br />
                            <span>{this.state.error}</span>
                        </Alert>
                    </Col>
                </Row> ) : <span /> ;

            return (
                <div>
                    <div className="modal-body">
                        <Grid fluid={true}>
                            <Row>
                                <Col xs={12}>
                                    <FormGroup>
                                        <FormControl type="text" placeholder="pipeline-url"
                                                     ref="pipelineUrl"
                                                     autoFocus
                                                     onChange={this.updateControlState} />
                                    </FormGroup>
                                </Col>
                            </Row>
                            <Row>
                                <Col xs={12}>
                                    <h5 style={{textAlign: "center", fontWeight:"bold", color:"#999", paddingBottom: "10px"}}>OR</h5>
                                </Col>
                            </Row>
                            <Row>
                                <Col xs={4}>
                                    <FormGroup>
                                        <FormControl type="text" placeholder="server-address"
                                                     ref="serverAddress"
                                                     onChange={this.updateControlState} />
                                    </FormGroup>
                                </Col>
                                <Col xs={8}>
                                    <FormGroup>
                                        <FormControl type="text" placeholder="pipeline-id"
                                                     ref="pipelineId"
                                                     onChange={this.updateControlState} />
                                    </FormGroup>
                                </Col>
                            </Row>
                            <Row>
                                <Col xs={12}>
                                    <h5 style={{textAlign: "center", fontWeight:"bold", color:"#999", paddingBottom: "10px"}}>OR</h5>
                                </Col>
                            </Row>
                            <Row>
                                <Col xs={6}>
                                    <Button type="button"
                                        onClick={this.openQuickPipeline.bind(this, "d7b7380b4529abaacbd963ab4c6c474b")}
                                        className="btn-block btn-default">Low Density Autzen</Button>
                                </Col>
                                <Col xs={6}>
                                    <Button type="button"
                                        onClick={this.openQuickPipeline.bind(this, "de18c06d3bbd7777b5c0bd141af81b34")}
                                        className="btn-block btn-default">High Density Half Dome</Button>
                                </Col>
                            </Row>
                            { error }
                        </Grid>
                    </div>
                    <div className="modal-footer">
                        <Button onClick={this.props.onRequestHide} bsStyle="warning">Cancel</Button>
                        <Button onClick={this.handleOpen} bsStyle="success" disabled={!this.state.canOpen}>Open Pipeline</Button>
                    </div>
                </div>
            );
        }
    });

    var openGreyhoundPipelineButton = React.createClass({
        getInitialState: function() {
            return { show: false };
        },

        render:function() {
            var o = this;
            var close = function() { o.setState({ show: false }); };
            var open = function() { o.setState({show: true}); };

            return (
                <div className="modal-container">
                    <Button
                        bsStyle="default"
                        bsSize="small"
                        className="btn-block"
                        onClick={open}
                    >
                        Open
                    </Button>

                    <Modal
                        show={this.state.show}
                        onHide={close}
                        container={this}
                        aria-labelledby="contained-modal-title"
                    >
                        <OpenGreyhoundPipeline onRequestHide={close} />
                    </Modal>
                </div>
            );
        },
    });

    // export stuff
    scope.InundationControls = InundationControls;
    scope.LineSegmentsBox = LineSegmentsBox;
    scope.RegionsBox = RegionsBox;
    scope.openGreyhoundPipelineButton = openGreyhoundPipelineButton;

    // Σ�������ע�������
    // 修改现有的 DangerZoneControls 组件
    var DangerZoneControls = React.createClass({
        getInitialState: function() {
            return {
                isActive: false,
                riskLevel: 'low',
                regions: []
            };
        },
    
        componentWillMount: function() {
            var o = this;
            $(document).on("plasio.dangerzone.new", function(e) {
                o.setState({ regions: o.state.regions.concat(e.region) });
            });
    
            $(document).on("plasio.dangerzone.reset", function() {
                o.setState({ regions: [] });
            });
        },
    
        toggleDangerMode: function() {
            var newState = !this.state.isActive;
            this.setState({ isActive: newState });
            
            $.event.trigger({
                type: 'plasio.dangerzone.toggle',
                active: newState,
                riskLevel: this.state.riskLevel
            });
        },
    
        setRiskLevel: function(level) {
            this.setState({ riskLevel: level });
            
            $.event.trigger({
                type: 'plasio.dangerzone.riskLevelChanged',
                riskLevel: level
            });
        },
    
        removeDangerZone: function(index) {
            var region = this.state.regions[index];
            this.setState({ regions: _.without(this.state.regions, region) });
            
            $.event.trigger({
                type: 'plasio.dangerzone.remove',
                region: region
            });
        },
    
        // 设置危险区域为条带类型
        setRibbon: withRefresh(function(i) {
            this.state.regions[i].type = 1;
            this.setState({ regions: this.state.regions });
            render.getDangerZoneController().setRibbon(i);
        }),
    
        // 设置危险区域为轴对称类型
        setAxisAligned: withRefresh(function(i) {
            this.state.regions[i].type = 2;
            this.setState({ regions: this.state.regions });
            render.getDangerZoneController().setAxisAligned(i);
        }),
    
        // 设置危险区域宽度
        setWidth: withRefresh(function(i, w) {
            this.state.regions[i].widthScale = w;
            this.setState({ regions: this.state.regions });
            render.getDangerZoneController().setWidth(i, w);
        }),
    
        // 设置危险区域高度
        setHeight: withRefresh(function(i, h) {
            this.state.regions[i].heightScale = h;
            this.setState({ regions: this.state.regions });
            render.getDangerZoneController().setHeight(i, h);
        }),
    
        // 切换危险区域显示状态
        toggle: withRefresh(function(i) {
            this.state.regions[i].active = !this.state.regions[i].active;
            this.setState({ regions: this.state.regions });
            render.getDangerZoneController().toggle(i);
        }),
    
        render: function() {
            var o = this;
            var buttonClass = this.state.isActive ? 'btn btn-danger btn-sm btn-block active' : 'btn btn-default btn-sm btn-block';
            
            var riskColors = {
                'low': '#28a745',    // 低风险颜色
                'medium': '#ffc107', // 中风险颜色
                'high': '#dc3545'    // 高风险颜色
            };
            
            var dangerZonesList = this.state.regions.map(function(region, index) {
                return (
                    <div key={index} className="danger-zone-item" style={{marginBottom: '5px', padding: '5px', border: '1px solid #ddd', borderRadius: '3px'}}>
                        <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
                            <span style={{color: riskColors[region.riskLevel], fontWeight: 'bold'}}>
                                {region.riskLevel === 'low' ? '低风险' : region.riskLevel === 'medium' ? '中风险' : '高风险'} 危险区域 {index + 1}
                            </span>
                            <button 
                                className="btn btn-xs btn-danger"
                                onClick={function() { o.removeDangerZone(index); }}
                                title="删除危险区域">
                                删除危险区域
                            </button>
                        </div>
                    </div>
                );
            });
            
            return (
                <div>
                    <button 
                        className={buttonClass}
                        onClick={this.toggleDangerMode}
                        style={{marginBottom: '10px'}}>
                        {this.state.isActive ? '停止危险区域标记' : '开始危险区域标记'}
                    </button>
                    
                    {this.state.isActive && (
                        <div style={{marginBottom: '10px'}}>
                            <div className="btn-group btn-group-sm" style={{width: '100%'}}>
                                <button 
                                    className={this.state.riskLevel === 'low' ? 'btn btn-success active' : 'btn btn-default'}
                                    onClick={function() { o.setRiskLevel('low'); }}
                                    style={{flex: 1}}>
                                    低风险
                                </button>
                                <button 
                                    className={this.state.riskLevel === 'medium' ? 'btn btn-warning active' : 'btn btn-default'}
                                    onClick={function() { o.setRiskLevel('medium'); }}
                                    style={{flex: 1}}>
                                    中风险
                                </button>
                                <button 
                                    className={this.state.riskLevel === 'high' ? 'btn btn-danger active' : 'btn btn-default'}
                                    onClick={function() { o.setRiskLevel('high'); }}
                                    style={{flex: 1}}>
                                    高风险
                                </button>
                            </div>
                            <small className="text-muted" style={{display: 'block', marginTop: '5px'}}>
                                选择危险区域的风险等级，用于确定危险区域的颜色
                            </small>
                        </div>
                    )}
                    
                    {this.state.regions.length > 0 && (
                        <div>
                            <h5>危险区域列表</h5>
                            {dangerZonesList}
                            <button 
                                className="btn btn-warning btn-xs btn-block"
                                onClick={function() { 
                                    $.event.trigger({type: 'plasio.dangerzone.reset'});
                                }}
                                style={{marginTop: '10px'}}>
                                重置危险区域
                            </button>
                        </div>
                    )}
                </div>
            );
        }
    });

    var openGreyhoundPipelineButton = React.createClass({
        getInitialState: function() {
            return { show: false };
        },

        render:function() {
            var o = this;
            var close = function() { o.setState({ show: false }); };
            var open = function() { o.setState({show: true}); };

            return (
                <div className="modal-container">
                    <Button
                        bsStyle="default"
                        bsSize="small"
                        className="btn-block"
                        onClick={open}
                    >
                        Open
                    </Button>

                    <Modal
                        show={this.state.show}
                        onHide={close}
                        container={this}
                        aria-labelledby="contained-modal-title"
                    >
                        <OpenGreyhoundPipeline onRequestHide={close} />
                    </Modal>
                </div>
            );
        },
    });

    // export stuff
    scope.InundationControls = InundationControls;
    scope.LineSegmentsBox = LineSegmentsBox;
    scope.RegionsBox = RegionsBox;
    scope.openGreyhoundPipelineButton = openGreyhoundPipelineButton;

    scope.DangerZoneControls = DangerZoneControls;

})(module.exports);

// 危险区域大小滑块组件
var DangerZoneSizeSlider = React.createClass({
    componentDidMount: function() {
        var a = this.refs.sliderNode;
        $(a).noUiSlider({
            range: [1, 10],
            start: Math.round(this.props.startScale),
            step: 1,
            handles: 1,
            connect: false,
            slide: this.setSize
        });
    },
    render: function() {
        return (
            <div style={{ marginBottom: '15px' }} ref="sliderNode" />
        );
    },
    setSize: function() {
        var v = $(this.refs.sliderNode).val();
        this.props.setSize(v);
    }
});

// 单个危险区域控制组件
var DangerZoneItem = React.createClass({
    render: function() {
        var cx = classNames;
        var classesFor = function(active) {
            return cx({
                'btn': true,
                'btn-default': true,
                'btn-sm': true,
                'active': active
            });
        };
        
        var riskColors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#dc3545'
        };
        
        var riskLabels = {
            'low': '低风险',
            'medium': '中风险',
            'high': '高风险'
        };
        
        // 只有条带类型才显示大小调整滑块
        var dangerZoneControls = this.props.dangerZone.type === 1 ? (
            <div>
                <div style={{marginBottom: '5px', fontSize: '12px', color: '#666'}}>宽度调整</div>
                <DangerZoneSizeSlider
                    dangerZone={this.props.dangerZone}
                    startScale={this.props.dangerZone.widthScale}
                    setSize={_.partial(this.props.setWidth, this.props.index)} />
                <div style={{marginBottom: '5px', fontSize: '12px', color: '#666'}}>高度调整</div>
                <DangerZoneSizeSlider
                    dangerZone={this.props.dangerZone}
                    startScale={this.props.dangerZone.heightScale}
                    setSize={_.partial(this.props.setHeight, this.props.index)} />
            </div>
        ) : <div />;
        
        return (
            <div style={{
                borderLeft: '10px solid ' + riskColors[this.props.dangerZone.riskLevel],
                marginBottom: '5px',
                paddingLeft: '5px',
                boxSizing: 'border-box'
            }}>
                <div
                    className="btn btn-link btn-sm"
                    onClick={_.partial(this.props.remove, this.props.index)}
                    type="button"
                    style={{
                        float: 'right',
                        padding: '0px'
                    }}>
                    <span className="glyphicon glyphicon-remove" />
                </div>
                <div style={{marginBottom: '5px', fontWeight: 'bold'}}>
                    {riskLabels[this.props.dangerZone.riskLevel]} 危险区域 {this.props.index + 1}
                </div>
                <div
                    className="btn-group btn-group-justified"
                    style={{marginBottom: '10px'}}>
                    <div
                        className={classesFor(this.props.dangerZone.type === 1)}
                        onClick={_.partial(this.props.setRibbon, this.props.index)}
                        type="button">条带</div>
                    <div
                        className={classesFor(this.props.dangerZone.type === 2)}
                        onClick={_.partial(this.props.setAxisAligned, this.props.index)}
                        type="button">轴对齐</div>
                </div>
                {dangerZoneControls}
                <button
                    className={this.props.dangerZone.active ? 'btn btn-success btn-sm btn-block' : 'btn btn-default btn-sm btn-block'}
                    onClick={_.partial(this.props.toggle, this.props.index)}>
                    {this.props.dangerZone.active ? "禁用" : "启用"}
                </button>
            </div>
        );
    }
});
