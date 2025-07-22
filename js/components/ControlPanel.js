// 创建独立的控制面板组件
class ControlPanel {
  constructor(container) {
    this.container = container;
    this.sections = new Map();
    this.init();
  }
  
  init() {
    this.createSections();
    this.bindEvents();
    this.setupCollapsible();
  }
  
  createSections() {
    const sections = ['file', 'camera', 'data', 'tools'];
    sections.forEach(section => {
      this.sections.set(section, new Section(section));
    });
  }
}