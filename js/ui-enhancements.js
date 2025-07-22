class CollapsibleSection {
  constructor(element) {
    this.element = element;
    this.header = element.querySelector('.p-head');
    this.body = element.querySelector('.p-body');
    this.isCollapsed = false;
    
    this.init();
  }
  
  init() {
    this.addToggleButton();
    this.bindEvents();
    this.loadState();
  }
  
  addToggleButton() {
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'section-toggle';
    toggleBtn.innerHTML = '<i class="icon-chevron-down"></i>';
    this.header.appendChild(toggleBtn);
  }
  
  toggle() {
    this.isCollapsed = !this.isCollapsed;
    this.element.classList.toggle('collapsed', this.isCollapsed);
    this.saveState();
  }
}