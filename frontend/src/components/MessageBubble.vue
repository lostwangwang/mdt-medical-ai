<!-- frontend/src/components/MessageBubble.vue -->
<template>
  <div class="message-bubble" :class="[message.type, getAgentClass]">
    <!-- 用户消息 -->
    <div v-if="message.type === 'user'" class="user-message">
      <div class="message-content">{{ message.content }}</div>
      <div class="message-time">{{ formatTime(message.timestamp) }}</div>
    </div>

    <!-- 系统消息 -->
    <div v-else-if="message.type === 'system'" class="system-message">
      <div class="system-icon">ℹ️</div>
      <div class="message-content">{{ message.content }}</div>
    </div>

    <!-- 异步初始响应消息 -->
    <div v-else-if="message.type === 'async_initial'" class="async-initial-message">
      <div class="async-initial-header">
        <span class="async-icon">🤖</span>
        <span class="async-title">医疗AI助手</span>
      </div>
      <div class="message-content">{{ message.content }}</div>
      <div v-if="message.status && message.status !== 'completed'" class="status-progress">
        <div class="status-indicator">
          <div class="spinner"></div>
          <span class="status-text">{{ getStatusText(message.status) }}</span>
        </div>
      </div>
      <div class="message-time">{{ formatTime(message.timestamp) }}</div>
    </div>

    <!-- 智能体消息 -->
    <div v-else-if="message.type === 'agent'" class="agent-message">
      <div class="agent-header">
        <span class="agent-icon">{{ getAgentIcon(message.agent) }}</span>
        <span class="agent-name">{{ message.agentName }}</span>
        <span class="confidence-badge"
          >{{ (message.confidence * 100).toFixed(0) }}%</span
        >
      </div>
      <div
        class="message-content"
        v-html="formatContent(message.content)"
      ></div>
      <div
        v-if="message.recommendations && message.recommendations.length > 0"
        class="recommendations"
      >
        <div class="rec-title">💡 建议：</div>
        <ul>
          <li v-for="(rec, idx) in message.recommendations" :key="idx">
            {{ rec }}
          </li>
        </ul>
      </div>
      <div class="message-time">{{ formatTime(message.timestamp) }}</div>
    </div>

    <!-- 共识消息 -->
    <div v-else-if="message.type === 'consensus'" class="consensus-message">
      <div class="consensus-header">
        <span class="consensus-icon">🎯</span>
        <span class="consensus-title">多学科团队共识</span>
        <span class="confidence-badge high"
          >{{ (message.confidence * 100).toFixed(0) }}%</span
        >
      </div>
      <div
        class="message-content"
        v-html="formatContent(message.content)"
      ></div>
      <div class="message-time">{{ formatTime(message.timestamp) }}</div>
    </div>

    <!-- 错误消息 -->
    <div v-else-if="message.type === 'error'" class="error-message">
      <div class="error-icon">⚠️</div>
      <div class="message-content">{{ message.content }}</div>
    </div>
  </div>
</template>

<script>
import { marked } from 'marked';
import hljs from 'highlight.js';

export default {
  name: "MessageBubble",
  props: {
    message: {
      type: Object,
      required: true,
    },
  },
  computed: {
    getAgentClass() {
      return this.message.agent ? `agent-${this.message.agent}` : "";
    },
  },
  methods: {
    getAgentIcon(agent) {
      const icons = {
        oncologist: "🩺",
        radiologist: "🔍",
        nurse: "👨‍⚕️",
        patient_advocate: "👨⚖️",
        psychologist: "🧠",
      };
      return icons[agent] || "🤖";
    },
    formatTime(timestamp) {
      if (!timestamp) return "";
      const date = new Date(timestamp);
      return date.toLocaleTimeString("zh-CN", {
        hour: "2-digit",
        minute: "2-digit",
      });
    },
    formatContent(content) {
      if (!content) return "";
      
      // 配置marked选项
      marked.setOptions({
        highlight: function(code, lang) {
          if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
          }
          return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true,
        sanitize: false,
      });
      
      return marked(content);
    },
    getStatusText(status) {
      if (status.includes('%')) {
        return `分析进度: ${status}`;
      }
      
      const statusMap = {
        'processing': '正在处理您的请求...',
        'analyzing': '专家团队正在分析...',
        'completed': '分析已完成',
        'error': '处理出错'
      };
      
      return statusMap[status] || `状态: ${status}`;
    }
  },
};
</script>

<style scoped>
.message-bubble {
  max-width: 80%;
  animation: fadeIn 0.3s ease-out;
  margin-bottom: 10px;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.user-message {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 1.25rem;
  border-radius: 18px 18px 4px 18px;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
  word-wrap: break-word;
}

.system-message {
  background: #f8f9fa;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  border-left: 3px solid #667eea;
  max-width: 90%;
  margin: 0 auto;
  color: #666;
}

.system-icon {
  font-size: 1.25rem;
}

/* 异步初始响应消息样式 */
.async-initial-message {
  background: #f8f0ff;
  border-radius: 18px 18px 18px 4px;
  overflow: hidden;
  box-shadow: 0 2px 12px rgba(138, 43, 226, 0.1);
  border: 1px solid #e9d5ff;
  transition: all 0.3s ease;
}

.async-initial-message:hover {
  box-shadow: 0 4px 16px rgba(138, 43, 226, 0.15);
}

.async-initial-header {
  background: linear-gradient(90deg, #f8f0ff 0%, #e9d5ff 100%);
  padding: 0.75rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  border-bottom: 1px solid #e9d5ff;
}

.async-icon {
  font-size: 1.5rem;
}

.async-title {
  font-weight: 600;
  color: #6b21a8;
  flex: 1;
}

/* 状态进度指示器样式 */
.status-progress {
  padding: 1rem 1.25rem;
  background: rgba(138, 43, 226, 0.05);
  margin: 0 1.25rem;
  border-radius: 8px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: #6b21a8;
  font-size: 0.9rem;
  font-weight: 500;
}

.spinner {
  border: 2px solid rgba(138, 43, 226, 0.3);
  border-top: 2px solid #8a2be2;
  border-radius: 50%;
  width: 16px;
  height: 16px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.agent-message {
  background: white;
  border-radius: 18px 18px 18px 4px;
  overflow: hidden;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  border: 1px solid #e9ecef;
  transition: all 0.3s ease;
}

.agent-message:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.agent-header {
  background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 0.75rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  border-bottom: 1px solid #dee2e6;
}

.agent-icon {
  font-size: 1.5rem;
}

.agent-name {
  font-weight: 600;
  color: #2c3e50;
  flex: 1;
}

.confidence-badge {
  background: #667eea;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
}

.confidence-badge.high {
  background: #2ecc71;
}

.message-content {
  padding: 1rem 1.25rem;
  color: #2c3e50;
  line-height: 1.6;
  word-wrap: break-word;
}

.message-content :deep(h1),
.message-content :deep(h2),
.message-content :deep(h3) {
  margin-top: 1rem;
  margin-bottom: 0.5rem;
  color: #2c3e50;
}

.message-content :deep(h1) {
  font-size: 1.4rem;
  border-bottom: 2px solid #e9ecef;
  padding-bottom: 0.3rem;
}

.message-content :deep(h2) {
  font-size: 1.2rem;
}

.message-content :deep(h3) {
  font-size: 1.1rem;
}

.message-content :deep(p) {
  margin-bottom: 0.75rem;
}

.message-content :deep(ul),
.message-content :deep(ol) {
  margin-left: 1.5rem;
  margin-bottom: 0.75rem;
}

.message-content :deep(li) {
  margin-bottom: 0.25rem;
}

.message-content :deep(strong) {
  font-weight: 600;
  color: #2c3e50;
}

.message-content :deep(em) {
  font-style: italic;
}

.message-content :deep(code) {
  background-color: #f1f1f1;
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.9em;
}

.message-content :deep(pre) {
  background-color: #2d2d2d;
  color: #f8f8f2;
  padding: 1rem;
  border-radius: 8px;
  overflow-x: auto;
  margin-bottom: 0.75rem;
}

.message-content :deep(pre code) {
  background-color: transparent;
  padding: 0;
  color: inherit;
}

/* 代码高亮样式 */
.message-content :deep(.hljs-comment),
.message-content :deep(.hljs-quote) {
  color: #6a9955;
}

.message-content :deep(.hljs-keyword),
.message-content :deep(.hljs-selector-tag),
.message-content :deep(.hljs-literal),
.message-content :deep(.hljs-section),
.message-content :deep(.hljs-link) {
  color: #569cd6;
}

.message-content :deep(.hljs-name),
.message-content :deep(.hljs-tag) {
  color: #808080;
}

.message-content :deep(.hljs-string),
.message-content :deep(.hljs-title),
.message-content :deep(.hljs-attribute),
.message-content :deep(.hljs-symbol),
.message-content :deep(.hljs-bullet),
.message-content :deep(.hljs-addition),
.message-content :deep(.hljs-variable),
.message-content :deep(.hljs-template-tag),
.message-content :deep(.hljs-template-variable) {
  color: #ce9178;
}

.message-content :deep(.hljs-deletion),
.message-content :deep(.hljs-selector-attr),
.message-content :deep(.hljs-selector-pseudo),
.message-content :deep(.hljs-meta) {
  color: #d7ba7d;
}

.message-content :deep(.hljs-number),
.message-content :deep(.hljs-regexp),
.message-content :deep(.hljs-literal) {
  color: #b5cea8;
}

.message-content :deep(.hljs-title.class_),
.message-content :deep(.hljs-class .hljs-title) {
  color: #4ec9b0;
}

.message-content :deep(.bullet) {
  color: #667eea;
  margin-right: 0.5rem;
}

.recommendations {
  padding: 0 1.25rem 1rem;
}

.rec-title {
  font-weight: 600;
  color: #667eea;
  margin-bottom: 0.5rem;
}

.recommendations ul {
  list-style: none;
  padding-left: 0;
}

.recommendations li {
  padding: 0.5rem 0;
  padding-left: 1.5rem;
  position: relative;
  color: #495057;
  transition: color 0.3s ease;
}

.recommendations li:before {
  content: "✓";
  position: absolute;
  left: 0;
  color: #2ecc71;
  font-weight: bold;
}

.recommendations li:hover {
  color: #2c3e50;
}

.message-time {
  padding: 0.5rem 1.25rem;
  font-size: 0.75rem;
  color: #6c757d;
  text-align: right;
}

.consensus-message {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
  transition: all 0.3s ease;
}

.consensus-message:hover {
  box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
}

.consensus-header {
  background: rgba(255, 255, 255, 0.2);
  padding: 1rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.consensus-icon {
  font-size: 1.5rem;
}

.consensus-title {
  font-weight: 600;
  flex: 1;
}

.error-message {
  background: #fee;
  color: #c33;
  padding: 1rem 1.25rem;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  border-left: 3px solid #c33;
  max-width: 90%;
  margin: 0 auto;
}

.error-icon {
  font-size: 1.25rem;
}

/* 智能体特定颜色 */
.agent-oncologist .agent-header {
  background: linear-gradient(90deg, #d1fae5 0%, #a7f3d0 100%);
}

.agent-radiologist .agent-header {
  background: linear-gradient(90deg, #dbeafe 0%, #bfdbfe 100%);
}

.agent-nurse .agent-header {
  background: linear-gradient(90deg, #ffe4e6 0%, #fecdd3 100%);
}

.agent-patient_advocate .agent-header {
  background: linear-gradient(90deg, #fef9c3 0%, #fef08a 100%);
}

.agent-psychologist .agent-header {
  background: linear-gradient(90deg, #f3e8ff 0%, #e9d5ff 100%);
}

/* 移动端适配 */
@media (max-width: 768px) {
  .message-bubble {
    max-width: 90%;
  }
  
  .async-initial-header,
  .agent-header,
  .consensus-header {
    padding: 0.5rem 1rem;
  }
  
  .message-content {
    padding: 0.75rem 1rem;
  }
  
  .status-progress {
    padding: 0.75rem 1rem;
    margin: 0 1rem;
  }
}
</style>
