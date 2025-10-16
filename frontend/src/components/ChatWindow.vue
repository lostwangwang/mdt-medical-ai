<!-- frontend/src/components/ChatWindow.vue -->
<template>
  <div class="chat-window" ref="chatContainer">
    <!-- 消息历史 -->
    <div class="messages-container" ref="messagesContainer">
      <!-- 欢迎消息 -->
      <div v-if="messages.length === 0" class="welcome-message">
        <div class="welcome-icon">👨‍⚕️</div>
        <h2>您好，我是医疗AI助手</h2>
        <p>请输入您的健康问题或症状描述，我将为您提供专业的医疗建议。</p>
        <div class="welcome-tips">
          <div class="tip-item">💡 支持多种医疗专家意见</div>
          <div class="tip-item">📝 支持Markdown格式内容</div>
          <div class="tip-item">🏥 多智能体协作诊断</div>
        </div>
      </div>

      <!-- 消息列表 -->
      <TransitionGroup name="message" tag="div" class="message-list">
        <MessageBubble
          v-for="(message, index) in messages"
          :key="message.id || index"
          :message="message"
          :showRecommendations="showRecommendations"
          @select-recommendation="handleSelectRecommendation"
        />
      </TransitionGroup>

      <!-- 加载指示器 -->
      <div v-if="isLoading" class="loading-indicator">
        <div class="loading-bars">
          <div class="loading-bar"></div>
          <div class="loading-bar"></div>
          <div class="loading-bar"></div>
        </div>
        <p>医疗团队正在分析您的问题...</p>
        <div class="loading-steps">
          <div class="step-item">✅ 接收您的问题</div>
          <div class="step-item">🔍 提取关键信息</div>
          <div class="step-item">👥 各专家并行分析中</div>
          <div class="step-item">🤝 生成团队共识意见</div>
        </div>
      </div>

      <!-- 滚动到顶部按钮 -->
      <button
        class="scroll-to-top"
        :class="{ visible: showScrollToTop }"
        @click="scrollToTop"
        aria-label="滚动到顶部"
        :aria-hidden="!showScrollToTop"
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path
            d="M12 19V5"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <path
            d="M5 12L12 5L19 12"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import MessageBubble from './MessageBubble.vue'

// Props
const props = defineProps({
  messages: {
    type: Array,
    required: true,
    default: () => []
  },
  showRecommendations: {
    type: Boolean,
    default: true
  },
  isLoading: {
    type: Boolean,
    default: false
  }
})

// Emits
const emit = defineEmits(['select-recommendation'])

// 引用
const chatContainer = ref(null)
const messagesContainer = ref(null)

// 响应式数据
const showScrollToTop = ref(false)
const scrollTimeout = ref(null)

// 处理推荐选择
const handleSelectRecommendation = (recommendation) => {
  emit('select-recommendation', recommendation)
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// 滚动到顶部
const scrollToTop = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTo({
      top: 0,
      behavior: 'smooth'
    })
  }
}

// 处理滚动事件
const handleScroll = () => {
  if (scrollTimeout.value) {
    clearTimeout(scrollTimeout.value)
  }

  scrollTimeout.value = setTimeout(() => {
    if (messagesContainer.value) {
      // 当滚动位置大于300px时显示回到顶部按钮
      showScrollToTop.value = messagesContainer.value.scrollTop > 300
    }
  }, 100)
}

// 监听消息变化，自动滚动到底部
watch(
  () => props.messages,
  () => {
    if (props.messages.length > 0) {
      scrollToBottom()
    }
  },
  { deep: true }
)

// 监听加载状态变化，自动滚动到底部
watch(
  () => props.isLoading,
  (newVal) => {
    if (newVal) {
      // 开始加载时滚动到底部
      scrollToBottom()
    }
  }
)

// 生命周期钩子
onMounted(() => {
  if (messagesContainer.value) {
    messagesContainer.value.addEventListener('scroll', handleScroll)
    // 初始滚动到底部
    scrollToBottom()
  }
})

onUnmounted(() => {
  if (messagesContainer.value) {
    messagesContainer.value.removeEventListener('scroll', handleScroll)
  }
  if (scrollTimeout.value) {
    clearTimeout(scrollTimeout.value)
  }
})

// 导出方法供父组件使用
defineExpose({
  scrollToBottom,
  scrollToTop
})
</script>

<style scoped>
.chat-window {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #f8fafc;
  position: relative;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  scroll-behavior: smooth;
  height: 100%;
  min-height: 0;
  position: relative;
}

/* 欢迎消息样式 */
.welcome-message {
  text-align: center;
  max-width: 500px;
  margin: 4rem auto;
  padding: 2.5rem;
  background: white;
  border-radius: 16px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(0, 0, 0, 0.05);
  transition: transform 0.3s, box-shadow 0.3s;
}

.welcome-message:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08);
}

.welcome-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  opacity: 0.9;
}

.welcome-message h2 {
  margin: 0 0 1rem;
  color: #1e293b;
  font-size: 1.5rem;
  font-weight: 600;
}

.welcome-message p {
  margin: 0 0 1.5rem;
  color: #64748b;
  line-height: 1.6;
}

.welcome-tips {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  text-align: left;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
}

.tip-item {
  color: #334155;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  opacity: 0.9;
}

/* 消息列表 */
.message-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* 消息过渡动画 */
.message-enter-active,
.message-leave-active {
  transition: all 0.3s ease;
}

.message-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.message-leave-to {
  opacity: 0;
  transform: translateX(30px);
}

/* 加载指示器 */
  .loading-indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
    margin: 1rem 0;
    background: #f1f5f9;
    border-radius: 12px;
    gap: 1rem;
  }

  .loading-bars {
    display: flex;
    gap: 4px;
  }

  .loading-bar {
    width: 4px;
    height: 20px;
    background: #667eea;
    border-radius: 2px;
    animation: loading 1.4s infinite ease-in-out both;
  }

  .loading-bar:nth-child(1) {
    animation-delay: -0.32s;
  }

  .loading-bar:nth-child(2) {
    animation-delay: -0.16s;
  }

  @keyframes loading {
    0%, 80%, 100% {
      transform: scaleY(0.4);
    }
    40% {
      transform: scaleY(1);
    }
  }

  .loading-indicator p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
  }
  
  /* 加载步骤样式 */
  .loading-steps {
    margin-top: 0.5rem;
    width: 100%;
    max-width: 300px;
  }

  .step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
    font-size: 0.9rem;
    color: #64748b;
  }

/* 滚动到顶部按钮 */
.scroll-to-top {
  position: fixed;
  bottom: 80px;
  right: 20px;
  width: 44px;
  height: 44px;
  border: none;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  transition: transform 0.3s, opacity 0.3s, box-shadow 0.3s;
  opacity: 0;
  pointer-events: none;
  z-index: 100;
}

.scroll-to-top.visible {
  opacity: 1;
  pointer-events: all;
}

.scroll-to-top:hover {
  transform: scale(1.1);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
}

/* 自定义滚动条 */
.messages-container::-webkit-scrollbar {
  width: 6px;
}

.messages-container::-webkit-scrollbar-track {
  background: transparent;
}

.messages-container::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
  transition: background 0.3s;
}

.messages-container::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* 移动端适配 */
@media (max-width: 768px) {
  .messages-container {
    padding: 1rem;
  }
  
  .welcome-message {
    margin: 2rem auto;
    padding: 1.5rem;
  }
  
  .welcome-message h2 {
    font-size: 1.25rem;
  }
  
  .welcome-icon {
    font-size: 2.5rem;
  }
  
  .scroll-to-top {
    bottom: 90px;
    right: 10px;
    width: 38px;
    height: 38px;
  }
}
</style>
