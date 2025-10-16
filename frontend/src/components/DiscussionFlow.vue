<!-- frontend/src/components/DiscussionFlow.vue -->
<template>
  <div class="discussion-flow">
    <!-- 讨论阶段指示器 -->
    <div class="discussion-stages">
      <div 
        v-for="(stage, index) in discussionStages" 
        :key="stage.id"
        class="stage-indicator"
        :class="{ 
          active: currentStage === stage.id, 
          completed: completedStages.includes(stage.id) 
        }"
      >
        <div class="stage-icon">{{ stage.icon }}</div>
        <div class="stage-label">{{ stage.label }}</div>
      </div>
    </div>

    <!-- 讨论消息流 -->
    <div class="discussion-messages" ref="messagesContainer">
      <TransitionGroup name="discussion-message" tag="div">
        <div 
          v-for="message in discussionMessages" 
          :key="message.id"
          class="discussion-message-wrapper"
        >
          <!-- 角色消息 -->
          <RoleMessage 
            v-if="message.type === 'role'"
            :message="message"
            :is-typing="message.isTyping"
          />
          
          <!-- 系统消息 -->
          <SystemMessage 
            v-else-if="message.type === 'system'"
            :message="message"
          />
          
          <!-- 共识消息 -->
          <ConsensusMessage 
            v-else-if="message.type === 'consensus'"
            :message="message"
          />
        </div>
      </TransitionGroup>

      <!-- 正在输入指示器 -->
      <div v-if="typingRoles.length > 0" class="typing-indicators">
        <div 
          v-for="role in typingRoles" 
          :key="role.id"
          class="typing-indicator"
        >
          <div class="role-avatar" :style="{ backgroundColor: role.color }">
            {{ role.emoji }}
          </div>
          <div class="typing-animation">
            <span>{{ role.name }}正在思考</span>
            <div class="typing-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 讨论进度 -->
    <div class="discussion-progress" v-if="showProgress">
      <div class="progress-header">
        <h4>讨论进度</h4>
        <span class="progress-percentage">{{ progressPercentage }}%</span>
      </div>
      <div class="progress-bar">
        <div 
          class="progress-fill" 
          :style="{ width: progressPercentage + '%' }"
        ></div>
      </div>
      <div class="progress-details">
        <span>已完成 {{ completedStages.length }}/{{ discussionStages.length }} 个阶段</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import RoleMessage from './RoleMessage.vue'
import SystemMessage from './SystemMessage.vue'
import ConsensusMessage from './ConsensusMessage.vue'

// Props
const props = defineProps({
  discussionMessages: {
    type: Array,
    default: () => []
  },
  currentStage: {
    type: String,
    default: 'initial'
  },
  completedStages: {
    type: Array,
    default: () => []
  },
  typingRoles: {
    type: Array,
    default: () => []
  },
  showProgress: {
    type: Boolean,
    default: true
  }
})

// Emits
const emit = defineEmits(['stage-completed', 'message-sent'])

// 引用
const messagesContainer = ref(null)

// 讨论阶段定义
const discussionStages = ref([
  { id: 'initial', label: '问题分析', icon: '🔍' },
  { id: 'individual', label: '专家分析', icon: '👥' },
  { id: 'discussion', label: '团队讨论', icon: '💬' },
  { id: 'consensus', label: '达成共识', icon: '🤝' },
  { id: 'recommendation', label: '治疗建议', icon: '📋' }
])

// 计算属性
const progressPercentage = computed(() => {
  if (discussionStages.value.length === 0) return 0
  return Math.round((props.completedStages.length / discussionStages.value.length) * 100)
})

// 方法
const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// 监听消息变化，自动滚动
watch(
  () => props.discussionMessages,
  () => {
    scrollToBottom()
  },
  { deep: true }
)

// 监听正在输入状态变化
watch(
  () => props.typingRoles,
  () => {
    scrollToBottom()
  },
  { deep: true }
)

// 导出方法
defineExpose({
  scrollToBottom
})
</script>

<style scoped>
.discussion-flow {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #f8fafc;
}

/* 讨论阶段指示器 */
.discussion-stages {
  display: flex;
  justify-content: space-between;
  padding: 1rem;
  background: white;
  border-bottom: 1px solid #e2e8f0;
  margin-bottom: 1rem;
}

.stage-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0.5rem;
  border-radius: 8px;
  transition: all 0.3s ease;
  opacity: 0.6;
}

.stage-indicator.active {
  opacity: 1;
  background: #eff6ff;
  border: 2px solid #3b82f6;
}

.stage-indicator.completed {
  opacity: 1;
  background: #f0fdf4;
  border: 2px solid #22c55e;
}

.stage-icon {
  font-size: 1.5rem;
  margin-bottom: 0.25rem;
}

.stage-label {
  font-size: 0.75rem;
  font-weight: 500;
  text-align: center;
  color: #64748b;
}

.stage-indicator.active .stage-label {
  color: #3b82f6;
}

.stage-indicator.completed .stage-label {
  color: #22c55e;
}

/* 讨论消息流 */
.discussion-messages {
  flex: 1;
  overflow-y: auto;
  padding: 0 1rem;
  scroll-behavior: smooth;
}

.discussion-message-wrapper {
  margin-bottom: 1rem;
}

/* 正在输入指示器 */
.typing-indicators {
  padding: 1rem 0;
}

.typing-indicator {
  display: flex;
  align-items: center;
  margin-bottom: 0.5rem;
  padding: 0.75rem;
  background: white;
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.role-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  margin-right: 0.75rem;
  color: white;
}

.typing-animation {
  display: flex;
  align-items: center;
  color: #64748b;
  font-size: 0.875rem;
}

.typing-dots {
  display: flex;
  margin-left: 0.5rem;
}

.typing-dots span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #94a3b8;
  margin: 0 1px;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% {
    transform: scale(0);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

/* 讨论进度 */
.discussion-progress {
  padding: 1rem;
  background: white;
  border-top: 1px solid #e2e8f0;
  margin-top: auto;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.progress-header h4 {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
}

.progress-percentage {
  font-size: 0.875rem;
  font-weight: 600;
  color: #3b82f6;
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #1d4ed8);
  border-radius: 3px;
  transition: width 0.5s ease;
}

.progress-details {
  font-size: 0.75rem;
  color: #6b7280;
}

/* 动画 */
.discussion-message-enter-active {
  transition: all 0.3s ease;
}

.discussion-message-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.discussion-message-enter-to {
  opacity: 1;
  transform: translateY(0);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .discussion-stages {
    padding: 0.5rem;
    overflow-x: auto;
  }
  
  .stage-indicator {
    min-width: 60px;
    padding: 0.25rem;
  }
  
  .stage-icon {
    font-size: 1.25rem;
  }
  
  .stage-label {
    font-size: 0.625rem;
  }
}
</style>