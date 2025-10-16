<!-- frontend/src/components/RoleMessage.vue -->
<template>
  <div class="role-message" :class="[`role-${message.role}`, { typing: isTyping }]">
    <!-- 角色头像和信息 -->
    <div class="role-header">
      <div class="role-avatar" :style="{ backgroundColor: roleConfig.color }">
        {{ roleConfig.emoji }}
      </div>
      <div class="role-info">
        <div class="role-name">{{ roleConfig.name }}</div>
        <div class="role-specialty">{{ roleConfig.specialty }}</div>
        <div class="message-time">{{ formatTime(message.timestamp) }}</div>
      </div>
      <div class="confidence-badge" v-if="message.confidence">
        <span class="confidence-label">置信度</span>
        <span class="confidence-value" :class="getConfidenceClass(message.confidence)">
          {{ Math.round(message.confidence * 100) }}%
        </span>
      </div>
    </div>

    <!-- 消息内容 -->
    <div class="message-content">
      <!-- 正在输入动画 -->
      <div v-if="isTyping" class="typing-content">
        <div class="typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <span class="typing-text">{{ roleConfig.name }}正在分析...</span>
      </div>

      <!-- 实际消息内容 -->
      <div v-else class="message-text">
        <!-- 主要观点 -->
        <div v-if="message.mainPoint" class="main-point">
          <h4>🎯 主要观点</h4>
          <p>{{ message.mainPoint }}</p>
        </div>

        <!-- 详细分析 -->
        <div v-if="message.analysis" class="analysis-section">
          <h4>🔍 专业分析</h4>
          <div class="analysis-content" v-html="formatAnalysis(message.analysis)"></div>
        </div>

        <!-- 治疗建议 -->
        <div v-if="message.recommendations && message.recommendations.length > 0" class="recommendations">
          <h4>💡 治疗建议</h4>
          <ul class="recommendation-list">
            <li 
              v-for="(rec, index) in message.recommendations" 
              :key="index"
              class="recommendation-item"
              :class="getRecommendationClass(rec.priority)"
            >
              <span class="rec-priority">{{ getPriorityIcon(rec.priority) }}</span>
              <span class="rec-text">{{ rec.text }}</span>
              <span v-if="rec.evidence" class="rec-evidence">{{ rec.evidence }}</span>
            </li>
          </ul>
        </div>

        <!-- 风险评估 -->
        <div v-if="message.risks && message.risks.length > 0" class="risks-section">
          <h4>⚠️ 风险评估</h4>
          <div class="risks-list">
            <div 
              v-for="(risk, index) in message.risks" 
              :key="index"
              class="risk-item"
              :class="getRiskClass(risk.level)"
            >
              <span class="risk-level">{{ getRiskIcon(risk.level) }}</span>
              <span class="risk-description">{{ risk.description }}</span>
            </div>
          </div>
        </div>

        <!-- 支持证据 */
        <div v-if="message.evidence && message.evidence.length > 0" class="evidence-section">
          <h4>📚 支持证据</h4>
          <div class="evidence-list">
            <div 
              v-for="(evidence, index) in message.evidence" 
              :key="index"
              class="evidence-item"
            >
              <span class="evidence-type">{{ evidence.type }}</span>
              <span class="evidence-description">{{ evidence.description }}</span>
              <span v-if="evidence.level" class="evidence-level">{{ evidence.level }}</span>
            </div>
          </div>
        </div>

        <!-- 关键指标 -->
        <div v-if="message.metrics" class="metrics-section">
          <h4>📊 关键指标</h4>
          <div class="metrics-grid">
            <div 
              v-for="(metric, key) in message.metrics" 
              :key="key"
              class="metric-item"
            >
              <span class="metric-label">{{ getMetricLabel(key) }}</span>
              <span class="metric-value">{{ formatMetricValue(metric) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 消息操作 -->
    <div class="message-actions" v-if="!isTyping">
      <button 
        class="action-btn"
        @click="toggleDetails"
        :class="{ active: showDetails }"
      >
        {{ showDetails ? '收起详情' : '查看详情' }}
      </button>
      <button class="action-btn" @click="askQuestion">
        提问
      </button>
      <button class="action-btn" @click="agreeWithOpinion">
        赞同
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

// Props
const props = defineProps({
  message: {
    type: Object,
    required: true
  },
  isTyping: {
    type: Boolean,
    default: false
  }
})

// Emits
const emit = defineEmits(['ask-question', 'agree-opinion', 'toggle-details'])

// 响应式数据
const showDetails = ref(false)

// 角色配置
const roleConfigs = {
  oncologist: {
    name: '肿瘤科医生',
    specialty: '肿瘤治疗专家',
    emoji: '👨‍⚕️',
    color: '#ef4444'
  },
  radiologist: {
    name: '放射科医生',
    specialty: '影像诊断专家',
    emoji: '🔬',
    color: '#3b82f6'
  },
  nurse: {
    name: '护理专家',
    specialty: '护理与康复',
    emoji: '👩‍⚕️',
    color: '#22c55e'
  },
  psychologist: {
    name: '心理医生',
    specialty: '心理健康专家',
    emoji: '🧠',
    color: '#a855f7'
  },
  patient_advocate: {
    name: '患者代表',
    specialty: '患者权益维护',
    emoji: '🤝',
    color: '#f59e0b'
  }
}

// 计算属性
const roleConfig = computed(() => {
  return roleConfigs[props.message.role] || {
    name: '医疗专家',
    specialty: '专业医师',
    emoji: '👨‍⚕️',
    color: '#6b7280'
  }
})

// 方法
const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}

const formatAnalysis = (analysis) => {
  if (typeof analysis === 'string') {
    return analysis.replace(/\n/g, '<br>')
  }
  return analysis
}

const getConfidenceClass = (confidence) => {
  if (confidence >= 0.8) return 'high'
  if (confidence >= 0.6) return 'medium'
  return 'low'
}

const getRecommendationClass = (priority) => {
  return `priority-${priority || 'normal'}`
}

const getPriorityIcon = (priority) => {
  const icons = {
    high: '🔴',
    medium: '🟡',
    low: '🟢',
    normal: '⚪'
  }
  return icons[priority] || icons.normal
}

const getRiskClass = (level) => {
  return `risk-${level}`
}

const getRiskIcon = (level) => {
  const icons = {
    high: '🔴',
    medium: '🟡',
    low: '🟢'
  }
  return icons[level] || '⚪'
}

const getMetricLabel = (key) => {
  const labels = {
    survival_rate: '生存率',
    success_rate: '成功率',
    side_effects: '副作用',
    recovery_time: '恢复时间',
    cost: '治疗费用'
  }
  return labels[key] || key
}

const formatMetricValue = (value) => {
  if (typeof value === 'number') {
    if (value < 1) {
      return `${Math.round(value * 100)}%`
    }
    return value.toString()
  }
  return value
}

const toggleDetails = () => {
  showDetails.value = !showDetails.value
  emit('toggle-details', props.message.id)
}

const askQuestion = () => {
  emit('ask-question', props.message)
}

const agreeWithOpinion = () => {
  emit('agree-opinion', props.message)
}
</script>

<style scoped>
.role-message {
  background: white;
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border-left: 4px solid #e5e7eb;
  transition: all 0.3s ease;
}

.role-message:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.role-message.typing {
  border-left-color: #3b82f6;
  animation: pulse 2s infinite;
}

/* 角色头部 */
.role-header {
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
}

.role-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  color: white;
  margin-right: 1rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.role-info {
  flex: 1;
}

.role-name {
  font-weight: 600;
  font-size: 1rem;
  color: #1f2937;
  margin-bottom: 0.25rem;
}

.role-specialty {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.message-time {
  font-size: 0.75rem;
  color: #9ca3af;
}

.confidence-badge {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0.5rem;
  background: #f9fafb;
  border-radius: 8px;
}

.confidence-label {
  font-size: 0.75rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.confidence-value {
  font-weight: 600;
  font-size: 0.875rem;
}

.confidence-value.high { color: #22c55e; }
.confidence-value.medium { color: #f59e0b; }
.confidence-value.low { color: #ef4444; }

/* 消息内容 */
.message-content {
  line-height: 1.6;
}

.typing-content {
  display: flex;
  align-items: center;
  color: #6b7280;
  font-style: italic;
}

.typing-indicator {
  display: flex;
  margin-right: 0.5rem;
}

.typing-indicator span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #94a3b8;
  margin: 0 1px;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

.main-point {
  background: #eff6ff;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  border-left: 4px solid #3b82f6;
}

.main-point h4 {
  margin: 0 0 0.5rem 0;
  color: #1e40af;
  font-size: 0.875rem;
}

.analysis-section,
.recommendations,
.risks-section,
.evidence-section,
.metrics-section {
  margin-bottom: 1rem;
}

.analysis-section h4,
.recommendations h4,
.risks-section h4,
.evidence-section h4,
.metrics-section h4 {
  margin: 0 0 0.75rem 0;
  font-size: 0.875rem;
  color: #374151;
  display: flex;
  align-items: center;
}

.recommendation-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.recommendation-item {
  display: flex;
  align-items: flex-start;
  padding: 0.5rem;
  margin-bottom: 0.5rem;
  background: #f9fafb;
  border-radius: 6px;
}

.rec-priority {
  margin-right: 0.5rem;
  font-size: 0.75rem;
}

.rec-text {
  flex: 1;
  font-size: 0.875rem;
}

.rec-evidence {
  font-size: 0.75rem;
  color: #6b7280;
  margin-left: 0.5rem;
}

.risks-list {
  space-y: 0.5rem;
}

.risk-item {
  display: flex;
  align-items: center;
  padding: 0.5rem;
  border-radius: 6px;
}

.risk-item.risk-high { background: #fef2f2; }
.risk-item.risk-medium { background: #fffbeb; }
.risk-item.risk-low { background: #f0fdf4; }

.risk-level {
  margin-right: 0.5rem;
}

.evidence-list {
  space-y: 0.5rem;
}

.evidence-item {
  display: flex;
  align-items: center;
  padding: 0.5rem;
  background: #f9fafb;
  border-radius: 6px;
  font-size: 0.875rem;
}

.evidence-type {
  font-weight: 500;
  margin-right: 0.5rem;
  color: #374151;
}

.evidence-description {
  flex: 1;
}

.evidence-level {
  font-size: 0.75rem;
  color: #6b7280;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.5rem;
}

.metric-item {
  display: flex;
  flex-direction: column;
  padding: 0.75rem;
  background: #f9fafb;
  border-radius: 6px;
  text-align: center;
}

.metric-label {
  font-size: 0.75rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.metric-value {
  font-weight: 600;
  font-size: 1rem;
  color: #1f2937;
}

/* 消息操作 */
.message-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.action-btn {
  padding: 0.5rem 1rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: white;
  color: #374151;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-btn:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}

.action-btn.active {
  background: #3b82f6;
  color: white;
  border-color: #3b82f6;
}

/* 动画 */
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

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.8;
  }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .role-message {
    padding: 0.75rem;
  }
  
  .role-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .role-avatar {
    margin-bottom: 0.5rem;
    margin-right: 0;
  }
  
  .confidence-badge {
    align-self: flex-end;
    margin-top: -2rem;
  }
  
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .message-actions {
    flex-wrap: wrap;
  }
}
</style>