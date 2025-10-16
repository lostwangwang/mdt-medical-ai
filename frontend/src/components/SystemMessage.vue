<!-- frontend/src/components/SystemMessage.vue -->
<template>
  <div class="system-message" :class="messageTypeClass">
    <div class="system-icon">
      {{ getSystemIcon() }}
    </div>
    <div class="system-content">
      <div class="system-title" v-if="message.title">
        {{ message.title }}
      </div>
      <div class="system-text">
        {{ message.content }}
      </div>
      <div class="system-details" v-if="message.details">
        <div class="detail-item" v-for="(detail, index) in message.details" :key="index">
          {{ detail }}
        </div>
      </div>
    </div>
    <div class="system-timestamp">
      {{ formatTime(message.timestamp) }}
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

// Props
const props = defineProps({
  message: {
    type: Object,
    required: true
  }
})

// 计算属性
const messageTypeClass = computed(() => {
  return `system-${props.message.subType || 'info'}`
})

// 方法
const getSystemIcon = () => {
  const icons = {
    info: 'ℹ️',
    success: '✅',
    warning: '⚠️',
    error: '❌',
    progress: '⏳',
    stage: '🔄',
    consensus: '🤝',
    analysis: '🔍'
  }
  return icons[props.message.subType] || icons.info
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}
</script>

<style scoped>
.system-message {
  display: flex;
  align-items: flex-start;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  border-radius: 8px;
  font-size: 0.875rem;
  border-left: 4px solid;
}

.system-info {
  background: #eff6ff;
  border-left-color: #3b82f6;
  color: #1e40af;
}

.system-success {
  background: #f0fdf4;
  border-left-color: #22c55e;
  color: #166534;
}

.system-warning {
  background: #fffbeb;
  border-left-color: #f59e0b;
  color: #92400e;
}

.system-error {
  background: #fef2f2;
  border-left-color: #ef4444;
  color: #dc2626;
}

.system-progress {
  background: #f3f4f6;
  border-left-color: #6b7280;
  color: #374151;
}

.system-stage {
  background: #faf5ff;
  border-left-color: #a855f7;
  color: #7c3aed;
}

.system-consensus {
  background: #ecfdf5;
  border-left-color: #10b981;
  color: #047857;
}

.system-analysis {
  background: #fef3c7;
  border-left-color: #f59e0b;
  color: #d97706;
}

.system-icon {
  margin-right: 0.75rem;
  font-size: 1rem;
  flex-shrink: 0;
}

.system-content {
  flex: 1;
}

.system-title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.system-text {
  line-height: 1.5;
}

.system-details {
  margin-top: 0.5rem;
  padding-left: 1rem;
}

.detail-item {
  margin-bottom: 0.25rem;
  position: relative;
}

.detail-item::before {
  content: '•';
  position: absolute;
  left: -0.75rem;
}

.system-timestamp {
  font-size: 0.75rem;
  opacity: 0.7;
  margin-left: 0.75rem;
  flex-shrink: 0;
}
</style>