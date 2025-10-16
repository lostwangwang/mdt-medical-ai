<!-- frontend/src/components/ConsensusMessage.vue -->
<template>
  <div class="consensus-message">
    <div class="consensus-header">
      <div class="consensus-icon">🤝</div>
      <div class="consensus-title">
        <h3>团队共识</h3>
        <div class="consensus-score">
          共识度: <span :class="getScoreClass(message.consensusScore)">
            {{ Math.round(message.consensusScore * 100) }}%
          </span>
        </div>
      </div>
      <div class="consensus-timestamp">
        {{ formatTime(message.timestamp) }}
      </div>
    </div>

    <div class="consensus-content">
      <!-- 主要结论 -->
      <div class="consensus-conclusion">
        <h4>🎯 主要结论</h4>
        <p class="conclusion-text">{{ message.conclusion }}</p>
      </div>

      <!-- 推荐治疗方案 -->
      <div class="recommended-treatment" v-if="message.recommendedTreatment">
        <h4>💊 推荐治疗方案</h4>
        <div class="treatment-card">
          <div class="treatment-name">{{ message.recommendedTreatment.name }}</div>
          <div class="treatment-description">{{ message.recommendedTreatment.description }}</div>
          <div class="treatment-confidence">
            推荐强度: 
            <span :class="getConfidenceClass(message.recommendedTreatment.confidence)">
              {{ getConfidenceText(message.recommendedTreatment.confidence) }}
            </span>
          </div>
        </div>
      </div>

      <!-- 专家意见汇总 -->
      <div class="expert-opinions" v-if="message.expertOpinions">
        <h4>👥 专家意见汇总</h4>
        <div class="opinions-grid">
          <div 
            v-for="(opinion, role) in message.expertOpinions" 
            :key="role"
            class="opinion-card"
          >
            <div class="opinion-header">
              <span class="role-emoji">{{ getRoleEmoji(role) }}</span>
              <span class="role-name">{{ getRoleName(role) }}</span>
              <span class="agreement-level" :class="getAgreementClass(opinion.agreement)">
                {{ getAgreementText(opinion.agreement) }}
              </span>
            </div>
            <div class="opinion-summary">{{ opinion.summary }}</div>
          </div>
        </div>
      </div>

      <!-- 关键考虑因素 -->
      <div class="key-factors" v-if="message.keyFactors && message.keyFactors.length > 0">
        <h4>🔑 关键考虑因素</h4>
        <ul class="factors-list">
          <li v-for="(factor, index) in message.keyFactors" :key="index" class="factor-item">
            <span class="factor-icon">{{ getFactorIcon(factor.type) }}</span>
            <span class="factor-text">{{ factor.description }}</span>
            <span v-if="factor.impact" class="factor-impact" :class="getImpactClass(factor.impact)">
              {{ getImpactText(factor.impact) }}
            </span>
          </li>
        </ul>
      </div>

      <!-- 风险与收益分析 -->
      <div class="risk-benefit" v-if="message.riskBenefit">
        <h4>⚖️ 风险与收益分析</h4>
        <div class="risk-benefit-grid">
          <div class="benefit-section">
            <h5>✅ 预期收益</h5>
            <ul>
              <li v-for="(benefit, index) in message.riskBenefit.benefits" :key="index">
                {{ benefit }}
              </li>
            </ul>
          </div>
          <div class="risk-section">
            <h5>⚠️ 潜在风险</h5>
            <ul>
              <li v-for="(risk, index) in message.riskBenefit.risks" :key="index">
                {{ risk }}
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 后续建议 -->
      <div class="next-steps" v-if="message.nextSteps && message.nextSteps.length > 0">
        <h4>📋 后续建议</h4>
        <ol class="steps-list">
          <li v-for="(step, index) in message.nextSteps" :key="index" class="step-item">
            <span class="step-number">{{ index + 1 }}</span>
            <span class="step-text">{{ step.description }}</span>
            <span v-if="step.timeline" class="step-timeline">{{ step.timeline }}</span>
          </li>
        </ol>
      </div>

      <!-- 共识统计 -->
      <div class="consensus-stats" v-if="message.stats">
        <h4>📊 共识统计</h4>
        <div class="stats-grid">
          <div class="stat-item">
            <span class="stat-label">参与专家</span>
            <span class="stat-value">{{ message.stats.totalExperts }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">完全同意</span>
            <span class="stat-value">{{ message.stats.fullAgreement }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">部分同意</span>
            <span class="stat-value">{{ message.stats.partialAgreement }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">不同意</span>
            <span class="stat-value">{{ message.stats.disagreement }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 操作按钮 -->
    <div class="consensus-actions">
      <button class="action-btn primary" @click="acceptConsensus">
        接受建议
      </button>
      <button class="action-btn secondary" @click="requestMoreInfo">
        需要更多信息
      </button>
      <button class="action-btn secondary" @click="seekSecondOpinion">
        寻求第二意见
      </button>
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

// Emits
const emit = defineEmits(['accept-consensus', 'request-more-info', 'seek-second-opinion'])

// 方法
const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}

const getScoreClass = (score) => {
  if (score >= 0.8) return 'high'
  if (score >= 0.6) return 'medium'
  return 'low'
}

const getConfidenceClass = (confidence) => {
  if (confidence >= 0.8) return 'high'
  if (confidence >= 0.6) return 'medium'
  return 'low'
}

const getConfidenceText = (confidence) => {
  if (confidence >= 0.8) return '强烈推荐'
  if (confidence >= 0.6) return '推荐'
  return '谨慎推荐'
}

const getRoleEmoji = (role) => {
  const emojis = {
    oncologist: '👨‍⚕️',
    radiologist: '🔬',
    nurse: '👩‍⚕️',
    psychologist: '🧠',
    patient_advocate: '🤝'
  }
  return emojis[role] || '👨‍⚕️'
}

const getRoleName = (role) => {
  const names = {
    oncologist: '肿瘤科医生',
    radiologist: '放射科医生',
    nurse: '护理专家',
    psychologist: '心理医生',
    patient_advocate: '患者代表'
  }
  return names[role] || '医疗专家'
}

const getAgreementClass = (agreement) => {
  if (agreement >= 0.8) return 'full'
  if (agreement >= 0.5) return 'partial'
  return 'low'
}

const getAgreementText = (agreement) => {
  if (agreement >= 0.8) return '完全同意'
  if (agreement >= 0.5) return '部分同意'
  return '保留意见'
}

const getFactorIcon = (type) => {
  const icons = {
    medical: '🏥',
    patient: '👤',
    resource: '💰',
    time: '⏰',
    risk: '⚠️'
  }
  return icons[type] || '📌'
}

const getImpactClass = (impact) => {
  if (impact === 'high') return 'high-impact'
  if (impact === 'medium') return 'medium-impact'
  return 'low-impact'
}

const getImpactText = (impact) => {
  const texts = {
    high: '高影响',
    medium: '中等影响',
    low: '低影响'
  }
  return texts[impact] || impact
}

const acceptConsensus = () => {
  emit('accept-consensus', props.message)
}

const requestMoreInfo = () => {
  emit('request-more-info', props.message)
}

const seekSecondOpinion = () => {
  emit('seek-second-opinion', props.message)
}
</script>

<style scoped>
.consensus-message {
  background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
  border: 2px solid #22c55e;
  border-radius: 16px;
  padding: 1.5rem;
  margin: 1rem 0;
  box-shadow: 0 4px 16px rgba(34, 197, 94, 0.1);
}

/* 头部 */
.consensus-header {
  display: flex;
  align-items: center;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #bbf7d0;
}

.consensus-icon {
  font-size: 2rem;
  margin-right: 1rem;
}

.consensus-title {
  flex: 1;
}

.consensus-title h3 {
  margin: 0 0 0.25rem 0;
  color: #166534;
  font-size: 1.25rem;
}

.consensus-score {
  font-size: 0.875rem;
  color: #166534;
}

.consensus-score .high { color: #22c55e; font-weight: 600; }
.consensus-score .medium { color: #f59e0b; font-weight: 600; }
.consensus-score .low { color: #ef4444; font-weight: 600; }

.consensus-timestamp {
  font-size: 0.75rem;
  color: #6b7280;
}

/* 内容区域 */
.consensus-content > div {
  margin-bottom: 1.5rem;
}

.consensus-content h4 {
  margin: 0 0 0.75rem 0;
  color: #166534;
  font-size: 1rem;
  display: flex;
  align-items: center;
}

.consensus-conclusion {
  background: white;
  padding: 1rem;
  border-radius: 8px;
  border-left: 4px solid #22c55e;
}

.conclusion-text {
  margin: 0;
  font-size: 1rem;
  line-height: 1.6;
  color: #1f2937;
}

/* 推荐治疗方案 */
.treatment-card {
  background: white;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #d1fae5;
}

.treatment-name {
  font-weight: 600;
  font-size: 1.125rem;
  color: #166534;
  margin-bottom: 0.5rem;
}

.treatment-description {
  color: #374151;
  margin-bottom: 0.5rem;
  line-height: 1.5;
}

.treatment-confidence .high { color: #22c55e; }
.treatment-confidence .medium { color: #f59e0b; }
.treatment-confidence .low { color: #ef4444; }

/* 专家意见网格 */
.opinions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.opinion-card {
  background: white;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #d1fae5;
}

.opinion-header {
  display: flex;
  align-items: center;
  margin-bottom: 0.5rem;
}

.role-emoji {
  margin-right: 0.5rem;
}

.role-name {
  flex: 1;
  font-weight: 500;
  color: #374151;
}

.agreement-level {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-weight: 500;
}

.agreement-level.full {
  background: #dcfce7;
  color: #166534;
}

.agreement-level.partial {
  background: #fef3c7;
  color: #92400e;
}

.agreement-level.low {
  background: #fee2e2;
  color: #dc2626;
}

.opinion-summary {
  font-size: 0.875rem;
  color: #6b7280;
  line-height: 1.4;
}

/* 关键因素列表 */
.factors-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.factor-item {
  display: flex;
  align-items: center;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #d1fae5;
}

.factor-icon {
  margin-right: 0.75rem;
  font-size: 1rem;
}

.factor-text {
  flex: 1;
  color: #374151;
}

.factor-impact {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-weight: 500;
}

.factor-impact.high-impact {
  background: #fee2e2;
  color: #dc2626;
}

.factor-impact.medium-impact {
  background: #fef3c7;
  color: #92400e;
}

.factor-impact.low-impact {
  background: #dcfce7;
  color: #166534;
}

/* 风险收益分析 */
.risk-benefit-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.benefit-section,
.risk-section {
  background: white;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #d1fae5;
}

.benefit-section h5,
.risk-section h5 {
  margin: 0 0 0.75rem 0;
  color: #374151;
}

.benefit-section ul,
.risk-section ul {
  margin: 0;
  padding-left: 1.25rem;
}

.benefit-section li,
.risk-section li {
  margin-bottom: 0.5rem;
  color: #6b7280;
  line-height: 1.4;
}

/* 后续步骤 */
.steps-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.step-item {
  display: flex;
  align-items: flex-start;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #d1fae5;
}

.step-number {
  width: 24px;
  height: 24px;
  background: #22c55e;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
  margin-right: 0.75rem;
  flex-shrink: 0;
}

.step-text {
  flex: 1;
  color: #374151;
}

.step-timeline {
  font-size: 0.75rem;
  color: #6b7280;
  margin-left: 0.5rem;
}

/* 统计网格 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #d1fae5;
}

.stat-label {
  font-size: 0.75rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: #166534;
}

/* 操作按钮 */
.consensus-actions {
  display: flex;
  gap: 0.75rem;
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid #bbf7d0;
}

.action-btn {
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  border: none;
}

.action-btn.primary {
  background: #22c55e;
  color: white;
}

.action-btn.primary:hover {
  background: #16a34a;
}

.action-btn.secondary {
  background: white;
  color: #166534;
  border: 1px solid #22c55e;
}

.action-btn.secondary:hover {
  background: #f0fdf4;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .consensus-message {
    padding: 1rem;
  }
  
  .consensus-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .consensus-icon {
    margin-bottom: 0.5rem;
  }
  
  .opinions-grid {
    grid-template-columns: 1fr;
  }
  
  .risk-benefit-grid {
    grid-template-columns: 1fr;
  }
  
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .consensus-actions {
    flex-direction: column;
  }
}
</style>