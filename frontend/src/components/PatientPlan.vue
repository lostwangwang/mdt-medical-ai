<!-- frontend/src/components/PatientPlan.vue -->
<template>
  <div class="patient-plan">
    <div class="plan-header">
      <div class="plan-icon">📋</div>
      <div class="plan-title">
        <h2>患者治疗方案</h2>
        <div class="plan-meta">
          <span class="patient-info">{{ plan?.patient?.name || '患者姓名' }} | {{ plan?.patient?.id || 'P2024001' }}</span>
          <span class="plan-date">制定时间: {{ formatDate(new Date()) }}</span>
        </div>
      </div>
      <div class="plan-status approved">
        已批准
      </div>
    </div>

    <div class="plan-content">
      <!-- 患者基本信息 -->
      <div class="section patient-info-section">
        <h3>👤 患者信息</h3>
        <div class="info-grid">
          <div class="info-item">
            <span class="label">姓名:</span>
            <span class="value">{{ plan?.patient?.name || '张女士' }}</span>
          </div>
          <div class="info-item">
            <span class="label">年龄:</span>
            <span class="value">{{ plan?.patient?.age || '45岁' }}</span>
          </div>
          <div class="info-item">
            <span class="label">性别:</span>
            <span class="value">{{ plan?.patient?.gender || '女性' }}</span>
          </div>
          <div class="info-item">
            <span class="label">诊断:</span>
            <span class="value">{{ plan?.patient?.diagnosis || '疑似恶性肿瘤' }}</span>
          </div>
        </div>
      </div>

      <!-- 推荐治疗方案 -->
      <div class="section treatment-section">
        <h3>🏥 推荐治疗方案</h3>
        <div class="treatment-plan">
          <div class="primary-treatment">
            <h4>主要治疗</h4>
            <p>{{ plan?.treatment?.primary || '手术治疗' }}</p>
          </div>
          <div class="secondary-treatments" v-if="plan?.treatment?.secondary">
            <h4>辅助治疗</h4>
            <ul>
              <li v-for="treatment in plan.treatment.secondary" :key="treatment">{{ treatment }}</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 用药方案 -->
      <div class="section medication-section" v-if="plan?.medications">
        <h3>💊 用药方案</h3>
        <div class="medication-list">
          <div v-for="med in plan.medications" :key="med.name" class="medication-item">
            <div class="med-name">{{ med.name }}</div>
            <div class="med-details">
              <span>剂量: {{ med.dosage }}</span>
              <span>频次: {{ med.frequency }}</span>
              <span>疗程: {{ med.duration }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 治疗时间线 -->
      <div class="section timeline-section" v-if="plan?.timeline">
        <h3>📅 治疗时间线</h3>
        <div class="timeline">
          <div v-for="phase in plan.timeline" :key="phase.phase" class="timeline-item">
            <div class="timeline-marker"></div>
            <div class="timeline-content">
              <h4>{{ phase.phase }}</h4>
              <p>持续时间: {{ phase.duration }}</p>
              <ul>
                <li v-for="activity in phase.activities" :key="activity">{{ activity }}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <!-- 预期结果与风险 -->
      <div class="section outcomes-section" v-if="plan?.outcomes">
        <h3>📊 预期结果与风险</h3>
        <div class="outcomes-grid">
          <div class="expected-outcomes">
            <h4>预期结果</h4>
            <ul>
              <li v-for="outcome in plan.outcomes.expected" :key="outcome">{{ outcome }}</li>
            </ul>
          </div>
          <div class="risks">
            <h4>潜在风险</h4>
            <ul>
              <li v-for="risk in plan.outcomes.risks" :key="risk">{{ risk }}</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 随访计划 -->
      <div class="section followup-section" v-if="plan?.followUp">
        <h3>📞 随访计划</h3>
        <div class="followup-plan">
          <div class="followup-schedule">
            <h4>随访时间</h4>
            <p>{{ plan.followUp.schedule }}</p>
          </div>
          <div class="followup-tests">
            <h4>检查项目</h4>
            <ul>
              <li v-for="test in plan.followUp.tests" :key="test">{{ test }}</li>
            </ul>
          </div>
          <div class="emergency-contact">
            <h4>紧急联系</h4>
            <p>{{ plan.followUp.contact }}</p>
          </div>
        </div>
      </div>

      <!-- 专家团队 -->
      <div class="section team-section" v-if="plan?.team">
        <h3>👥 专家团队</h3>
        <div class="team-grid">
          <div v-for="member in plan.team" :key="member.name" class="expert-card">
            <div class="expert-avatar">👨‍⚕️</div>
            <div class="expert-info">
              <h4>{{ member.name }}</h4>
              <p>{{ member.role }}</p>
              <span class="contact">{{ member.contact }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 操作按钮 -->
    <div class="plan-actions">
      <button class="action-btn primary" @click="approvePlan">
        批准方案
      </button>
      <button class="action-btn secondary" @click="requestModification">
        申请修改
      </button>
      <button class="action-btn secondary" @click="exportPlan">
        导出方案
      </button>
      <button class="action-btn secondary" @click="printPlan">
        打印方案
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

// Props
const props = defineProps({
  plan: {
    type: Object,
    required: true
  }
})

// Emits
const emit = defineEmits(['approve-plan', 'request-modification', 'export-plan', 'print-plan'])

// 方法
const formatDate = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 操作方法
const approvePlan = () => {
  emit('approve-plan', props.plan)
}

const requestModification = () => {
  emit('request-modification', props.plan)
}

const exportPlan = () => {
  emit('export-plan', props.plan)
}

const printPlan = () => {
  emit('print-plan', props.plan)
}
</script>

<style scoped>
.patient-plan {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  margin: 1rem 0;
}

/* 头部 */
.plan-header {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
  padding: 2rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
}

.plan-icon {
  font-size: 3rem;
}

.plan-title {
  flex: 1;
}

.plan-title h2 {
  margin: 0 0 0.5rem 0;
  font-size: 1.8rem;
  font-weight: 600;
}

.plan-meta {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  opacity: 0.9;
}

.plan-status {
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.9rem;
}

.plan-status.approved {
  background: rgba(34, 197, 94, 0.2);
  color: #16a34a;
}

/* 内容区域 */
.plan-content {
  padding: 2rem;
}

.section {
  margin-bottom: 2rem;
  padding: 1.5rem;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #fafafa;
}

.section h3 {
  margin: 0 0 1rem 0;
  color: #1f2937;
  font-size: 1.2rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.info-item {
  display: flex;
  justify-content: space-between;
  padding: 0.75rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.label {
  font-weight: 600;
  color: #6b7280;
}

.value {
  color: #1f2937;
}

.treatment-plan {
  display: grid;
  gap: 1rem;
}

.primary-treatment, .secondary-treatments {
  padding: 1rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.primary-treatment h4, .secondary-treatments h4 {
  margin: 0 0 0.5rem 0;
  color: #1f2937;
}

.medication-list {
  display: grid;
  gap: 1rem;
}

.medication-item {
  padding: 1rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.med-name {
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 0.5rem;
}

.med-details {
  display: flex;
  gap: 1rem;
  font-size: 0.9rem;
  color: #6b7280;
}

.timeline {
  position: relative;
}

.timeline-item {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  position: relative;
}

.timeline-marker {
  width: 12px;
  height: 12px;
  background: #3b82f6;
  border-radius: 50%;
  margin-top: 0.25rem;
  flex-shrink: 0;
}

.timeline-content {
  flex: 1;
  padding: 0.5rem 0;
}

.timeline-content h4 {
  margin: 0 0 0.25rem 0;
  color: #1f2937;
}

.outcomes-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.expected-outcomes, .risks {
  padding: 1rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.followup-plan {
  display: grid;
  gap: 1rem;
}

.followup-schedule, .followup-tests, .emergency-contact {
  padding: 1rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.expert-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.expert-avatar {
  font-size: 2rem;
}

.expert-info h4 {
  margin: 0 0 0.25rem 0;
  color: #1f2937;
}

.expert-info p {
  margin: 0 0 0.25rem 0;
  color: #6b7280;
  font-size: 0.9rem;
}

.contact {
  font-size: 0.8rem;
  color: #9ca3af;
}

/* 操作按钮 */
.plan-actions {
  display: flex;
  gap: 1rem;
  padding: 1.5rem 2rem;
  border-top: 1px solid #e5e7eb;
  background: #f9fafb;
}

.action-btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.action-btn.primary {
  background: #3b82f6;
  color: white;
}

.action-btn.primary:hover {
  background: #2563eb;
}

.action-btn.secondary {
  background: #f3f4f6;
  color: #374151;
  border: 1px solid #d1d5db;
}

.action-btn.secondary:hover {
  background: #e5e7eb;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .plan-header {
    flex-direction: column;
    text-align: center;
    gap: 1rem;
  }
  
  .plan-content {
    padding: 1rem;
  }
  
  .info-grid {
    grid-template-columns: 1fr;
  }
  
  .outcomes-grid {
    grid-template-columns: 1fr;
  }
  
  .team-grid {
    grid-template-columns: 1fr;
  }
  
  .plan-actions {
    flex-direction: column;
  }
}
</style>